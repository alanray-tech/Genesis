"""Top-level MPM stages used only by the whole-scene graph pipeline."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.coupling_runtime import func_collide_in_rigid_geom
from genesis.engine.mpm_runtime import backward_svd, stencil_range
from genesis.utils import array_class


@qd.func(requires_top_level=True)
def graph_compute_F_tmp_and_svd(system: qd.template(), f: qd.i32):
    # Fused F_tmp + SVD: keeps F_tmp in register/local-memory for the SVD instead of round-tripping through global
    # memory. Only used on the forward pass; the backward path continues to call compute_F_tmp / svd separately so
    # their autodiff remains composed unchanged.
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng.active[f, i_p, i_b]:
            F_tmp = (
                qd.Matrix.identity(gs.qd_float, 3) + system.substep_dt * system.particles.C[f, i_p, i_b]
            ) @ system.particles.F[f, i_p, i_b]
            system.particles.F_tmp[f, i_p, i_b] = F_tmp
            system.particles.U[f, i_p, i_b], system.particles.S[f, i_p, i_b], system.particles.V[f, i_p, i_b] = qd.svd(
                F_tmp, gs.qd_float
            )


@qd.func(requires_top_level=True)
def graph_compute_F_tmp_only(system: qd.template(), f: qd.i32):
    # Fast path when no registered material needs U/V/S: compute F_tmp without SVD. p2g then derives J from
    # det(F_tmp) directly. Only used on the forward pass.
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng.active[f, i_p, i_b]:
            system.particles.F_tmp[f, i_p, i_b] = (
                qd.Matrix.identity(gs.qd_float, 3) + system.substep_dt * system.particles.C[f, i_p, i_b]
            ) @ system.particles.F[f, i_p, i_b]


@qd.func(requires_top_level=True)
def graph_p2g(
    system: qd.template(),
    f: qd.i32,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
    cpic_active: qd.template(),
    mpm_rigid_coupling: qd.template(),
    n_rigid_geoms: qd.template(),
):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng.active[f, i_p, i_b]:
            # A. update F (deformation gradient), S (Sigma from SVD(F), essentially represents volume) and Jp
            # (volume compression ratio) based on material type.
            # det(F_tmp) == det(U S V^T) == det(S) for qd.svd (U, V are proper rotations), so when no material
            # needs U/V/S we can read J directly from F_tmp and skip the SVD kernel. Predeclared outside the
            # qd.static branch because quadrants scopes qd.static branches locally.
            J = gs.qd_float(0.0)
            if qd.static(system.material_config.needs_svd):
                J = system.particles.S[f, i_p, i_b].determinant()
            else:
                J = system.particles.F_tmp[f, i_p, i_b].determinant()
            F_new = qd.Matrix.zero(gs.qd_float, 3, 3)
            S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
            Jp_new = gs.qd_float(1.0)
            for material_idx in qd.static(system.material_config.indices):
                if system.particles_info.material_idx[i_p] == material_idx:
                    F_new, S_new, Jp_new = system.material_config.update_F_S_Jp[material_idx](
                        J=J,
                        F_tmp=system.particles.F_tmp[f, i_p, i_b],
                        U=system.particles.U[f, i_p, i_b],
                        S=system.particles.S[f, i_p, i_b],
                        V=system.particles.V[f, i_p, i_b],
                        Jp=system.particles.Jp[f, i_p, i_b],
                    )
            system.particles.F[f + 1, i_p, i_b] = F_new
            system.particles.Jp[f + 1, i_p, i_b] = Jp_new

            # B. compute stress
            # NOTE:
            # 1. Here we pass in both F_tmp and the updated F_new because in the official Quadrants example, F_new is
            # used for stress computation. However, although this works for both elastic and elasto-plastic
            # materials, it is mathematically incorrect for liquid material with non-zero viscosity (mu). In the
            # latter case, stress computation needs to be based on the F_tmp (deformation gradient before resetting
            # to identity).
            # 2. Jp is only used by Snow material, and it uses Jp from the previous frame, not the updated one.
            stress = qd.Matrix.zero(gs.qd_float, 3, 3)
            for material_idx in qd.static(system.material_config.indices):
                if system.particles_info.material_idx[i_p] == material_idx:
                    stress = system.material_config.update_stress[material_idx](
                        U=system.particles.U[f, i_p, i_b],
                        S=S_new,
                        V=system.particles.V[f, i_p, i_b],
                        F_tmp=system.particles.F_tmp[f, i_p, i_b],
                        F_new=F_new,
                        J=J,
                        Jp=system.particles.Jp[f, i_p, i_b],
                        actu=system.particles.actu[f, i_p, i_b],
                        m_dir=system.particles_info.muscle_direction[i_p],
                    )
            stress = (-system.substep_dt * system._particle_volume * 4 * system._inv_dx * system._inv_dx) * stress
            affine = stress + system.particles_info.mass[i_p] * system.particles.C[f, i_p, i_b]

            # C. project onto grid
            base = qd.floor(system.particles.pos[f, i_p, i_b] * system._inv_dx - 0.5).cast(gs.qd_int)
            fx = system.particles.pos[f, i_p, i_b] * system._inv_dx - base.cast(gs.qd_float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            for offset in qd.static(qd.grouped(stencil_range(system))):
                dpos = (offset.cast(gs.qd_float) - fx) * system._dx
                weight = gs.qd_float(1.0)
                for d in qd.static(range(3)):
                    weight *= w[offset[d]][d]

                sep_geom_idx = -1
                if qd.static(cpic_active):
                    # check if particle and cell center are at different side of any thin object
                    cell_pos = (base + offset) * system._dx

                    for i_g in range(n_rigid_geoms):
                        if geoms_info.needs_coup[i_g]:
                            sdf_normal_particle = mpm_rigid_coupling.rigid_normal[i_p, i_g, i_b]
                            sdf_normal_cell = sdf.sdf_func_normal_world(
                                i_g,
                                i_b,
                                cell_pos,
                                geoms_state,
                                geoms_info,
                                rigid_info,
                                sdf_info,
                                collider_static_config,
                            )

                            if sdf_normal_particle.dot(sdf_normal_cell) < 0:  # separated by geom i_g
                                sep_geom_idx = i_g
                                break
                    mpm_rigid_coupling.cpic_flag[i_p, offset[0], offset[1], offset[2], i_b] = sep_geom_idx
                if sep_geom_idx == -1:
                    cell_ijk = base - system._grid_offset + offset
                    system.grid.vel_in[f, cell_ijk, i_b] += weight * (
                        system.particles_info.mass[i_p] * system.particles.vel[f, i_p, i_b] + affine @ dpos
                    )
                    mass_contrib = weight * system.particles_info.mass[i_p]
                    prev_mass = qd.atomic_add(system.grid.mass[f, cell_ijk, i_b], mass_contrib)
                    # Sparse-reset bookkeeping runs forward-only: backward mode composes p2g through autodiff where
                    # these atomics are meaningless. Per-env first-writer (prev_mass == 0) tries to claim the cell
                    # in the env-shared dirty flag via atomic_or; only the very first env to touch this cell across
                    # the whole batch then appends to the global list. List size is bounded by grid_total because
                    # dedup ensures uniqueness.
                    if qd.static(not system._sim.requires_grad):
                        if prev_mass == gs.qd_float(0.0) and mass_contrib > gs.qd_float(0.0):
                            was_dirty = qd.atomic_or(system.grid_dirty_flag[cell_ijk], gs.qd_int(1))
                            if was_dirty == gs.qd_int(0):
                                slot_idx = qd.atomic_add(system.grid_dirty_count[0], 1)
                                flat = (cell_ijk[0] * system._grid_res[1] + cell_ijk[1]) * system._grid_res[2] + (
                                    cell_ijk[2]
                                )
                                system.grid_dirty_list[slot_idx] = flat

                if not system.particles_info.free[i_p]:  # non-free particles behave as boundary conditions
                    system.grid.vel_in[f, base - system._grid_offset + offset, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func(requires_top_level=True)
def graph_g2p(
    system: qd.template(),
    f: qd.i32,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    rigid_info: array_class.RigidInfo,
    cpic_active: qd.template(),
    mpm_rigid_coupling: qd.template(),
):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng.active[f, i_p, i_b]:
            base = qd.floor(system.particles.pos[f, i_p, i_b] * system._inv_dx - 0.5).cast(gs.qd_int)
            fx = system.particles.pos[f, i_p, i_b] * system._inv_dx - base.cast(gs.qd_float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_vel = qd.Vector.zero(gs.qd_float, 3)
            new_C = qd.Matrix.zero(gs.qd_float, 3, 3)
            for offset in qd.static(qd.grouped(stencil_range(system))):
                dpos = offset.cast(gs.qd_float) - fx
                grid_vel = system.grid.vel_out[f, base - system._grid_offset + offset, i_b]
                weight = gs.qd_float(1.0)
                for d in qd.static(range(3)):
                    weight *= w[offset[d]][d]

                if qd.static(cpic_active):
                    sep_geom_idx = mpm_rigid_coupling.cpic_flag[i_p, offset[0], offset[1], offset[2], i_b]
                    if sep_geom_idx != -1:
                        grid_vel = func_collide_in_rigid_geom(
                            system.particles.pos[f, i_p, i_b],
                            system.particles.vel[f, i_p, i_b],
                            system.particles_info.mass[i_p] * weight / system._particle_volume_scale,
                            mpm_rigid_coupling.rigid_normal[i_p, sep_geom_idx, i_b],
                            1.0,
                            sep_geom_idx,
                            i_b,
                            geoms_info=geoms_info,
                            links_state=links_state,
                            rigid_info=rigid_info,
                        )

                new_vel += weight * grid_vel
                new_C += 4 * system._inv_dx * weight * grid_vel.outer_product(dpos)

            # compute actual new_pos with new_vel
            new_pos = system.particles.pos[f, i_p, i_b] + system.substep_dt * new_vel

            # impose boundary for safety, in case simulation explodes and tries to access illegal cell address
            new_pos, new_vel = system.boundary.impose_pos_vel(new_pos, new_vel)

            # advect to next frame
            system.particles.vel[f + 1, i_p, i_b] = new_vel
            system.particles.C[f + 1, i_p, i_b] = new_C
            system.particles.pos[f + 1, i_p, i_b] = new_pos
        else:
            system.particles.vel[f + 1, i_p, i_b] = system.particles.vel[f, i_p, i_b]
            system.particles.pos[f + 1, i_p, i_b] = system.particles.pos[f, i_p, i_b]
            system.particles.C[f + 1, i_p, i_b] = system.particles.C[f, i_p, i_b]
            system.particles.F[f + 1, i_p, i_b] = system.particles.F[f, i_p, i_b]
            system.particles.Jp[f + 1, i_p, i_b] = system.particles.Jp[f, i_p, i_b]

        system.particles_ng.active[f + 1, i_p, i_b] = system.particles_ng.active[f, i_p, i_b]


@qd.func(requires_top_level=True)
def graph_reset_dirty_cells(system: qd.template(), f: qd.i32):
    # Zero the cells p2g touched this substep, across all envs, plus their dirty flags. The dirty list is shared
    # across envs (deduplicated via grid_dirty_flag), so threads with slot >= grid_dirty_count idle. Only one thread
    # per slot (i_b == 0) writes the flag back to zero so it can be reused next substep.
    for slot, i_b in qd.ndrange(system._grid_total, system._B):
        if slot < system.grid_dirty_count[0]:
            flat = system.grid_dirty_list[slot]
            k = flat % system._grid_res[2]
            rem = flat // system._grid_res[2]
            j = rem % system._grid_res[1]
            i = rem // system._grid_res[1]
            system.grid.mass[f, i, j, k, i_b] = gs.qd_float(0.0)
            system.grid.vel_in[f, i, j, k, i_b] = qd.Vector.zero(gs.qd_float, 3)
            system.grid.vel_out[f, i, j, k, i_b] = qd.Vector.zero(gs.qd_float, 3)
            if i_b == 0:
                system.grid_dirty_flag[i, j, k] = gs.qd_int(0)


@qd.func(requires_top_level=True)
def graph_reset_dirty_count(system: qd.template()):
    for _ in range(1):
        system.grid_dirty_count[0] = 0


@qd.func(requires_top_level=True)
def graph_apply_particle_constraints(system: qd.template(), f: qd.i32, links_state: array_class.LinksState):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particle_constraints.is_constrained[i_p, i_b]:
            # Update target position from link pose
            i_l = system.particle_constraints.link_idx[i_p, i_b]
            if i_l >= 0:
                link_pos = links_state.pos[i_l, i_b]
                link_quat = links_state.quat[i_l, i_b]
                local_pos = system.particle_constraints.link_local_pos[i_p, i_b]
                target = gu.qd_transform_by_trans_quat(local_pos, link_pos, link_quat)
                system.particle_constraints.target_pos[i_p, i_b] = target

            # Apply spring force to velocity
            target_pos = system.particle_constraints.target_pos[i_p, i_b]
            stiffness = system.particle_constraints.stiffness[i_p, i_b]
            mass = system.particles_info.mass[i_p] / system._particle_volume_scale

            pos = system.particles.pos[f + 1, i_p, i_b]
            vel = system.particles.vel[f + 1, i_p, i_b]

            pos_error = pos - target_pos
            spring_force = -stiffness * pos_error
            damping_force = -2.0 * qd.math.sqrt(stiffness * mass) * vel

            dv = system.substep_dt * (spring_force + damping_force) / mass
            system.particles.vel[f + 1, i_p, i_b] = vel + dv
