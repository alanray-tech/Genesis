"""Free kernels and functions of MPM simulation over MPMSolverSystem-owned data."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.utils import array_class
from genesis.engine.coupling_runtime import func_collide_in_rigid_geom


@qd.func
def stencil_range(system: qd.template()):
    return qd.ndrange(3, 3, 3)


@qd.kernel
def compute_F_tmp(system: qd.template(), f: qd.i32):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng.active[f, i_p, i_b]:
            system.particles.F_tmp[f, i_p, i_b] = (
                qd.Matrix.identity(gs.qd_float, 3) + system.substep_dt * system.particles.C[f, i_p, i_b]
            ) @ system.particles.F[f, i_p, i_b]


@qd.kernel
def svd(system: qd.template(), f: qd.i32):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng.active[f, i_p, i_b]:
            system.particles.U[f, i_p, i_b], system.particles.S[f, i_p, i_b], system.particles.V[f, i_p, i_b] = qd.svd(
                system.particles.F_tmp[f, i_p, i_b], gs.qd_float
            )


@qd.kernel
def compute_F_tmp_and_svd(system: qd.template(), f: qd.i32):
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


@qd.kernel
def compute_F_tmp_only(system: qd.template(), f: qd.i32):
    # Fast path when no registered material needs U/V/S: compute F_tmp without SVD. p2g then derives J from
    # det(F_tmp) directly. Only used on the forward pass.
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng.active[f, i_p, i_b]:
            system.particles.F_tmp[f, i_p, i_b] = (
                qd.Matrix.identity(gs.qd_float, 3) + system.substep_dt * system.particles.C[f, i_p, i_b]
            ) @ system.particles.F[f, i_p, i_b]


@qd.kernel
def svd_grad(system: qd.template(), f: qd.i32):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng.active[f, i_p, i_b]:
            system.particles.F_tmp.grad[f, i_p, i_b] += backward_svd(
                system.particles.U.grad[f, i_p, i_b],
                system.particles.S.grad[f, i_p, i_b],
                system.particles.V.grad[f, i_p, i_b],
                system.particles.U[f, i_p, i_b],
                system.particles.S[f, i_p, i_b],
                system.particles.V[f, i_p, i_b],
            )


@qd.kernel
def p2g(
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


@qd.kernel
def g2p(
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


@qd.kernel
def _is_state_valid(system: qd.template(), f: qd.i32) -> qd.i32:
    is_success = True
    for i_p, i_b, i_3 in qd.ndrange(system._n_particles, system._B, 3):
        if qd.math.isnan(system.particles.pos[f, i_p, i_b][i_3]):
            is_success = False
    return is_success


@qd.kernel
def copy_frame(system: qd.template(), source: qd.i32, target: qd.i32):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        system.particles.pos[target, i_p, i_b] = system.particles.pos[source, i_p, i_b]
        system.particles.vel[target, i_p, i_b] = system.particles.vel[source, i_p, i_b]
        system.particles.F[target, i_p, i_b] = system.particles.F[source, i_p, i_b]
        system.particles.C[target, i_p, i_b] = system.particles.C[source, i_p, i_b]
        system.particles.Jp[target, i_p, i_b] = system.particles.Jp[source, i_p, i_b]

        system.particles_ng.active[target, i_p, i_b] = system.particles_ng.active[source, i_p, i_b]


@qd.kernel
def copy_grad(system: qd.template(), source: qd.i32, target: qd.i32):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        system.particles.pos.grad[target, i_p, i_b] = system.particles.pos.grad[source, i_p, i_b]
        system.particles.vel.grad[target, i_p, i_b] = system.particles.vel.grad[source, i_p, i_b]
        system.particles.F.grad[target, i_p, i_b] = system.particles.F.grad[source, i_p, i_b]
        system.particles.C.grad[target, i_p, i_b] = system.particles.C.grad[source, i_p, i_b]
        system.particles.Jp.grad[target, i_p, i_b] = system.particles.Jp.grad[source, i_p, i_b]
        system.particles_ng.active[target, i_p, i_b] = system.particles_ng.active[source, i_p, i_b]


@qd.kernel
def copy_grad_ndarray(
    source: qd.i32,
    target: qd.i32,
    n_particles: qd.template(),
    n_envs: qd.template(),
    pos_grad: qd.types.ndarray(),
    vel_grad: qd.types.ndarray(),
    F_grad: qd.types.ndarray(),
    C_grad: qd.types.ndarray(),
    Jp_grad: qd.types.ndarray(),
    active: qd.types.ndarray(),
):
    for i_p, i_b in qd.ndrange(n_particles, n_envs):
        pos_grad[target, i_p, i_b] = pos_grad[source, i_p, i_b]
        vel_grad[target, i_p, i_b] = vel_grad[source, i_p, i_b]
        F_grad[target, i_p, i_b] = F_grad[source, i_p, i_b]
        C_grad[target, i_p, i_b] = C_grad[source, i_p, i_b]
        Jp_grad[target, i_p, i_b] = Jp_grad[source, i_p, i_b]
        active[target, i_p, i_b] = active[source, i_p, i_b]


@qd.kernel
def reset_grid_and_grad(system: qd.template(), f: qd.i32):
    # Zero out the grid at frame f for *all* grid cells and *all* batch indices
    for i, j, k, i_b in qd.ndrange(*system._grid_res, system._B):
        system.grid.vel_in[f, i, j, k, i_b] = qd.Vector.zero(gs.qd_float, 3)
        system.grid.mass[f, i, j, k, i_b] = gs.qd_float(0.0)
        system.grid.vel_out[f, i, j, k, i_b] = qd.Vector.zero(gs.qd_float, 3)

        system.grid.vel_in.grad[f, i, j, k, i_b] = qd.Vector.zero(gs.qd_float, 3)
        system.grid.mass.grad[f, i, j, k, i_b] = gs.qd_float(0.0)
        system.grid.vel_out.grad[f, i, j, k, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def reset_dirty_cells(system: qd.template(), f: qd.i32):
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


@qd.kernel
def reset_grad_till_frame(system: qd.template(), f: qd.i32):
    # Zero out particle grads in frames [0, f-1], for all particles, all batch indices
    for i_f, i_p, i_b in qd.ndrange(f, system._n_particles, system._B):
        system.particles.pos.grad[i_f, i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        system.particles.vel.grad[i_f, i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        system.particles.C.grad[i_f, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
        system.particles.F.grad[i_f, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
        system.particles.F_tmp.grad[i_f, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
        system.particles.Jp.grad[i_f, i_p, i_b] = gs.qd_float(0.0)
        system.particles.U.grad[i_f, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
        system.particles.V.grad[i_f, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
        system.particles.S.grad[i_f, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)


@qd.kernel
def add_grad_from_pos(system: qd.template(), f: qd.i32, pos_grad: qd.types.ndarray()):
    # pos_grad shape: [B, n_particles, 3]
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        for j in qd.static(range(3)):
            system.particles.pos.grad[f, i_p, i_b][j] += pos_grad[i_b, i_p, j]


@qd.kernel
def add_grad_from_vel(system: qd.template(), f: qd.i32, vel_grad: qd.types.ndarray()):
    # vel_grad shape: [B, n_particles, 3]
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        for j in qd.static(range(3)):
            system.particles.vel.grad[f, i_p, i_b][j] += vel_grad[i_b, i_p, j]


@qd.kernel
def add_grad_from_C(system: qd.template(), f: qd.i32, C_grad: qd.types.ndarray()):
    # C_grad shape: [B, n_particles, 3, 3]
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        for j in qd.static(range(3)):
            for k in qd.static(range(3)):
                system.particles.C.grad[f, i_p, i_b][j, k] += C_grad[i_b, i_p, j, k]


@qd.kernel
def add_grad_from_F(system: qd.template(), f: qd.i32, F_grad: qd.types.ndarray()):
    # F_grad shape: [B, n_particles, 3, 3]
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        for j in qd.static(range(3)):
            for k in qd.static(range(3)):
                system.particles.F.grad[f, i_p, i_b][j, k] += F_grad[i_b, i_p, j, k]


@qd.kernel
def add_grad_from_Jp(system: qd.template(), f: qd.i32, Jp_grad: qd.types.ndarray()):
    # Jp_grad shape: [B, n_particles]
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        system.particles.Jp.grad[f, i_p, i_b] += Jp_grad[i_b, i_p]


@qd.kernel
def _kernel_set_state(
    system: qd.template(),
    f: qd.i32,
    pos: qd.types.ndarray(),  # shape [B, n_particles, 3]
    vel: qd.types.ndarray(),  # shape [B, n_particles, 3]
    C: qd.types.ndarray(),  # shape [B, n_particles, 3, 3]
    F: qd.types.ndarray(),  # shape [B, n_particles, 3, 3]
    Jp: qd.types.ndarray(),  # shape [B, n_particles]
    active: qd.types.ndarray(),  # shape [B, n_particles]
):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        # Write pos, vel
        for j in qd.static(range(3)):
            system.particles.pos[f, i_p, i_b][j] = pos[i_b, i_p, j]
            system.particles.vel[f, i_p, i_b][j] = vel[i_b, i_p, j]
            # Write C, F
            for k in qd.static(range(3)):
                system.particles.C[f, i_p, i_b][j, k] = C[i_b, i_p, j, k]
                system.particles.F[f, i_p, i_b][j, k] = F[i_b, i_p, j, k]
        # Write Jp, active
        system.particles.Jp[f, i_p, i_b] = Jp[i_b, i_p]
        system.particles_ng.active[f, i_p, i_b] = active[i_b, i_p]


@qd.kernel
def _kernel_get_state(
    system: qd.template(),
    f: qd.i32,
    pos: qd.types.ndarray(),  # shape [B, n_particles, 3]
    vel: qd.types.ndarray(),  # shape [B, n_particles, 3]
    C: qd.types.ndarray(),  # shape [B, n_particles, 3, 3]
    F: qd.types.ndarray(),  # shape [B, n_particles, 3, 3]
    Jp: qd.types.ndarray(),  # shape [B, n_particles]
    active: qd.types.ndarray(),  # shape [B, n_particles]
):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        for j in qd.static(range(3)):
            pos[i_b, i_p, j] = system.particles.pos[f, i_p, i_b][j]
            vel[i_b, i_p, j] = system.particles.vel[f, i_p, i_b][j]
            for k in qd.static(range(3)):
                C[i_b, i_p, j, k] = system.particles.C[f, i_p, i_b][j, k]
                F[i_b, i_p, j, k] = system.particles.F[f, i_p, i_b][j, k]
        Jp[i_b, i_p] = system.particles.Jp[f, i_p, i_b]
        active[i_b, i_p] = qd.cast(system.particles_ng.active[f, i_p, i_b], gs.qd_bool)


@qd.kernel
def _kernel_update_render_fields(system: qd.template(), f: qd.i32):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng.active[f, i_p, i_b]:
            system.particles_render.pos[i_p, i_b] = system.particles.pos[f, i_p, i_b]
            system.particles_render.vel[i_p, i_b] = system.particles.vel[f, i_p, i_b]
        else:
            system.particles_render.pos[i_p, i_b] = gu.qd_nowhere()
        system.particles_render.active[i_p, i_b] = system.particles_ng.active[f, i_p, i_b]

    if qd.static(system._n_vverts > 0):
        for i_v, i_b in qd.ndrange(system._n_vverts, system._B):
            vvert_pos = qd.Vector.zero(gs.qd_float, 3)
            for j in range(system._n_vvert_supports):
                i_p = system.vverts_info.support_idxs[i_v][j]
                vvert_pos += system.particles.pos[f, i_p, i_b] * system.vverts_info.support_weights[i_v][j]
            system.vverts_render.pos[i_v, i_b] = vvert_pos
            i_p = system.vverts_info.support_idxs[i_v][0]
            system.vverts_render.active[i_v, i_b] = system.particles_render.active[i_p, i_b]


@qd.kernel
def _kernel_add_particles(
    system: qd.template(),
    f: qd.i32,
    active: qd.i32,
    particle_start: qd.i32,
    n_particles: qd.i32,
    material_idx: qd.i32,
    mat_default_Jp: qd.f32,
    mat_rho: qd.f32,
    pos: qd.types.ndarray(),  # shape [n_particles, 3]
):
    for i_p_ in range(n_particles):
        i_p = i_p_ + particle_start

        system.particles_info.material_idx[i_p] = material_idx
        system.particles_info.default_Jp[i_p] = mat_default_Jp
        system.particles_info.mass[i_p] = system._particle_volume * mat_rho
        system.particles_info.free[i_p] = True
        system.particles_info.muscle_group[i_p] = 0
        system.particles_info.muscle_direction[i_p] = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)

    for i_p_, i_b in qd.ndrange(n_particles, system._B):
        i_p = i_p_ + particle_start

        system.particles_ng.active[f, i_p, i_b] = qd.cast(active, gs.qd_bool)
        for i in qd.static(range(3)):
            system.particles.pos[f, i_p, i_b][i] = pos[i_p_, i]

        system.particles.vel[f, i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        system.particles.F[f, i_p, i_b] = qd.Matrix.identity(gs.qd_float, 3)
        system.particles.C[f, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
        system.particles.Jp[f, i_p, i_b] = mat_default_Jp
        system.particles.actu[f, i_p, i_b] = gs.qd_float(0.0)


@qd.kernel
def _kernel_set_particles_pos(
    system: qd.template(),
    f: qd.i32,
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]

        for i in qd.static(range(3)):
            system.particles.pos[f, i_p, i_b][i] = poss[i_b_, i_p_, i]

        # Reset these attributes whenever overwritting particle positions manually
        system.particles.vel[f, i_p, i_b].fill(0.0)
        system.particles.F[f, i_p, i_b] = qd.Matrix.identity(gs.qd_float, 3)
        system.particles.C[f, i_p, i_b].fill(0.0)
        system.particles.Jp[f, i_p, i_b] = system.particles_info.default_Jp[i_p]


@qd.kernel
def _kernel_set_particles_pos_grad(
    system: qd.template(),
    f: qd.i32,
    particle_start: qd.i32,
    n_particles: qd.i32,
    poss_grad: qd.types.ndarray(),  # shape [B, n_particles, 3]
):
    for i_p_, i_b in qd.ndrange(n_particles, system._B):
        i_p = i_p_ + particle_start
        for i in qd.static(range(3)):
            poss_grad[i_b, i_p_, i] = system.particles.pos.grad[f, i_p, i_b][i]


@qd.kernel
def _kernel_get_particles_pos(
    system: qd.template(),
    f: qd.i32,
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
):
    for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
        i_p = i_p_ + particle_start
        i_b = envs_idx[i_b_]
        for i in qd.static(range(3)):
            poss[i_b_, i_p_, i] = system.particles.pos[f, i_p, i_b][i]


@qd.kernel
def _kernel_set_particles_vel(
    system: qd.template(),
    f: qd.i32,
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    vels: qd.types.ndarray(),  # shape [B, n_particles, 3]
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        for i in qd.static(range(3)):
            system.particles.vel[f, i_p, i_b][i] = vels[i_b_, i_p_, i]


@qd.kernel
def _kernel_set_particles_vel_grad(
    system: qd.template(),
    f: qd.i32,
    particle_start: qd.i32,
    n_particles: qd.i32,
    vels_grad: qd.types.ndarray(),  # shape [B, n_particles, 3]
):
    for i_p_, i_b in qd.ndrange(n_particles, system._B):
        i_p = i_p_ + particle_start
        for i in qd.static(range(3)):
            vels_grad[i_b, i_p_, i] = system.particles.vel.grad[f, i_p, i_b][i]


@qd.kernel
def _kernel_get_particles_vel(
    system: qd.template(),
    f: qd.i32,
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    vels: qd.types.ndarray(),
):
    for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
        i_p = i_p_ + particle_start
        i_b = envs_idx[i_b_]
        for i in qd.static(range(3)):
            vels[i_b_, i_p_, i] = system.particles.vel[f, i_p, i_b][i]


@qd.kernel
def _kernel_set_particles_active(
    system: qd.template(),
    f: qd.i32,
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    actives: qd.types.ndarray(),  # shape [B, n_particles]
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        system.particles_ng.active[f, i_p, i_b] = qd.cast(actives[i_b_, i_p_], gs.qd_bool)


@qd.kernel
def _kernel_get_particles_active(
    system: qd.template(),
    f: qd.i32,
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    actives: qd.types.ndarray(),  # shape [B, n_particles]
):
    for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
        i_p = i_p_ + particle_start
        i_b = envs_idx[i_b_]
        actives[i_b_, i_p_] = system.particles_ng.active[f, i_p, i_b]


@qd.kernel
def _kernel_set_particles_actu(
    system: qd.template(),
    f: qd.i32,
    n_groups: qd.i32,
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    actus: qd.types.ndarray(),  # shape [B, n_particles, n_groups]
):
    for i_p_, i_g, i_b_ in qd.ndrange(particles_idx.shape[1], n_groups, envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        if system.particles_info.muscle_group[i_p] == i_g:
            system.particles.actu[f, i_p, i_b] = actus[i_b_, i_p_, i_g]


@qd.kernel
def _kernel_set_particles_actu_grad(
    system: qd.template(),
    f: qd.i32,
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    actus_grad: qd.types.ndarray(),  # shape [B, n_particles]
):
    for i_p_, i_g, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
        i_p = i_p_ + particle_start
        i_b = envs_idx[i_b_]
        actus_grad[i_b_, i_p_] = system.particles.actu.grad[f, i_p, i_b]


@qd.kernel
def _kernel_get_particles_actu(
    system: qd.template(),
    f: qd.i32,
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    actus: qd.types.ndarray(),  # shape [B, n_particles]
):
    for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
        i_p = i_p_ + particle_start
        i_b = envs_idx[i_b_]
        actus[i_b_, i_p_] = system.particles.actu[f, i_p, i_b]


@qd.kernel
def _kernel_set_particles_muscle_group(
    system: qd.template(), particles_idx: qd.types.ndarray(), muscle_group: qd.types.ndarray()
):
    for i_p_ in range(particles_idx.shape[0]):
        i_p = particles_idx[i_p_]
        system.particles_info.muscle_group[i_p] = muscle_group[i_p_]


@qd.kernel
def _kernel_get_particles_muscle_group(
    system: qd.template(), particle_start: qd.i32, n_particles: qd.i32, muscle_group: qd.types.ndarray()
):
    for i_p_ in range(n_particles):
        i_p = i_p_ + particle_start
        muscle_group[i_p_] = system.particles_info.muscle_group[i_p]


@qd.kernel
def _kernel_set_particles_muscle_direction(
    system: qd.template(), particles_idx: qd.types.ndarray(), muscle_direction: qd.types.ndarray()
):
    for i_p_ in range(particles_idx.shape[0]):
        i_p = particles_idx[i_p_]
        for i in qd.static(range(3)):
            system.particles_info.muscle_direction[i_p][i] = muscle_direction[i_p_, i]


@qd.kernel
def _kernel_set_particles_free(system: qd.template(), particles_idx: qd.types.ndarray(), free: qd.types.ndarray()):
    for i_p_ in range(particles_idx.shape[0]):
        i_p = particles_idx[i_p_]
        system.particles_info.free[i_p] = free[i_p_]


@qd.kernel
def _kernel_get_particles_free(
    system: qd.template(), particle_start: qd.i32, n_particles: qd.i32, free: qd.types.ndarray()
):
    for i_p_ in range(n_particles):
        i_p = i_p_ + particle_start
        free[i_p_] = system.particles_info.free[i_p]


@qd.kernel
def _kernel_get_mass(
    system: qd.template(),
    particle_start: qd.i32,
    n_particles: qd.i32,
    mass: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
):
    total_mass = gs.qd_float(0.0)
    for i_p_ in range(n_particles):
        i_p = i_p_ + particle_start
        total_mass += system.particles_info.mass[i_p]
    total_mass = total_mass / system._particle_volume_scale
    for i_b_ in range(envs_idx.shape[0]):
        mass[i_b_] = total_mass


@qd.kernel
def _kernel_set_particle_constraints(
    system: qd.template(),
    f: qd.i32,
    particles_mask: qd.types.ndarray(),  # shape [n_envs, n_particles] boolean mask
    particle_start: qd.i32,
    stiffness: qd.f32,
    link_idx: qd.i32,
    link_pos: qd.types.ndarray(),  # shape [n_envs, 3]
    link_quat: qd.types.ndarray(),  # shape [n_envs, 4]
):
    for i_p_local, i_b in qd.ndrange(particles_mask.shape[1], particles_mask.shape[0]):
        if particles_mask[i_b, i_p_local]:
            i_p = i_p_local + particle_start

            # Get current particle position
            pos = system.particles.pos[f, i_p, i_b]

            # Get link transform
            l_pos = qd.Vector([link_pos[i_b, 0], link_pos[i_b, 1], link_pos[i_b, 2]], dt=gs.qd_float)
            l_quat = qd.Vector(
                [link_quat[i_b, 0], link_quat[i_b, 1], link_quat[i_b, 2], link_quat[i_b, 3]], dt=gs.qd_float
            )

            # Compute offset in link's local frame
            local_pos = gu.qd_inv_transform_by_trans_quat(pos, l_pos, l_quat)

            # Store constraint info
            system.particle_constraints.is_constrained[i_p, i_b] = True
            system.particle_constraints.target_pos[i_p, i_b] = pos  # initial target is current position
            system.particle_constraints.stiffness[i_p, i_b] = stiffness
            system.particle_constraints.link_idx[i_p, i_b] = link_idx
            system.particle_constraints.link_local_pos[i_p, i_b] = local_pos


@qd.kernel
def _kernel_remove_particle_constraints(
    system: qd.template(),
    particles_mask: qd.types.ndarray(),  # shape [n_envs, n_particles] boolean mask
    particle_start: qd.i32,
):
    for i_p_local, i_b in qd.ndrange(particles_mask.shape[1], particles_mask.shape[0]):
        if particles_mask[i_b, i_p_local]:
            i_p = i_p_local + particle_start
            system.particle_constraints.is_constrained[i_p, i_b] = False
            system.particle_constraints.link_idx[i_p, i_b] = -1


@qd.kernel
def apply_particle_constraints(system: qd.template(), f: qd.i32, links_state: array_class.LinksState):
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


@qd.func
def signmax(a, eps):
    sign = qd.select(a >= 0, 1.0, -1.0)
    return sign * qd.max(qd.abs(a), eps)


@qd.func
def backward_svd(grad_U, grad_S, grad_V, U, S, V):
    vt = V.transpose()
    ut = U.transpose()
    S_term = U @ grad_S @ vt

    s = qd.Vector([S[0, 0], S[1, 1], S[2, 2]]) ** 2
    F = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i, j in qd.static(qd.ndrange(3, 3)):
        if i == j:
            F[i, j] = 0.0
        else:
            F[i, j] = 1.0 / signmax(s[j] - s[i], 1e-6)
    u_term = U @ ((F * (ut @ grad_U - grad_U.transpose() @ U)) @ S) @ vt
    v_term = U @ (S @ ((F * (vt @ grad_V - grad_V.transpose() @ V)) @ vt))
    return u_term + v_term + S_term
