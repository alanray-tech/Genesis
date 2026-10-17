"""Device kernels for the individual FEM coupling pairs and the FEM boundary."""

import quadrants as qd

import genesis as gs
from genesis.engine.pairwise_coupling_runtime import func_collide_with_rigid
from genesis.utils import array_class
from genesis.utils.spatial_hash import func_grid_to_slot, func_pos_to_grid


@qd.kernel
def kernel_rigid_fem_couple(
    f: qd.i32,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    fem_system: qd.template(),
    n_geoms: qd.template(),
    collider_static_config: qd.template(),
):
    for i_s, i_b in qd.ndrange(fem_system.n_surfaces, fem_system._B):
        if fem_system.surface.active[i_s]:
            iel = fem_system.surface.tri2el[i_s]
            mass = fem_system.elements_i.mass_scaled[iel] / fem_system.vol_scale
            for j in qd.static(range(3)):
                iv = fem_system.surface.tri2v[i_s][j]
                fem_system.elements_v.vel[f + 1, iv, i_b] = func_collide_with_rigid(
                    fem_system.elements_v.pos[f, iv, i_b],
                    fem_system.elements_v.vel[f + 1, iv, i_b],
                    mass / 3.0,
                    i_b,
                    n_geoms,
                    geoms_state,
                    geoms_info,
                    links_state,
                    rigid_info,
                    sdf_info,
                    collider_static_config,
                )


@qd.kernel
def kernel_fem_mpm_couple(f: qd.i32, fem_system: qd.template(), mpm_system: qd.template()):
    for i_s, i_b in qd.ndrange(fem_system.n_surfaces, fem_system._B):
        if fem_system.surface.active[i_s]:
            iel = fem_system.surface.tri2el[i_s]
            mass = fem_system.elements_i.mass_scaled[iel] / fem_system.vol_scale
            p1 = fem_system.elements_v.pos[f, fem_system.surface.tri2v[i_s][0], i_b]
            p2 = fem_system.elements_v.pos[f, fem_system.surface.tri2v[i_s][1], i_b]
            p3 = fem_system.elements_v.pos[f, fem_system.surface.tri2v[i_s][2], i_b]
            surface_normal = qd.math.cross(p2 - p1, p3 - p1)
            surface_normal = surface_normal / surface_normal.norm(gs.EPS)

            for j in qd.static(range(3)):
                iv = fem_system.surface.tri2v[i_s][j]
                pos = fem_system.elements_v.pos[f, iv, i_b]
                vel_fem_sv = fem_system.elements_v.vel[f + 1, iv, i_b]
                mass_fem_sv = mass / 4.0
                vel_mpm = qd.Vector([0.0, 0.0, 0.0])
                mass_mpm = 0.0
                mpm_base = qd.floor(pos * mpm_system._inv_dx - 0.5).cast(gs.qd_int)
                mpm_fx = pos * mpm_system._inv_dx - mpm_base.cast(gs.qd_float)
                mpm_w = [0.5 * (1.5 - mpm_fx) ** 2, 0.75 - (mpm_fx - 1.0) ** 2, 0.5 * (mpm_fx - 0.5) ** 2]
                new_vel_fem_sv = vel_fem_sv
                for mpm_offset in qd.static(qd.grouped(qd.ndrange(3, 3, 3))):
                    mpm_grid_I = mpm_base - mpm_system.grid_offset + mpm_offset
                    mpm_grid_mass = mpm_system.grid.mass[f, mpm_grid_I, i_b] / mpm_system._particle_volume_scale
                    mpm_weight = gs.qd_float(1.0)
                    for d in qd.static(range(3)):
                        mpm_weight *= mpm_w[mpm_offset[d]][d]
                    mpm_grid_pos = (mpm_grid_I + mpm_system.grid_offset) * mpm_system.dx
                    if (mpm_grid_pos - pos).dot(surface_normal) <= mpm_system.dx:
                        mass_mpm_at_cell = mpm_weight * mpm_grid_mass
                        vel_mpm += mpm_weight * mpm_system.grid.vel_out[f, mpm_grid_I, i_b]
                        mass_mpm += mass_mpm_at_cell
                        if mass_mpm_at_cell > gs.EPS:
                            delta_mpm_vel_at_cell = (
                                vel_fem_sv * mpm_weight - mpm_system.grid.vel_out[f, mpm_grid_I, i_b]
                            ) * (mpm_grid_mass / mass_fem_sv)
                            mpm_system.grid.vel_out[f, mpm_grid_I, i_b] += delta_mpm_vel_at_cell
                            new_vel_fem_sv -= delta_mpm_vel_at_cell * mass_mpm_at_cell / mass_fem_sv
                if mass_mpm > gs.EPS:
                    fem_system.elements_v.vel[f + 1, iv, i_b] = new_vel_fem_sv


@qd.kernel
def kernel_fem_sph_couple(f: qd.i32, fem_system: qd.template(), sph_system: qd.template()):
    for i_s, i_b in qd.ndrange(fem_system.n_surfaces, fem_system._B):
        if fem_system.surface.active[i_s]:
            iel = fem_system.surface.tri2el[i_s]
            mass = fem_system.elements_i.mass_scaled[iel] / fem_system.vol_scale
            p1 = fem_system.elements_v.pos[f, fem_system.surface.tri2v[i_s][0], i_b]
            p2 = fem_system.elements_v.pos[f, fem_system.surface.tri2v[i_s][1], i_b]
            p3 = fem_system.elements_v.pos[f, fem_system.surface.tri2v[i_s][2], i_b]
            surface_normal = qd.math.cross(p2 - p1, p3 - p1)
            surface_normal = surface_normal / surface_normal.norm(gs.EPS)

            for j in qd.static(range(3)):
                iv = fem_system.surface.tri2v[i_s][j]
                pos = fem_system.elements_v.pos[f, iv, i_b]
                vel_fem_sv = fem_system.elements_v.vel[f + 1, iv, i_b]
                mass_fem_sv = mass / 4.0
                dx = sph_system.sh.cell_size
                stencil_size = 2
                base = func_pos_to_grid(pos - 0.5 * dx, sph_system.sh)
                sph_vel = qd.Vector([0.0, 0.0, 0.0])
                colliding_particles = 0
                for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                    slot_idx = func_grid_to_slot(base + offset, sph_system.sh)
                    for k in range(
                        sph_system.spatial_hasher.slot_start[slot_idx, i_b],
                        sph_system.spatial_hasher.slot_start[slot_idx, i_b]
                        + sph_system.spatial_hasher.slot_size[slot_idx, i_b],
                    ):
                        if qd.abs(pos - sph_system.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                            sph_vel += sph_system.particles_reordered.vel[k, i_b]
                            colliding_particles += 1
                if colliding_particles > 0:
                    vel_old = vel_fem_sv
                    vel_fem_sv = (sph_vel / colliding_particles).dot(surface_normal) * surface_normal
                    delta_mv = mass_fem_sv * (vel_fem_sv - vel_old)
                    for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                        slot_idx = func_grid_to_slot(base + offset, sph_system.sh)
                        for k in range(
                            sph_system.spatial_hasher.slot_start[slot_idx, i_b],
                            sph_system.spatial_hasher.slot_start[slot_idx, i_b]
                            + sph_system.spatial_hasher.slot_size[slot_idx, i_b],
                        ):
                            if qd.abs(pos - sph_system.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                sph_system.particles_reordered.vel[k, i_b] -= (
                                    delta_mv / sph_system.particles_info_reordered.mass[k, i_b]
                                )
                    fem_system.elements_v.vel[f + 1, iv, i_b] = vel_fem_sv


@qd.kernel
def kernel_fem_boundary(f: qd.i32, fem_system: qd.template()):
    for i_s, i_b in qd.ndrange(fem_system.n_surfaces, fem_system._B):
        if fem_system.surface.active[i_s]:
            for j in qd.static(range(3)):
                iv = fem_system.surface.tri2v[i_s][j]
                _, fem_system.elements_v.vel[f + 1, iv, i_b] = fem_system.boundary.impose_pos_vel(
                    fem_system.elements_v.pos[f, iv, i_b],
                    fem_system.elements_v.vel[f + 1, iv, i_b],
                )
