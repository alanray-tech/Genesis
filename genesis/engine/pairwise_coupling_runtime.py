"""Device stages used by the pairwise Systems replacing the former compound coupling kernels."""

import quadrants as qd

import genesis as gs
import genesis.utils.sdf as sdf
from genesis.engine import sph_runtime
from genesis.engine.coupling_runtime import func_apply_rigid_coupling_force, func_collide_in_rigid_geom
from genesis.engine.entities.tool_entity.data import ToolSolverInfo
from genesis.engine.entities.tool_entity.mesh import func_collide
from genesis.utils import array_class
from genesis.utils.geom import qd_inv_transform_by_trans_quat, qd_transform_by_trans_quat
from genesis.utils.spatial_hash import func_grid_to_slot, func_pos_to_grid


@qd.kernel
def kernel_reset_rigid_normals_ndarray(
    rigid_normal: qd.types.ndarray(),
    n_particles: qd.template(),
    n_geoms: qd.template(),
    envs_idx: qd.types.ndarray(),
):
    for i_p, i_g, i_b_ in qd.ndrange(n_particles, n_geoms, envs_idx.shape[0]):
        rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0


@qd.kernel
def kernel_reset_rigid_normals_field(
    rigid_normal: qd.template(),
    n_particles: qd.template(),
    n_geoms: qd.template(),
    envs_idx: qd.types.ndarray(),
):
    for i_p, i_g, i_b_ in qd.ndrange(n_particles, n_geoms, envs_idx.shape[0]):
        rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0


@qd.func
def func_mpm_tool(
    f: int,
    pos_world: qd.types.vector(3),
    vel: qd.types.vector(3),
    i_b: int,
    tool_data: qd.template(),
    tool_info: ToolSolverInfo,
):
    for i_e in qd.static(range(len(tool_data.entities))):
        if qd.static(tool_data.meshes[i_e].collision):
            vel = func_collide(f, pos_world, vel, i_b, tool_data.entities[i_e], tool_info, tool_data.meshes[i_e])
    return vel


@qd.func
def func_collide_with_rigid_geom(
    pos_world,
    vel,
    mass,
    geom_idx,
    batch_idx,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    signed_dist = sdf.sdf_func_world(geom_idx, batch_idx, pos_world, geoms_state, geoms_info, sdf_info)
    influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)
    if influence > 0.1:
        normal_rigid = sdf.sdf_func_normal_world(
            geom_idx,
            batch_idx,
            pos_world,
            geoms_state,
            geoms_info,
            rigid_info,
            sdf_info,
            collider_static_config,
        )
        vel = func_collide_in_rigid_geom(
            pos_world,
            vel,
            mass,
            normal_rigid,
            influence,
            geom_idx,
            batch_idx,
            geoms_info,
            links_state,
            rigid_info,
        )
    return vel


@qd.func
def func_collide_with_rigid(
    pos_world,
    vel,
    mass,
    i_b,
    n_geoms: qd.template(),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    for i_g in range(n_geoms):
        if geoms_info.needs_coup[i_g]:
            vel = func_collide_with_rigid_geom(
                pos_world,
                vel,
                mass,
                i_g,
                i_b,
                geoms_state,
                geoms_info,
                links_state,
                rigid_info,
                sdf_info,
                collider_static_config,
            )
    return vel


@qd.func
def func_collide_with_rigid_geom_robust(
    pos_world,
    vel,
    mass,
    pressure,
    normal_prev,
    geom_idx,
    batch_idx,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    sph: qd.template(),
    batch_links_info: qd.template(),
    collider_static_config: qd.template(),
):
    signed_dist = sdf.sdf_func_world(geom_idx, batch_idx, pos_world, geoms_state, geoms_info, sdf_info)
    normal_rigid = sdf.sdf_func_normal_world(
        geom_idx,
        batch_idx,
        pos_world,
        geoms_state,
        geoms_info,
        rigid_info,
        sdf_info,
        collider_static_config,
    )
    influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)
    if influence > 0.1:
        vel = func_collide_in_rigid_geom(
            pos_world,
            vel,
            mass,
            normal_rigid,
            influence,
            geom_idx,
            batch_idx,
            geoms_info,
            links_state,
            rigid_info,
        )

    link_idx = geoms_info.link_idx[geom_idx]
    I_l = [link_idx, batch_idx] if qd.static(batch_links_info) else link_idx
    if signed_dist < sph._support_radius and pressure > 0 and not links_info.is_fixed[I_l]:
        pressure_force = (
            -2.0
            * pressure
            * sph._particle_volume
            * sph_runtime.cubic_kernel_plane_integral(sph, signed_dist)
            * normal_rigid
        )
        func_apply_rigid_coupling_force(link_idx, batch_idx, pos_world, pressure_force, links_state)
        vel = vel - pressure_force * (rigid_info.substep_dt[None] / mass)
    return vel, normal_rigid


@qd.kernel
def kernel_sph_rigid_couple(
    f: qd.i32,
    sph: qd.template(),
    coupling: qd.template(),
    n_geoms: qd.template(),
    batch_links_info: qd.template(),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    for i_p, i_b in qd.ndrange(sph._n_particles, sph._B):
        if sph.particles_ng_reordered.active[i_p, i_b]:
            for i_g in range(n_geoms):
                if geoms_info.needs_coup[i_g]:
                    (
                        sph.particles_reordered.vel[i_p, i_b],
                        coupling.rigid_normal_reordered[i_p, i_g, i_b],
                    ) = func_collide_with_rigid_geom_robust(
                        sph.particles_reordered.pos[i_p, i_b],
                        sph.particles_reordered.vel[i_p, i_b],
                        sph.particles_info_reordered.mass[i_p, i_b],
                        sph.particles_reordered.p[i_p, i_b],
                        coupling.rigid_normal_reordered[i_p, i_g, i_b],
                        i_g,
                        i_b,
                        geoms_state,
                        geoms_info,
                        links_state,
                        links_info,
                        rigid_info,
                        sdf_info,
                        sph,
                        batch_links_info,
                        collider_static_config,
                    )


@qd.func
def func_pbd_collide_with_rigid_geom(
    i,
    pos_world,
    vel,
    mass,
    geom_idx,
    batch_idx,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    sdf_info: array_class.SDFInfo,
    rigid_info: array_class.RigidInfo,
    pbd: qd.template(),
    rigid_substep_dt: qd.template(),
    collider_static_config: qd.template(),
):
    signed_dist = sdf.sdf_func_world(geom_idx, batch_idx, pos_world, geoms_state, geoms_info, sdf_info)
    contact_normal = sdf.sdf_func_normal_world(
        geom_idx,
        batch_idx,
        pos_world,
        geoms_state,
        geoms_info,
        rigid_info,
        sdf_info,
        collider_static_config,
    )
    new_pos = pos_world
    new_vel = vel
    if signed_dist < pbd._particle_size / 2:
        new_pos = pos_world + contact_normal * (pbd._particle_size / 2 - signed_dist)
        new_vel = (new_pos - pbd.particles_reordered.ipos[i, batch_idx]) / pbd._substep_dt
        force = -(mass * (new_vel - vel)) / rigid_substep_dt
        func_apply_rigid_coupling_force(geoms_info.link_idx[geom_idx], batch_idx, pos_world, force, links_state)
    return new_pos, new_vel, contact_normal


@qd.kernel
def kernel_pbd_rigid_couple(
    pbd: qd.template(),
    coupling: qd.template(),
    n_geoms: qd.template(),
    rigid_substep_dt: qd.template(),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    sdf_info: array_class.SDFInfo,
    rigid_info: array_class.RigidInfo,
    collider_static_config: qd.template(),
):
    for i_p, i_b in qd.ndrange(pbd._n_particles, pbd._B):
        if pbd.particles_ng_reordered.active[i_p, i_b]:
            for i_g in range(n_geoms):
                if geoms_info.needs_coup[i_g]:
                    (
                        pbd.particles_reordered.pos[i_p, i_b],
                        pbd.particles_reordered.vel[i_p, i_b],
                        coupling.rigid_normal_reordered[i_p, i_b, i_g],
                    ) = func_pbd_collide_with_rigid_geom(
                        i_p,
                        pbd.particles_reordered.pos[i_p, i_b],
                        pbd.particles_reordered.vel[i_p, i_b],
                        pbd.particles_info_reordered.mass[i_p, i_b],
                        i_g,
                        i_b,
                        geoms_state,
                        geoms_info,
                        links_state,
                        sdf_info,
                        rigid_info,
                        pbd,
                        rigid_substep_dt,
                        collider_static_config,
                    )


@qd.kernel
def kernel_attach_pbd_to_rigid_link(
    pbd: qd.template(),
    coupling: qd.template(),
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    link_idx: qd.i32,
    links_state: array_class.LinksState,
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        local_pos = qd_inv_transform_by_trans_quat(
            pbd.particles.pos[i_p, i_b],
            links_state.pos[link_idx, i_b],
            links_state.quat[link_idx, i_b],
        )
        pbd.particles.free[i_p, i_b] = False
        coupling.attachment.link_idx[i_p, i_b] = link_idx
        coupling.attachment.local_pos[i_p, i_b] = local_pos


@qd.kernel
def kernel_clear_pbd_rigid_attachment(
    pbd: qd.template(),
    coupling: qd.template(),
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        pbd.particles.free[i_p, i_b] = True
        coupling.attachment.link_idx[i_p, i_b] = -1
        coupling.attachment.local_pos[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_solve_pbd_rigid_attachment(
    pbd: qd.template(),
    coupling: qd.template(),
    clamped_inv_dt: qd.f32,
    links_state: array_class.LinksState,
):
    for i_p, i_b in qd.ndrange(pbd._n_particles, pbd._B):
        if coupling.attachment.link_idx[i_p, i_b] >= 0:
            link_idx = coupling.attachment.link_idx[i_p, i_b]
            target_world_pos = qd_transform_by_trans_quat(
                coupling.attachment.local_pos[i_p, i_b],
                links_state.pos[link_idx, i_b],
                links_state.quat[link_idx, i_b],
            )
            link_com = links_state.root_COM[link_idx, i_b] + links_state.i_pos[link_idx, i_b]
            target_world_vel = links_state.cd_vel[link_idx, i_b] + links_state.cd_ang[link_idx, i_b].cross(
                target_world_pos - link_com
            )
            i_rp = pbd.particles_ng.reordered_idx[i_p, i_b]
            corrective_vel = (target_world_pos - pbd.particles_reordered.pos[i_rp, i_b]) * clamped_inv_dt
            pbd.particles_reordered.vel[i_rp, i_b] = corrective_vel + target_world_vel


@qd.kernel
def kernel_mpm_surface_to_particle(
    f: qd.i32,
    mpm: qd.template(),
    coupling: qd.template(),
    n_geoms: qd.template(),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    """Update CPIC particle-side normals before MPM P2G."""
    for i_p, i_b in qd.ndrange(mpm._n_particles, mpm._B):
        if mpm.particles_ng.active[f, i_p, i_b]:
            for i_g in range(n_geoms):
                if geoms_info.needs_coup[i_g]:
                    normal = sdf.sdf_func_normal_world(
                        i_g,
                        i_b,
                        mpm.particles.pos[f, i_p, i_b],
                        geoms_state,
                        geoms_info,
                        rigid_info,
                        sdf_info,
                        collider_static_config,
                    )
                    if normal.dot(coupling.rigid_normal[i_p, i_g, i_b]) >= 0:
                        coupling.rigid_normal[i_p, i_g, i_b] = normal


@qd.func(requires_top_level=True)
def func_mpm_grid_begin(f: int, t: float, mpm: qd.template()):
    """Convert grid momentum to velocity, then apply gravity and external force fields."""
    for ii, jj, kk, i_b in qd.ndrange(*mpm.grid_res, mpm._B):
        I = (ii, jj, kk)
        if mpm.grid.mass[f, I, i_b] > gs.EPS:
            vel = (1 / mpm.grid.mass[f, I, i_b]) * mpm.grid.vel_in[f, I, i_b]
            vel += mpm.substep_dt * mpm._gravity[i_b]
            pos = (I + mpm.grid_offset) * mpm.dx
            for i_ff in qd.static(range(len(mpm._ffs))):
                vel += mpm._ffs[i_ff].get_acc(pos, vel, t, -1) * mpm.substep_dt
            mpm.grid.vel_out[f, I, i_b] = vel


@qd.kernel
def kernel_mpm_grid_begin(f: qd.i32, t: qd.f32, mpm: qd.template()):
    for ii, jj, kk, i_b in qd.ndrange(*mpm.grid_res, mpm._B):
        I = (ii, jj, kk)
        if mpm.grid.mass[f, I, i_b] > gs.EPS:
            vel = (1 / mpm.grid.mass[f, I, i_b]) * mpm.grid.vel_in[f, I, i_b]
            vel += mpm.substep_dt * mpm._gravity[i_b]
            pos = (I + mpm.grid_offset) * mpm.dx
            for i_ff in qd.static(range(len(mpm._ffs))):
                vel += mpm._ffs[i_ff].get_acc(pos, vel, t, -1) * mpm.substep_dt
            mpm.grid.vel_out[f, I, i_b] = vel


@qd.func(requires_top_level=True)
def func_mpm_tool_couple(
    f: int,
    mpm: qd.template(),
    tool: qd.template(),
):
    """Apply MPM-Tool collision to the current grid velocity."""
    for ii, jj, kk, i_b in qd.ndrange(*mpm.grid_res, mpm._B):
        I = (ii, jj, kk)
        if mpm.grid.mass[f, I, i_b] > gs.EPS:
            pos = (I + mpm.grid_offset) * mpm.dx
            mpm.grid.vel_out[f, I, i_b] = func_mpm_tool(f, pos, mpm.grid.vel_out[f, I, i_b], i_b, tool.data, tool.info)


@qd.kernel
def kernel_mpm_tool_couple(
    f: qd.i32,
    mpm: qd.template(),
    tool: qd.template(),
):
    for ii, jj, kk, i_b in qd.ndrange(*mpm.grid_res, mpm._B):
        I = (ii, jj, kk)
        if mpm.grid.mass[f, I, i_b] > gs.EPS:
            pos = (I + mpm.grid_offset) * mpm.dx
            mpm.grid.vel_out[f, I, i_b] = func_mpm_tool(f, pos, mpm.grid.vel_out[f, I, i_b], i_b, tool.data, tool.info)


@qd.func(requires_top_level=True)
def func_mpm_rigid_couple(
    f: int,
    mpm: qd.template(),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
    n_geoms: qd.template(),
):
    """Apply MPM-Rigid collision and accumulate the rigid reaction."""
    for ii, jj, kk, i_b in qd.ndrange(*mpm.grid_res, mpm._B):
        I = (ii, jj, kk)
        if mpm.grid.mass[f, I, i_b] > gs.EPS:
            pos = (I + mpm.grid_offset) * mpm.dx
            mass = mpm.grid.mass[f, I, i_b] / mpm._particle_volume_scale
            mpm.grid.vel_out[f, I, i_b] = func_collide_with_rigid(
                pos,
                mpm.grid.vel_out[f, I, i_b],
                mass,
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
def kernel_mpm_rigid_couple(
    f: qd.i32,
    mpm: qd.template(),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
    n_geoms: qd.template(),
):
    for ii, jj, kk, i_b in qd.ndrange(*mpm.grid_res, mpm._B):
        I = (ii, jj, kk)
        if mpm.grid.mass[f, I, i_b] > gs.EPS:
            pos = (I + mpm.grid_offset) * mpm.dx
            mass = mpm.grid.mass[f, I, i_b] / mpm._particle_volume_scale
            mpm.grid.vel_out[f, I, i_b] = func_collide_with_rigid(
                pos,
                mpm.grid.vel_out[f, I, i_b],
                mass,
                i_b,
                n_geoms,
                geoms_state,
                geoms_info,
                links_state,
                rigid_info,
                sdf_info,
                collider_static_config,
            )


@qd.func(requires_top_level=True)
def func_mpm_sph_couple(f: int, mpm: qd.template(), sph: qd.template(), stencil_size: qd.template()):
    """Exchange grid momentum with SPH particles."""
    for ii, jj, kk, i_b in qd.ndrange(*mpm.grid_res, mpm._B):
        I = (ii, jj, kk)
        if mpm.grid.mass[f, I, i_b] > gs.EPS:
            pos = (I + mpm.grid_offset) * mpm.dx
            mass_mpm = mpm.grid.mass[f, I, i_b] / mpm._particle_volume_scale
            vel_mpm = mpm.grid.vel_out[f, I, i_b]
            base = func_pos_to_grid(pos - 0.5 * mpm.dx, sph.sh)

            sph_vel = qd.Vector([0.0, 0.0, 0.0])
            colliding_particles = 0
            for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                slot_idx = func_grid_to_slot(base + offset, sph.sh)
                for i in range(
                    sph.spatial_hasher.slot_start[slot_idx, i_b],
                    sph.spatial_hasher.slot_start[slot_idx, i_b] + sph.spatial_hasher.slot_size[slot_idx, i_b],
                ):
                    if qd.abs(pos - sph.particles_reordered.pos[i, i_b]).max() < mpm.dx * 0.5:
                        sph_vel += sph.particles_reordered.vel[i, i_b]
                        colliding_particles += 1
            if colliding_particles > 0:
                vel_old = vel_mpm
                vel_mpm = sph_vel / colliding_particles
                delta_mv = mass_mpm * (vel_mpm - vel_old)
                for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                    slot_idx = func_grid_to_slot(base + offset, sph.sh)
                    for i in range(
                        sph.spatial_hasher.slot_start[slot_idx, i_b],
                        sph.spatial_hasher.slot_start[slot_idx, i_b] + sph.spatial_hasher.slot_size[slot_idx, i_b],
                    ):
                        if qd.abs(pos - sph.particles_reordered.pos[i, i_b]).max() < mpm.dx * 0.5:
                            sph.particles_reordered.vel[i, i_b] -= delta_mv / sph.particles_info_reordered.mass[i, i_b]
                mpm.grid.vel_out[f, I, i_b] = vel_mpm


@qd.kernel
def kernel_mpm_sph_couple(
    f: qd.i32,
    mpm: qd.template(),
    sph: qd.template(),
    stencil_size: qd.template(),
):
    func_mpm_sph_couple(f, mpm, sph, stencil_size)


@qd.func(requires_top_level=True)
def func_mpm_pbd_couple(f: int, mpm: qd.template(), pbd: qd.template(), stencil_size: qd.template()):
    """Exchange grid momentum with free PBD particles."""
    for ii, jj, kk, i_b in qd.ndrange(*mpm.grid_res, mpm._B):
        I = (ii, jj, kk)
        if mpm.grid.mass[f, I, i_b] > gs.EPS:
            pos = (I + mpm.grid_offset) * mpm.dx
            mass_mpm = mpm.grid.mass[f, I, i_b] / mpm._particle_volume_scale
            vel_mpm = mpm.grid.vel_out[f, I, i_b]
            base = func_pos_to_grid(pos - 0.5 * mpm.dx, pbd.sh)

            pbd_vel = qd.Vector([0.0, 0.0, 0.0])
            colliding_particles = 0
            for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                slot_idx = func_grid_to_slot(base + offset, pbd.sh)
                for i in range(
                    pbd.spatial_hasher.slot_start[slot_idx, i_b],
                    pbd.spatial_hasher.slot_start[slot_idx, i_b] + pbd.spatial_hasher.slot_size[slot_idx, i_b],
                ):
                    if qd.abs(pos - pbd.particles_reordered.pos[i, i_b]).max() < mpm.dx * 0.5:
                        pbd_vel += pbd.particles_reordered.vel[i, i_b]
                        colliding_particles += 1
            if colliding_particles > 0:
                vel_old = vel_mpm
                vel_mpm = pbd_vel / colliding_particles
                delta_mv = mass_mpm * (vel_mpm - vel_old)
                for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                    slot_idx = func_grid_to_slot(base + offset, pbd.sh)
                    for i in range(
                        pbd.spatial_hasher.slot_start[slot_idx, i_b],
                        pbd.spatial_hasher.slot_start[slot_idx, i_b] + pbd.spatial_hasher.slot_size[slot_idx, i_b],
                    ):
                        if (
                            qd.abs(pos - pbd.particles_reordered.pos[i, i_b]).max() < mpm.dx * 0.5
                            and pbd.particles_reordered.free[i, i_b]
                        ):
                            pbd.particles_reordered.vel[i, i_b] -= delta_mv / pbd.particles_info_reordered.mass[i, i_b]
                mpm.grid.vel_out[f, I, i_b] = vel_mpm


@qd.kernel
def kernel_mpm_pbd_couple(
    f: qd.i32,
    mpm: qd.template(),
    pbd: qd.template(),
    stencil_size: qd.template(),
):
    func_mpm_pbd_couple(f, mpm, pbd, stencil_size)


@qd.func(requires_top_level=True)
def func_mpm_grid_end(f: int, mpm: qd.template()):
    """Apply the MPM grid boundary after every pairwise interaction."""
    for ii, jj, kk, i_b in qd.ndrange(*mpm.grid_res, mpm._B):
        I = (ii, jj, kk)
        if mpm.grid.mass[f, I, i_b] > gs.EPS:
            pos = (I + mpm.grid_offset) * mpm.dx
            _, mpm.grid.vel_out[f, I, i_b] = mpm.boundary.impose_pos_vel(pos, mpm.grid.vel_out[f, I, i_b])


@qd.kernel
def kernel_mpm_grid_end(f: qd.i32, mpm: qd.template()):
    for ii, jj, kk, i_b in qd.ndrange(*mpm.grid_res, mpm._B):
        I = (ii, jj, kk)
        if mpm.grid.mass[f, I, i_b] > gs.EPS:
            pos = (I + mpm.grid_offset) * mpm.dx
            _, mpm.grid.vel_out[f, I, i_b] = mpm.boundary.impose_pos_vel(pos, mpm.grid.vel_out[f, I, i_b])
