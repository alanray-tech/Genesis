"""Shared low-level device functions for pairwise coupling Systems."""

import quadrants as qd

import genesis as gs
from genesis.utils import array_class


@qd.func
def func_vel_at_rigid_point(link_idx: int, i_b: int, pos_world, links_state: array_class.LinksState):
    vel_rot = links_state.cd_ang[link_idx, i_b].cross(pos_world - links_state.root_COM[link_idx, i_b])
    return vel_rot + links_state.cd_vel[link_idx, i_b]


@qd.func
def func_apply_rigid_coupling_force(
    link_idx: int,
    i_b: int,
    pos_world,
    force,
    links_state: array_class.LinksState,
):
    torque = (pos_world - links_state.root_COM[link_idx, i_b]).cross(force)
    links_state.cfrc_coupling_ang[link_idx, i_b] -= torque
    links_state.cfrc_coupling_vel[link_idx, i_b] -= force


@qd.func
def func_collide_in_rigid_geom(
    pos_world,
    vel,
    mass,
    normal_rigid,
    influence,
    geom_idx,
    i_b,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    rigid_info: array_class.RigidInfo,
):
    vel_rigid = func_vel_at_rigid_point(geoms_info.link_idx[geom_idx], i_b, pos_world, links_state)
    rvel = vel - vel_rigid
    rvel_normal_magnitude = rvel.dot(normal_rigid)
    if rvel_normal_magnitude < 0:
        rvel_tan = rvel - rvel_normal_magnitude * normal_rigid
        rvel_tan_norm = rvel_tan.norm(gs.EPS)
        rvel_tan = (
            rvel_tan
            / rvel_tan_norm
            * qd.max(0, rvel_tan_norm + rvel_normal_magnitude * geoms_info.coup_friction[geom_idx])
        )
        rvel_normal = -normal_rigid * rvel_normal_magnitude * geoms_info.coup_restitution[geom_idx]
        rvel_new = rvel_tan + rvel_normal
        vel_old = vel
        vel = vel_rigid + rvel_new * influence + rvel * (1 - influence)
        force = -(mass * (vel - vel_old)) / rigid_info.substep_dt[None]
        func_apply_rigid_coupling_force(geoms_info.link_idx[geom_idx], i_b, pos_world, force, links_state)
    return vel
