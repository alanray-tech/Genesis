import numpy as np
import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.mesh import compute_sdf_data, load_mesh

from .data import ToolEntityData, ToolSolverInfo


class Mesh:
    def __init__(self, entity, material, morph):
        self.entity = entity
        self.material = material
        self.collision = material.collision
        self.sdf_res = material.sdf_res
        self.scale = morph.scale
        self.raw_file = morph.file

        self.load_file()

    def load_file(self):
        self.process_mesh()
        self.raw_vertices = np.asarray(self.mesh.vertices, dtype=gs.np_float, order="C")
        self.raw_vertex_normals = np.asarray(self.mesh.vertex_normals, dtype=gs.np_float, order="C")
        self.faces_np = np.asarray(self.mesh.faces, dtype=gs.np_int, order="C").reshape((-1))

        # apply initial transforms (scale then quat then pos)
        scale = np.array(self.scale, dtype=gs.np_float)
        T_init = gu.scale_to_T(scale)
        self.raw_vertices = gu.transform_by_T(self.raw_vertices, T_init)
        if self.collision:
            self.T_mesh_to_sdf_np = self.T_mesh_to_sdf_np @ gu.inv_T(T_init)

        self.n_vertices = len(self.raw_vertices)
        self.n_faces = len(self.faces_np)

    def process_mesh(self):
        # Normalize mesh
        gs.logger.debug(f"Processing mesh: {self.raw_file}.")
        mesh_orig = load_mesh(self.raw_file)
        scale = np.linalg.norm(mesh_orig.extents, ord=float("inf"))
        center = np.mean(mesh_orig.bounds, axis=0)
        normalized_vertices = (mesh_orig.vertices - center) / scale
        self.mesh = trimesh.Trimesh(
            vertices=normalized_vertices,
            faces=mesh_orig.faces,
            vertex_normals=mesh_orig.vertex_normals,
            face_normals=mesh_orig.face_normals,
        )

        # generate sdf
        if self.collision:
            sdf_data = compute_sdf_data(self.mesh, self.sdf_res)
            self.friction = self.material.friction
            self.sdf_voxels_np = sdf_data["voxels"].astype(gs.np_float, order="C", copy=False)
            self.sdf_res = self.sdf_voxels_np.shape[0]
            self.T_mesh_to_sdf_np = sdf_data["T_mesh_to_sdf"].astype(gs.np_float, order="C", copy=False)


@qd.func
def func_sdf_voxels(pos_voxels: qd.types.vector(3), entity_data: ToolEntityData, mesh: qd.template()):
    """Interpolate the signed distance field at a position in voxel coordinates."""
    base = qd.floor(pos_voxels, gs.qd_int)
    signed_dist = gs.qd_float(0.0)
    if (base >= mesh.sdf_res - 1).any() or (base < 0).any():
        signed_dist = 1.0
    else:
        signed_dist = 0.0
        for offset in qd.static(qd.grouped(qd.ndrange(2, 2, 2))):
            voxel_pos = base + offset
            w_xyz = 1 - qd.abs(pos_voxels - voxel_pos)
            w = w_xyz[0] * w_xyz[1] * w_xyz[2]
            signed_dist += w * entity_data.mesh_info.sdf_voxels[voxel_pos]
    return signed_dist


@qd.func
def func_sdf(f: int, pos_world: qd.types.vector(3), i_b: int, entity_data: ToolEntityData, mesh: qd.template()):
    """Evaluate the signed distance of a world-space position from a tool mesh."""
    pos_mesh = gu.qd_inv_transform_by_trans_quat(
        pos_world, entity_data.state.pos[f, i_b], entity_data.state.quat[f, i_b]
    )
    pos_voxels = gu.qd_transform_by_T(pos_mesh, entity_data.mesh_info.T_mesh_to_sdf[None])
    return func_sdf_voxels(pos_voxels, entity_data, mesh)


@qd.func
def func_normal_voxels(pos_voxels: qd.types.vector(3), entity_data: ToolEntityData, mesh: qd.template()):
    """Compute the signed-distance normal in voxel coordinates by central finite differences."""
    delta = gs.qd_float(1e-2)
    normal_vec = qd.Vector([0, 0, 0], dt=gs.qd_float)
    for i in qd.static(range(3)):
        inc = pos_voxels
        dec = pos_voxels
        inc[i] += delta
        dec[i] -= delta
        normal_vec[i] = (func_sdf_voxels(inc, entity_data, mesh) - func_sdf_voxels(dec, entity_data, mesh)) / (
            2 * delta
        )
    return gu.qd_normalize(normal_vec, gs.EPS)


@qd.func
def func_normal(
    f: int,
    pos_world: qd.types.vector(3),
    i_b: int,
    entity_data: ToolEntityData,
    mesh: qd.template(),
):
    """Compute the outward normal of a tool mesh at a world-space position."""
    pos_mesh = gu.qd_inv_transform_by_trans_quat(
        pos_world, entity_data.state.pos[f, i_b], entity_data.state.quat[f, i_b]
    )
    pos_voxels = gu.qd_transform_by_T(pos_mesh, entity_data.mesh_info.T_mesh_to_sdf[None])
    normal_vec_voxels = func_normal_voxels(pos_voxels, entity_data, mesh)
    R_voxels_to_mesh = entity_data.mesh_info.T_mesh_to_sdf[None][:3, :3].inverse()
    normal_vec_mesh = R_voxels_to_mesh @ normal_vec_voxels
    normal_vec_world = gu.qd_transform_by_quat(normal_vec_mesh, entity_data.state.quat[f, i_b])
    return gu.qd_normalize(normal_vec_world, gs.EPS)


@qd.func
def func_velocity(
    f: int,
    pos_world: qd.types.vector(3),
    i_b: int,
    entity_data: ToolEntityData,
    solver_info: ToolSolverInfo,
):
    """Compute the velocity of a material point fixed to a tool entity."""
    pos_mesh = gu.qd_inv_transform_by_trans_quat(
        pos_world, entity_data.state.pos[f, i_b], entity_data.state.quat[f, i_b]
    )
    pos_world_new = gu.qd_transform_by_trans_quat(
        pos_mesh, entity_data.state.pos[f + 1, i_b], entity_data.state.quat[f + 1, i_b]
    )
    return (pos_world_new - pos_world) / solver_info.substep_dt[None]


@qd.func
def func_collide(
    f: int,
    pos_world: qd.types.vector(3),
    vel_mat: qd.types.vector(3),
    i_b: int,
    entity_data: ToolEntityData,
    solver_info: ToolSolverInfo,
    mesh: qd.template(),
):
    """Project inward material velocity against a tool mesh with friction and coupling softness."""
    if qd.static(mesh.collision):
        signed_dist = func_sdf(f, pos_world, i_b, entity_data, mesh)
        influence = qd.min(qd.exp(-signed_dist / max(gs.EPS, entity_data.mesh_info.coup_softness[None])), 1)
        if signed_dist <= 0.0 or influence > 0.1:
            vel_collider = func_velocity(f, pos_world, i_b, entity_data, solver_info)
            rel_v = vel_mat - vel_collider
            normal_vec = func_normal(f, pos_world, i_b, entity_data, mesh)
            normal_component = rel_v.dot(normal_vec)
            if normal_component < 0.0:
                rel_v_t = rel_v - normal_component * normal_vec
                rel_v_t_norm = rel_v_t.norm(gs.EPS)
                rel_v_t_friction = (
                    rel_v_t
                    / rel_v_t_norm
                    * qd.max(0, rel_v_t_norm + normal_component * entity_data.mesh_info.friction[None])
                )
                # FIXME: The branch establishes that flag is always one, so this formula can be simplified
                flag = qd.cast(normal_component < 0.0, gs.qd_float)
                rel_v_t = rel_v_t_friction * flag + rel_v_t * (1.0 - flag)
                vel_mat = vel_collider + rel_v_t * influence + rel_v * (1.0 - influence)
    return vel_mat
