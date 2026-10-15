"""Typed device data of the tool solver, allocated and owned by ToolSolverSystem."""

import dataclasses
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.engine.core import Data
from genesis.utils.array_class import V, V_MAT, V_SCALAR_FROM, V_VEC, AutoInitMeta, DataKind, of_kind


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ToolEntityStateData:
    """Pose and velocity frames of one tool entity."""

    kind: ClassVar[DataKind] = DataKind.STATE

    pos: qd.Tensor
    quat: qd.Tensor
    vel: qd.Tensor
    ang: qd.Tensor
    latest_pos: qd.Tensor = of_kind(DataKind.DERIVED)


def get_tool_entity_state_data(entity):
    shape = (entity.sim.substeps_local + 1, entity.sim._B)
    return ToolEntityStateData(
        pos=V_VEC(3, dtype=gs.qd_float, shape=shape, needs_grad=True),
        quat=V_VEC(4, dtype=gs.qd_float, shape=shape, needs_grad=True),
        vel=V_VEC(3, dtype=gs.qd_float, shape=shape, needs_grad=True),
        ang=V_VEC(3, dtype=gs.qd_float, shape=shape, needs_grad=True),
        latest_pos=V_VEC(3, dtype=gs.qd_float, shape=(1,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ToolMeshInfo:
    """Signed distance field of one tool mesh and the transform from mesh space into its voxel space."""

    kind: ClassVar[DataKind] = DataKind.CONSTANT

    sdf_voxels: qd.Tensor
    T_mesh_to_sdf: qd.Tensor
    friction: qd.Tensor = of_kind(DataKind.INFO)
    coup_softness: qd.Tensor = of_kind(DataKind.INFO)


def get_tool_mesh_info(mesh):
    sdf_shape = mesh.sdf_voxels_np.shape if mesh.collision else ()
    info = ToolMeshInfo(
        sdf_voxels=V(dtype=gs.qd_float, shape=sdf_shape),
        T_mesh_to_sdf=V_MAT(4, 4, dtype=gs.qd_float, shape=()),
        friction=V_SCALAR_FROM(dtype=gs.qd_float, value=mesh.friction if mesh.collision else 0.0),
        coup_softness=V_SCALAR_FROM(dtype=gs.qd_float, value=mesh.material.coup_softness),
    )
    if mesh.collision:
        info.sdf_voxels.from_numpy(mesh.sdf_voxels_np)
        info.T_mesh_to_sdf.from_numpy(mesh.T_mesh_to_sdf_np)
    return info


@qd.data_oriented
class ToolMeshConfig(metaclass=AutoInitMeta):
    collision: bool
    sdf_res: int


def get_tool_mesh_config(mesh):
    return ToolMeshConfig(
        collision=mesh.collision,
        sdf_res=mesh.sdf_res,
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ToolEntityData:
    """Device data of one tool entity."""

    state: ToolEntityStateData
    mesh_info: ToolMeshInfo


def get_tool_entity_data(entity):
    return ToolEntityData(
        state=get_tool_entity_state_data(entity),
        mesh_info=get_tool_mesh_info(entity.mesh),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ToolSolverInfo:
    """Integration interval and step-to-substep ratio shared by every tool entity."""

    kind: ClassVar[DataKind] = DataKind.CONSTANT

    substep_dt: qd.Tensor
    substeps: qd.Tensor


def get_tool_solver_info(system):
    return ToolSolverInfo(
        substep_dt=V_SCALAR_FROM(dtype=gs.qd_float, value=system._substep_dt),
        substeps=V_SCALAR_FROM(dtype=gs.qd_int, value=system.substeps),
    )


@qd.data_oriented
class ToolSolverData(Data):
    """Per-entity device data and mesh configurations reached from cross-solver kernels."""

    def __init__(self, entities: tuple[ToolEntityData, ...], meshes: tuple[ToolMeshConfig, ...]) -> None:
        self.entities = entities
        self.meshes = meshes
        # WORKAROUND: Quadrants discovers ndarrays only through named attributes of a data-oriented kernel argument
        for i_e, entity_data in enumerate(entities):
            setattr(self, f"_entity_{i_e}", entity_data)


@qd.data_oriented
class ToolSolverConfig(metaclass=AutoInitMeta):
    boundary: object
