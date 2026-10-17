"""Typed device data of the FEM solver, allocated and owned by FEMSolverSystem."""

import dataclasses
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.engine.core import Data
from genesis.utils.array_class import V, V_MAT, V_VEC, AutoInitMeta, DataKind


@qd.data_oriented
class FEMMaterialConfig(metaclass=AutoInitMeta):
    """Compile-time material dispatch table captured by every FEM graph."""

    indices: tuple[int, ...]
    materials: tuple
    update_stress: tuple
    compute_energy_gradient_hessian: tuple
    compute_energy: tuple


def get_fem_material_config(solver) -> FEMMaterialConfig:
    return FEMMaterialConfig(
        indices=tuple(solver._mats_idx),
        materials=tuple(solver._mats),
        update_stress=tuple(solver._mats_update_stress),
        compute_energy_gradient_hessian=tuple(solver._mats_compute_energy_gradient_hessian),
        compute_energy=tuple(solver._mats_compute_energy),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMBatchState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    active: qd.Tensor
    pcg_active: qd.Tensor
    linesearch_active: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMPCGState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    rTr: qd.Tensor
    rTz: qd.Tensor
    rTr_new: qd.Tensor
    rTz_new: qd.Tensor
    pTAp: qd.Tensor
    alpha: qd.Tensor
    beta: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMLinesearchState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    prev_energy: qd.Tensor
    energy: qd.Tensor
    step_size: qd.Tensor
    m: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMVerticesState:
    kind: ClassVar[DataKind] = DataKind.STATE

    pos: qd.Tensor
    vel: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMElementsState:
    kind: ClassVar[DataKind] = DataKind.STATE

    actu: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMElementsNG:
    kind: ClassVar[DataKind] = DataKind.STATE

    active: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMElementsInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    el2v: qd.Tensor
    mu: qd.Tensor
    lam: qd.Tensor
    mass_scaled: qd.Tensor
    mat_idx: qd.Tensor
    B: qd.Tensor
    V: qd.Tensor
    V_scaled: qd.Tensor
    friction_mu: qd.Tensor
    muscle_group: qd.Tensor
    muscle_direction: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMElementEnergy:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    energy: qd.Tensor
    gradient: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMVertexEnergy:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    inertia: qd.Tensor
    force: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMVerticesInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    mass: qd.Tensor
    mass_inv: qd.Tensor
    mass_over_dt2: qd.Tensor
    friction_mu: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMPCGVerticesState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    diag3x3: qd.Tensor
    prec: qd.Tensor
    x: qd.Tensor
    r: qd.Tensor
    z: qd.Tensor
    p: qd.Tensor
    Ap: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMLinesearchVerticesState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    x_prev: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMSurfaceInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    tri2v: qd.Tensor
    tri2el: qd.Tensor
    active: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMVisualInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    vert_idx: qd.Tensor
    uvs: qd.Tensor
    faces: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMVisualState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    pos: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMVertexConstraints:
    kind: ClassVar[DataKind] = DataKind.STATE

    is_constrained: qd.Tensor
    target_pos: qd.Tensor
    is_soft_constraint: qd.Tensor
    stiffness: qd.Tensor
    link_idx: qd.Tensor
    link_offset_pos: qd.Tensor
    link_init_quat: qd.Tensor


def get_fem_vertex_constraints(system) -> FEMVertexConstraints:
    shape = (system.n_vertices, system._B)
    constraints = FEMVertexConstraints(
        is_constrained=V(dtype=gs.qd_bool, shape=shape),
        target_pos=V_VEC(3, dtype=gs.qd_float, shape=shape),
        is_soft_constraint=V(dtype=gs.qd_bool, shape=shape),
        stiffness=V(dtype=gs.qd_float, shape=shape),
        link_idx=V(dtype=gs.qd_int, shape=shape),
        link_offset_pos=V_VEC(3, dtype=gs.qd_float, shape=shape),
        link_init_quat=V_VEC(4, dtype=gs.qd_float, shape=shape),
    )
    constraints.is_constrained.fill(False)
    constraints.link_idx.fill(-1)
    return constraints


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FEMSurfaceMeta:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    vertices: qd.Tensor
    elements: qd.Tensor
    vertex_mass: qd.Tensor


@qd.data_oriented
class FEMSolverData(Data):
    def __init__(self, system) -> None:
        n_frames = system._sim.substeps_local + 1
        self.batch = FEMBatchState(
            active=V(dtype=gs.qd_bool, shape=(system._B,)),
            pcg_active=V(dtype=gs.qd_bool, shape=(system._B,)),
            linesearch_active=V(dtype=gs.qd_bool, shape=(system._B,)),
        )
        self.pcg = FEMPCGState(
            **{field.name: V(dtype=gs.qd_float, shape=(system._B,)) for field in dataclasses.fields(FEMPCGState)}
        )
        self.linesearch = FEMLinesearchState(
            **{field.name: V(dtype=gs.qd_float, shape=(system._B,)) for field in dataclasses.fields(FEMLinesearchState)}
        )

        state_shape = (n_frames, system.n_vertices, system._B)
        self.vertices = FEMVerticesState(
            pos=V_VEC(3, dtype=gs.qd_float, shape=state_shape, needs_grad=True),
            vel=V_VEC(3, dtype=gs.qd_float, shape=state_shape, needs_grad=True),
        )
        element_state_shape = (n_frames, system.n_elements, system._B)
        self.elements = FEMElementsState(
            actu=V(dtype=gs.qd_float, shape=element_state_shape, needs_grad=True),
        )
        self.elements_ng = FEMElementsNG(active=V(dtype=gs.qd_bool, shape=element_state_shape))
        element_shape = (system.n_elements,)
        self.elements_info = FEMElementsInfo(
            el2v=V_VEC(4, dtype=gs.qd_int, shape=element_shape),
            mu=V(dtype=gs.qd_float, shape=element_shape),
            lam=V(dtype=gs.qd_float, shape=element_shape),
            mass_scaled=V(dtype=gs.qd_float, shape=element_shape),
            mat_idx=V(dtype=gs.qd_int, shape=element_shape),
            B=V_MAT(3, 3, dtype=gs.qd_float, shape=element_shape),
            V=V(dtype=gs.qd_float, shape=element_shape),
            V_scaled=V(dtype=gs.qd_float, shape=element_shape),
            friction_mu=V(dtype=gs.qd_float, shape=element_shape),
            muscle_group=V(dtype=gs.qd_int, shape=element_shape),
            muscle_direction=V_VEC(3, dtype=gs.qd_float, shape=element_shape),
        )

        batch_element_shape = (system._B, system.n_elements)
        self.element_energy = FEMElementEnergy(
            energy=V(dtype=gs.qd_float, shape=batch_element_shape),
            gradient=V_MAT(3, 3, dtype=gs.qd_float, shape=batch_element_shape),
        )
        # Material callbacks take this store as an unannotated template and index it as
        # (batch, deformation-column, deformation-column, element)[matrix-entry]. Preserve the packed field and
        # exact axis convention; turning it into an ndarray makes Quadrants pass an AnyArray into those callbacks.
        self.element_hessian = qd.field(
            dtype=gs.qd_mat3,
            shape=(system._B, 3, 3, system.n_elements),
        )
        batch_vertex_shape = (system._B, system.n_vertices)
        self.vertex_energy = FEMVertexEnergy(
            inertia=V_VEC(3, dtype=gs.qd_float, shape=batch_vertex_shape),
            force=V_VEC(3, dtype=gs.qd_float, shape=batch_vertex_shape),
        )
        self.vertices_info = FEMVerticesInfo(
            mass=V(dtype=gs.qd_float, shape=(system.n_vertices,)),
            mass_inv=V(dtype=gs.qd_float, shape=(system.n_vertices,)),
            mass_over_dt2=V(dtype=gs.qd_float, shape=(system.n_vertices,)),
            friction_mu=V(dtype=gs.qd_float, shape=(system.n_vertices,)),
        )
        self.pcg_vertices = FEMPCGVerticesState(
            diag3x3=V_MAT(3, 3, dtype=gs.qd_float, shape=batch_vertex_shape),
            prec=V_MAT(3, 3, dtype=gs.qd_float, shape=batch_vertex_shape),
            x=V_VEC(3, dtype=gs.qd_float, shape=batch_vertex_shape),
            r=V_VEC(3, dtype=gs.qd_float, shape=batch_vertex_shape),
            z=V_VEC(3, dtype=gs.qd_float, shape=batch_vertex_shape),
            p=V_VEC(3, dtype=gs.qd_float, shape=batch_vertex_shape),
            Ap=V_VEC(3, dtype=gs.qd_float, shape=batch_vertex_shape),
        )
        self.linesearch_vertices = FEMLinesearchVerticesState(
            x_prev=V_VEC(3, dtype=gs.qd_float, shape=batch_vertex_shape),
        )
        self.surface = FEMSurfaceInfo(
            tri2v=V_VEC(3, dtype=gs.qd_int, shape=(system.n_surfaces,)),
            tri2el=V(dtype=gs.qd_int, shape=(system.n_surfaces,)),
            active=V(dtype=gs.qd_bool, shape=(system.n_surfaces,)),
        )

        n_vverts = max(system._n_vverts, 1)
        self.visual_info = FEMVisualInfo(
            vert_idx=V(dtype=gs.qd_int, shape=(n_vverts,)),
            uvs=V_VEC(2, dtype=gs.qd_float, shape=(n_vverts,)),
            faces=V_VEC(3, dtype=gs.qd_int, shape=(max(system._n_vfaces, 1),)),
        )
        self.visual_state = FEMVisualState(
            pos=V_VEC(3, dtype=gs.qd_float, shape=(n_vverts, system._B)),
        )
        self.tet_wrong_order = V(dtype=gs.qd_bool, shape=())
        self.vertices_on_surface = V(dtype=gs.qd_bool, shape=(system.n_vertices,))
        self.elements_on_surface = V(dtype=gs.qd_bool, shape=(system.n_elements,))
        self.surface_meta: FEMSurfaceMeta | None = None
        self.vertex_constraints: FEMVertexConstraints | None = None
