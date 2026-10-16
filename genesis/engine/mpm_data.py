"""Typed device data of the MPM solver, allocated and owned by MPMSolverSystem."""

import dataclasses
from typing import ClassVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.core import Data
from genesis.utils.array_class import V, V_MAT, V_VEC, AutoInitMeta, DataKind, of_kind


@qd.data_oriented
class MPMMaterialConfig(metaclass=AutoInitMeta):
    """Compile-time material dispatch table captured by every MPM graph."""

    indices: tuple[int, ...]
    update_F_S_Jp: tuple
    update_stress: tuple
    needs_svd: bool


def get_mpm_material_config(solver) -> MPMMaterialConfig:
    return MPMMaterialConfig(
        indices=tuple(solver._materials_idx),
        update_F_S_Jp=tuple(solver._materials_update_F_S_Jp),
        update_stress=tuple(solver._materials_update_stress),
        needs_svd=solver.needs_svd,
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPMParticlesState:
    kind: ClassVar[DataKind] = DataKind.STATE

    pos: qd.Tensor
    vel: qd.Tensor
    C: qd.Tensor
    F: qd.Tensor
    F_tmp: qd.Tensor = of_kind(DataKind.SCRATCH)
    U: qd.Tensor = of_kind(DataKind.SCRATCH)
    V: qd.Tensor = of_kind(DataKind.SCRATCH)
    S: qd.Tensor = of_kind(DataKind.SCRATCH)
    actu: qd.Tensor
    Jp: qd.Tensor


def get_mpm_particles_state(system) -> MPMParticlesState:
    shape = (system._sim.substeps_local + 1, system._n_particles, system._B)
    needs_grad = system._sim.requires_grad
    return MPMParticlesState(
        pos=V_VEC(3, dtype=gs.qd_float, shape=shape, needs_grad=needs_grad),
        vel=V_VEC(3, dtype=gs.qd_float, shape=shape, needs_grad=needs_grad),
        C=V_MAT(3, 3, dtype=gs.qd_float, shape=shape, needs_grad=needs_grad),
        F=V_MAT(3, 3, dtype=gs.qd_float, shape=shape, needs_grad=needs_grad),
        F_tmp=V_MAT(3, 3, dtype=gs.qd_float, shape=shape, needs_grad=needs_grad),
        U=V_MAT(3, 3, dtype=gs.qd_float, shape=shape, needs_grad=needs_grad),
        V=V_MAT(3, 3, dtype=gs.qd_float, shape=shape, needs_grad=needs_grad),
        S=V_MAT(3, 3, dtype=gs.qd_float, shape=shape, needs_grad=needs_grad),
        actu=V(dtype=gs.qd_float, shape=shape, needs_grad=needs_grad),
        Jp=V(dtype=gs.qd_float, shape=shape, needs_grad=needs_grad),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPMParticlesNG:
    kind: ClassVar[DataKind] = DataKind.STATE

    active: qd.Tensor


def get_mpm_particles_ng(system) -> MPMParticlesNG:
    shape = (system._sim.substeps_local + 1, system._n_particles, system._B)
    return MPMParticlesNG(active=V(dtype=gs.qd_bool, shape=shape))


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPMParticlesInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    material_idx: qd.Tensor
    mass: qd.Tensor
    default_Jp: qd.Tensor
    free: qd.Tensor
    muscle_group: qd.Tensor
    muscle_direction: qd.Tensor


def get_mpm_particles_info(system) -> MPMParticlesInfo:
    shape = (system._n_particles,)
    return MPMParticlesInfo(
        material_idx=V(dtype=gs.qd_int, shape=shape),
        mass=V(dtype=gs.qd_float, shape=shape),
        default_Jp=V(dtype=gs.qd_float, shape=shape),
        free=V(dtype=gs.qd_bool, shape=shape),
        muscle_group=V(dtype=gs.qd_int, shape=shape),
        muscle_direction=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPMParticlesRender:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    pos: qd.Tensor
    vel: qd.Tensor
    active: qd.Tensor


def get_mpm_particles_render(system) -> MPMParticlesRender:
    shape = (system._n_particles, system._B)
    return MPMParticlesRender(
        pos=V_VEC(3, dtype=gs.qd_float, shape=shape),
        vel=V_VEC(3, dtype=gs.qd_float, shape=shape),
        active=V(dtype=gs.qd_bool, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPMGridState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    mass: qd.Tensor
    vel_in: qd.Tensor
    vel_out: qd.Tensor


def get_mpm_grid_state(system) -> MPMGridState:
    shape = (system._sim.substeps_local, *system._grid_res, system._B)
    needs_grad = system._sim.requires_grad
    return MPMGridState(
        mass=V(dtype=gs.qd_float, shape=shape, needs_grad=needs_grad),
        vel_in=V_VEC(3, dtype=gs.qd_float, shape=shape, needs_grad=needs_grad),
        vel_out=V_VEC(3, dtype=gs.qd_float, shape=shape, needs_grad=needs_grad),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPMGridDirtyState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    flag: qd.Tensor
    cells: qd.Tensor
    count: qd.Tensor


def get_mpm_grid_dirty_state(system) -> MPMGridDirtyState:
    is_active = not system._sim.requires_grad
    grid_shape = tuple(system._grid_res) if is_active else ()
    n_cells = int(np.prod(system._grid_res)) if is_active else 0
    return MPMGridDirtyState(
        flag=V(dtype=gs.qd_int, shape=grid_shape),
        cells=V(dtype=gs.qd_int, shape=(n_cells,) if is_active else ()),
        count=V(dtype=gs.qd_int, shape=(1,) if is_active else ()),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPMVisualInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    support_idxs: qd.Tensor
    support_weights: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPMVisualState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    pos: qd.Tensor
    active: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPMParticleConstraints:
    kind: ClassVar[DataKind] = DataKind.STATE

    is_constrained: qd.Tensor
    target_pos: qd.Tensor
    stiffness: qd.Tensor
    link_idx: qd.Tensor
    link_local_pos: qd.Tensor


def get_mpm_particle_constraints(system) -> MPMParticleConstraints:
    shape = (system._n_particles, system._B)
    constraints = MPMParticleConstraints(
        is_constrained=V(dtype=gs.qd_bool, shape=shape),
        target_pos=V_VEC(3, dtype=gs.qd_float, shape=shape),
        stiffness=V(dtype=gs.qd_float, shape=shape),
        link_idx=V(dtype=gs.qd_int, shape=shape),
        link_local_pos=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )
    constraints.is_constrained.fill(False)
    constraints.link_idx.fill(-1)
    return constraints


@qd.data_oriented
class MPMSolverData(Data):
    def __init__(self, system) -> None:
        self.particles = get_mpm_particles_state(system)
        self.particles_ng = get_mpm_particles_ng(system)
        self.particles_info = get_mpm_particles_info(system)
        self.particles_render = get_mpm_particles_render(system)
        self.grid = get_mpm_grid_state(system)
        self.grid_dirty = get_mpm_grid_dirty_state(system)
        vvert_shape = (max(1, system._n_vverts),)
        self.visual_info = MPMVisualInfo(
            support_idxs=V_VEC(system._n_vvert_supports, dtype=gs.qd_int, shape=vvert_shape),
            support_weights=V_VEC(system._n_vvert_supports, dtype=gs.qd_float, shape=vvert_shape),
        )
        visual_shape = (max(1, system._n_vverts), system._B)
        self.visual_state = MPMVisualState(
            pos=V_VEC(3, dtype=gs.qd_float, shape=visual_shape),
            active=V(dtype=gs.qd_bool, shape=visual_shape),
        )
