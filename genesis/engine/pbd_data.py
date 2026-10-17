"""Typed device data of the PBD solver, allocated and owned by PBDSolverSystem."""

import dataclasses
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.engine.core import Data
from genesis.utils.array_class import V, V_VEC, DataKind, of_kind
from genesis.utils.spatial_hash import get_spatial_hasher_state


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class PBDParticlesInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    mass: qd.Tensor
    pos_rest: qd.Tensor
    rho_rest: qd.Tensor
    material_type: qd.Tensor
    mu_s: qd.Tensor
    mu_k: qd.Tensor
    air_resistance: qd.Tensor
    density_relaxation: qd.Tensor
    viscosity_relaxation: qd.Tensor


def get_pbd_particles_info(system, *, is_reordered: bool = False) -> PBDParticlesInfo:
    shape = (system._n_particles, system._B) if is_reordered else (system._n_particles,)
    return PBDParticlesInfo(
        mass=V(dtype=gs.qd_float, shape=shape),
        pos_rest=V_VEC(3, dtype=gs.qd_float, shape=shape),
        rho_rest=V(dtype=gs.qd_float, shape=shape),
        material_type=V(dtype=gs.qd_int, shape=shape),
        mu_s=V(dtype=gs.qd_float, shape=shape),
        mu_k=V(dtype=gs.qd_float, shape=shape),
        air_resistance=V(dtype=gs.qd_float, shape=shape),
        density_relaxation=V(dtype=gs.qd_float, shape=shape),
        viscosity_relaxation=V(dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class PBDParticlesState:
    kind: ClassVar[DataKind] = DataKind.STATE

    free: qd.Tensor
    pos: qd.Tensor
    ipos: qd.Tensor
    dpos: qd.Tensor
    vel: qd.Tensor
    lam: qd.Tensor
    rho: qd.Tensor


def get_pbd_particles_state(system) -> PBDParticlesState:
    shape = (system._n_particles, system._B)
    return PBDParticlesState(
        free=V(dtype=gs.qd_bool, shape=shape),
        pos=V_VEC(3, dtype=gs.qd_float, shape=shape),
        ipos=V_VEC(3, dtype=gs.qd_float, shape=shape),
        dpos=V_VEC(3, dtype=gs.qd_float, shape=shape),
        vel=V_VEC(3, dtype=gs.qd_float, shape=shape),
        lam=V(dtype=gs.qd_float, shape=shape),
        rho=V(dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class PBDParticlesNG:
    kind: ClassVar[DataKind] = DataKind.STATE

    reordered_idx: qd.Tensor = of_kind(DataKind.SCRATCH)
    active: qd.Tensor


def get_pbd_particles_ng(system) -> PBDParticlesNG:
    shape = (system._n_particles, system._B)
    return PBDParticlesNG(
        reordered_idx=V(dtype=gs.qd_int, shape=shape),
        active=V(dtype=gs.qd_bool, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class PBDParticlesRender:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    pos: qd.Tensor
    vel: qd.Tensor
    active: qd.Tensor


def get_pbd_particles_render(system) -> PBDParticlesRender:
    shape = (system._n_particles, system._B)
    return PBDParticlesRender(
        pos=V_VEC(3, dtype=gs.qd_float, shape=shape),
        vel=V_VEC(3, dtype=gs.qd_float, shape=shape),
        active=V(dtype=gs.qd_bool, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class PBDEdgesInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    len_rest: qd.Tensor
    stretch_compliance: qd.Tensor
    stretch_relaxation: qd.Tensor
    v1: qd.Tensor
    v2: qd.Tensor


def get_pbd_edges_info(system) -> PBDEdgesInfo:
    shape = (max(1, system._n_edges),)
    return PBDEdgesInfo(
        len_rest=V(dtype=gs.qd_float, shape=shape),
        stretch_compliance=V(dtype=gs.qd_float, shape=shape),
        stretch_relaxation=V(dtype=gs.qd_float, shape=shape),
        v1=V(dtype=gs.qd_int, shape=shape),
        v2=V(dtype=gs.qd_int, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class PBDInnerEdgesInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    len_rest: qd.Tensor
    bending_compliance: qd.Tensor
    bending_relaxation: qd.Tensor
    v1: qd.Tensor
    v2: qd.Tensor
    v3: qd.Tensor
    v4: qd.Tensor


def get_pbd_inner_edges_info(system) -> PBDInnerEdgesInfo:
    shape = (max(1, system._n_inner_edges),)
    return PBDInnerEdgesInfo(
        len_rest=V(dtype=gs.qd_float, shape=shape),
        bending_compliance=V(dtype=gs.qd_float, shape=shape),
        bending_relaxation=V(dtype=gs.qd_float, shape=shape),
        v1=V(dtype=gs.qd_int, shape=shape),
        v2=V(dtype=gs.qd_int, shape=shape),
        v3=V(dtype=gs.qd_int, shape=shape),
        v4=V(dtype=gs.qd_int, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class PBDElementsInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    vol_rest: qd.Tensor
    volume_compliance: qd.Tensor
    volume_relaxation: qd.Tensor
    v1: qd.Tensor
    v2: qd.Tensor
    v3: qd.Tensor
    v4: qd.Tensor


def get_pbd_elements_info(system) -> PBDElementsInfo:
    shape = (max(1, system._n_elems),)
    return PBDElementsInfo(
        vol_rest=V(dtype=gs.qd_float, shape=shape),
        volume_compliance=V(dtype=gs.qd_float, shape=shape),
        volume_relaxation=V(dtype=gs.qd_float, shape=shape),
        v1=V(dtype=gs.qd_int, shape=shape),
        v2=V(dtype=gs.qd_int, shape=shape),
        v3=V(dtype=gs.qd_int, shape=shape),
        v4=V(dtype=gs.qd_int, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class PBDVisualInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    support_idxs: qd.Tensor
    support_weights: qd.Tensor
    uvs: qd.Tensor
    faces: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class PBDVisualState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    pos: qd.Tensor
    active: qd.Tensor


@qd.data_oriented
class PBDSolverData(Data):
    def __init__(self, system) -> None:
        self.particles_info = get_pbd_particles_info(system)
        self.particles_info_reordered = get_pbd_particles_info(system, is_reordered=True)
        self.particles = get_pbd_particles_state(system)
        self.particles_reordered = get_pbd_particles_state(system)
        self.particles_ng = get_pbd_particles_ng(system)
        self.particles_ng_reordered = get_pbd_particles_ng(system)
        self.particles_render = get_pbd_particles_render(system)
        self.edges_info = get_pbd_edges_info(system)
        self.inner_edges_info = get_pbd_inner_edges_info(system)
        self.elems_info = get_pbd_elements_info(system)
        vvert_shape = (max(system._n_vverts, 1),)
        self.visual_info = PBDVisualInfo(
            support_idxs=V_VEC(system._n_vvert_supports, dtype=gs.qd_int, shape=vvert_shape),
            support_weights=V_VEC(system._n_vvert_supports, dtype=gs.qd_float, shape=vvert_shape),
            uvs=V_VEC(2, dtype=gs.qd_float, shape=vvert_shape),
            faces=V_VEC(3, dtype=gs.qd_int, shape=(max(system._n_vfaces, 1),)),
        )
        visual_shape = (max(system._n_vverts, 1), system._B)
        self.visual_state = PBDVisualState(
            pos=V_VEC(3, dtype=gs.qd_float, shape=visual_shape),
            active=V(dtype=gs.qd_bool, shape=visual_shape),
        )
        self.spatial_hasher = get_spatial_hasher_state(system.sh, system._B)
