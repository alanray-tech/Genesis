"""Typed device data of the SPH solver, allocated and owned by SPHSolverSystem."""

import dataclasses
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.engine.core import Data
from genesis.utils.array_class import V, V_VEC, DataKind, of_kind
from genesis.utils.misc import qd_to_numpy
from genesis.utils.spatial_hash import SpatialHasherState, get_spatial_hasher_state


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SPHParticlesState:
    """Dynamic position, velocity, pressure and DFSPH state of particles."""

    kind: ClassVar[DataKind] = DataKind.STATE

    pos: qd.Tensor
    vel: qd.Tensor
    acc: qd.Tensor
    rho: qd.Tensor
    p: qd.Tensor
    dfsph_factor: qd.Tensor
    drho: qd.Tensor


def get_sph_particles_state(system, *, is_active: bool = True) -> SPHParticlesState:
    shape = (system._n_particles, system._B) if is_active else ()
    return SPHParticlesState(
        pos=V_VEC(3, dtype=gs.qd_float, shape=shape),
        vel=V_VEC(3, dtype=gs.qd_float, shape=shape),
        acc=V_VEC(3, dtype=gs.qd_float, shape=shape),
        rho=V(dtype=gs.qd_float, shape=shape),
        p=V(dtype=gs.qd_float, shape=shape),
        dfsph_factor=V(dtype=gs.qd_float, shape=shape),
        drho=V(dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SPHParticlesNG:
    """Non-differentiable active flag and reordered index of particles."""

    kind: ClassVar[DataKind] = DataKind.STATE

    reordered_idx: qd.Tensor = of_kind(DataKind.SCRATCH)
    active: qd.Tensor


def get_sph_particles_ng(system, *, is_active: bool = True) -> SPHParticlesNG:
    shape = (system._n_particles, system._B) if is_active else ()
    return SPHParticlesNG(
        reordered_idx=V(dtype=gs.qd_int, shape=shape),
        active=V(dtype=gs.qd_bool, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SPHParticlesInfo:
    """Material parameters and mass of particles."""

    kind: ClassVar[DataKind] = DataKind.INFO

    rho: qd.Tensor
    mass: qd.Tensor
    stiffness: qd.Tensor
    exponent: qd.Tensor
    mu: qd.Tensor
    gamma: qd.Tensor

    def __getitem__(self, i_p: int):
        return SPHParticleInfoView(self, i_p)


class SPHParticleInfoView:
    """Host compatibility view of one particle's material information."""

    def __init__(self, info: SPHParticlesInfo, i_p: int) -> None:
        self.info = info
        self.i_p = i_p

    @property
    def rho(self):
        return qd_to_numpy(self.info.rho, transpose=True)[self.i_p]

    @property
    def mass(self):
        return qd_to_numpy(self.info.mass, transpose=True)[self.i_p]

    @property
    def stiffness(self):
        return qd_to_numpy(self.info.stiffness, transpose=True)[self.i_p]

    @property
    def exponent(self):
        return qd_to_numpy(self.info.exponent, transpose=True)[self.i_p]

    @property
    def mu(self):
        return qd_to_numpy(self.info.mu, transpose=True)[self.i_p]

    @property
    def gamma(self):
        return qd_to_numpy(self.info.gamma, transpose=True)[self.i_p]


def get_sph_particles_info(system, *, is_reordered: bool = False) -> SPHParticlesInfo:
    shape = (system._n_particles, system._B) if is_reordered else (system._n_particles,)
    return SPHParticlesInfo(
        rho=V(dtype=gs.qd_float, shape=shape),
        mass=V(dtype=gs.qd_float, shape=shape),
        stiffness=V(dtype=gs.qd_float, shape=shape),
        exponent=V(dtype=gs.qd_float, shape=shape),
        mu=V(dtype=gs.qd_float, shape=shape),
        gamma=V(dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SPHParticlesRender:
    """Particle state copied for rendering."""

    kind: ClassVar[DataKind] = DataKind.DERIVED

    pos: qd.Tensor
    vel: qd.Tensor
    active: qd.Tensor


def get_sph_particles_render(system) -> SPHParticlesRender:
    shape = (system._n_particles, system._B)
    return SPHParticlesRender(
        pos=V_VEC(3, dtype=gs.qd_float, shape=shape),
        vel=V_VEC(3, dtype=gs.qd_float, shape=shape),
        active=V(dtype=gs.qd_bool, shape=shape),
    )


@qd.data_oriented
class SPHSolverData(Data):
    """Particle, reordered, render and spatial-hash data of the SPH solver."""

    def __init__(self, system) -> None:
        self.particles = get_sph_particles_state(system)
        self.particles_ng = get_sph_particles_ng(system)
        self.particles_info = get_sph_particles_info(system)
        self.particles_reordered = get_sph_particles_state(system)
        self.particles_ng_reordered = get_sph_particles_ng(system)
        self.particles_info_reordered = get_sph_particles_info(system, is_reordered=True)
        self.particles_render = get_sph_particles_render(system)
        self.spatial_hasher = get_spatial_hasher_state(system.sh, system._B)
