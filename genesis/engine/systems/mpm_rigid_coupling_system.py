"""System owning MPM-Rigid collision and optional CPIC preprocessing."""

from typing import TYPE_CHECKING

import quadrants as qd

import genesis as gs
from genesis.engine.coupling_data import get_mpm_rigid_coupling_data
from genesis.engine.core import Require, System, register_system
from genesis.engine.pairwise_coupling_runtime import (
    graph_mpm_rigid_couple,
    graph_mpm_surface_to_particle,
    kernel_mpm_rigid_couple,
    kernel_mpm_surface_to_particle,
    kernel_reset_rigid_normals_field,
    kernel_reset_rigid_normals_ndarray,
)
from genesis.options.solvers import LegacyCouplerOptions

from .coupler_system import CouplerSystem
from .mpm_solver_system import MPMSolverSystem
from .rigid_sdf_system import RigidSdfSystem
from .rigid_solver_system import RigidSolverSystem
from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class MPMRigidCouplingSystem(System):
    """MPM-Rigid grid collision and optional CPIC preprocessing."""

    manager = Require(CouplerSystem)
    mpm = Require(MPMSolverSystem)
    rigid = Require(RigidSolverSystem)
    rigid_sdf = Require(RigidSdfSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.data = None

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="host")
    def preprocess(self):
        return self.run_preprocess

    @System.action(kind="host")
    def reset(self):
        return self.run_reset

    def initialize_data(self) -> None:
        self.rigid_sdf.activate()
        if self.mpm._enable_CPIC:
            self.data = get_mpm_rigid_coupling_data(self.mpm, self.rigid)
            self.cpic_flag = self.data.cpic_flag
            self.rigid_normal = self.data.rigid_normal

    def build(self) -> None:
        if self.mpm.solver.enable_CPIC:
            self.manager.on_pairwise_mpm_rigid_preprocess(action=self.preprocess)
            self.manager.on_pairwise_reset(action=self.reset)
        self.manager.on_pairwise_mpm_rigid_couple(action=self.couple)
        self.manager.on_pairwise_mpm_rigid_grad(action=self.couple_grad)

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    @System.action(kind="host")
    def couple_grad(self):
        return self.run_couple_grad

    @System.action(kind="stage")
    def graph_preprocess(self):
        return func_graph_mpm_rigid_preprocess, self

    @System.action(kind="stage")
    def graph_couple(self):
        return func_graph_mpm_rigid_couple, self

    def run_preprocess(self, f: int) -> None:
        kernel_mpm_surface_to_particle(
            f,
            self.mpm,
            self.data,
            self.rigid.n_geoms,
            self.rigid.dyn_state.geoms,
            self.rigid.dyn_info.geoms,
            self.rigid.rigid_info,
            self.rigid_sdf.sdf._sdf_info,
            self.rigid.collider_data.collider_config,
        )

    def run_couple(self, f: int) -> None:
        kernel_mpm_rigid_couple(
            f,
            self.mpm,
            self.rigid.dyn_state.geoms,
            self.rigid.dyn_info.geoms,
            self.rigid.dyn_state.links,
            self.rigid.rigid_info,
            self.rigid_sdf.sdf._sdf_info,
            self.rigid.collider_data.collider_config,
            self.rigid.n_geoms,
        )

    def run_couple_grad(self, f: int) -> None:
        kernel_mpm_rigid_couple.grad(
            f,
            self.mpm,
            self.rigid.dyn_state.geoms,
            self.rigid.dyn_info.geoms,
            self.rigid.dyn_state.links,
            self.rigid.rigid_info,
            self.rigid_sdf.sdf._sdf_info,
            self.rigid.collider_data.collider_config,
            self.rigid.n_geoms,
        )

    def run_reset(self, envs_idx) -> None:
        if envs_idx is None:
            self.data.rigid_normal.fill(0)
            return
        kernel = kernel_reset_rigid_normals_ndarray if gs.use_ndarray else kernel_reset_rigid_normals_field
        kernel(self.data.rigid_normal, self.mpm._n_particles, self.rigid.n_geoms, envs_idx)


@register_system(SimulatorEngine)
def create_mpm_rigid_coupling_system(scene: "Scene") -> MPMRigidCouplingSystem | None:
    options = scene.options.coupler
    if (
        isinstance(options, LegacyCouplerOptions)
        and scene.sim.mpm_solver.is_active
        and scene.sim.rigid_solver.is_active
        and options.rigid_mpm
    ):
        return MPMRigidCouplingSystem(scene)
    return None


@qd.func(requires_top_level=True)
def func_graph_mpm_rigid_preprocess(system: qd.template(), f: int, t: float):
    graph_mpm_surface_to_particle(
        f,
        system.mpm,
        system,
        system.rigid.n_geoms,
        system.rigid.dyn_state.geoms,
        system.rigid.dyn_info.geoms,
        system.rigid.rigid_info,
        system.rigid_sdf.sdf._sdf_info,
        system.rigid.collider_data.collider_config,
    )


@qd.func(requires_top_level=True)
def func_graph_mpm_rigid_couple(system: qd.template(), f: int, t: float):
    graph_mpm_rigid_couple(
        f,
        system.mpm,
        system.rigid.dyn_state.geoms,
        system.rigid.dyn_info.geoms,
        system.rigid.dyn_state.links,
        system.rigid.rigid_info,
        system.rigid_sdf.sdf._sdf_info,
        system.rigid.collider_data.collider_config,
        system.rigid.n_geoms,
    )
