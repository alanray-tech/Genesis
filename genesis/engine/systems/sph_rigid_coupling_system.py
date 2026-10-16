"""System owning two-way coupling between SPH particles and rigid geoms."""

from typing import TYPE_CHECKING

import quadrants as qd

import genesis as gs
from genesis.engine.coupling_data import get_sph_rigid_coupling_data
from genesis.engine.core import Require, System, register_system
from genesis.engine.pairwise_coupling_runtime import (
    graph_sph_rigid_couple,
    kernel_reset_rigid_normals_field,
    kernel_reset_rigid_normals_ndarray,
    kernel_sph_rigid_couple,
)
from genesis.options.solvers import LegacyCouplerOptions

from .coupler_system import CouplerSystem
from .rigid_sdf_system import RigidSdfSystem
from .rigid_solver_system import RigidSolverSystem
from .simulator_engine import SimulatorEngine
from .sph_solver_system import SPHSolverSystem

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class SPHRigidCouplingSystem(System):
    """Two-way Pairwise coupling between SPH particles and rigid geoms."""

    manager = Require(CouplerSystem)
    sph = Require(SPHSolverSystem)
    rigid = Require(RigidSolverSystem)
    rigid_sdf = Require(RigidSdfSystem)

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    @System.action(kind="host")
    def reset(self):
        return self.run_reset

    @System.action(kind="stage")
    def graph_couple(self):
        return func_graph_sph_rigid_couple, self

    def initialize_data(self) -> None:
        self.rigid_sdf.activate()
        self.data = get_sph_rigid_coupling_data(self.sph, self.rigid)
        self.rigid_normal = self.data.rigid_normal
        self.rigid_normal_reordered = self.data.rigid_normal_reordered

    def build(self) -> None:
        self.manager.on_pairwise_couple(action=self.couple)
        self.manager.on_pairwise_reset(action=self.reset)

    def run_couple(self, f: int) -> None:
        kernel_sph_rigid_couple(
            f,
            self.sph,
            self,
            self.rigid.n_geoms,
            self.rigid._options.batch_links_info,
            self.rigid.dyn_state.geoms,
            self.rigid.dyn_info.geoms,
            self.rigid.dyn_state.links,
            self.rigid.dyn_info.links,
            self.rigid.rigid_info,
            self.rigid_sdf.sdf._sdf_info,
            self.rigid.collider_data.collider_config,
        )

    def run_reset(self, envs_idx) -> None:
        if envs_idx is None:
            self.data.rigid_normal.fill(0)
            return
        kernel = kernel_reset_rigid_normals_ndarray if gs.use_ndarray else kernel_reset_rigid_normals_field
        kernel(self.data.rigid_normal, self.sph._n_particles, self.rigid.n_geoms, envs_idx)


@register_system(SimulatorEngine)
def create_sph_rigid_coupling_system(scene: "Scene") -> SPHRigidCouplingSystem | None:
    options = scene.options.coupler
    if (
        isinstance(options, LegacyCouplerOptions)
        and scene.sim.sph_solver.is_active
        and scene.sim.rigid_solver.is_active
        and options.rigid_sph
    ):
        return SPHRigidCouplingSystem(scene)
    return None


@qd.func(requires_top_level=True)
def func_graph_sph_rigid_couple(system: qd.template(), f: int, t: float):
    graph_sph_rigid_couple(
        f,
        system.sph,
        system,
        system.rigid.n_geoms,
        system.rigid._options.batch_links_info,
        system.rigid.dyn_state.geoms,
        system.rigid.dyn_info.geoms,
        system.rigid.dyn_state.links,
        system.rigid.dyn_info.links,
        system.rigid.rigid_info,
        system.rigid_sdf.sdf._sdf_info,
        system.rigid.collider_data.collider_config,
    )
