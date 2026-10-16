"""System owning the Rigid-FEM coupling pair."""

from typing import TYPE_CHECKING

from genesis.engine import fem_runtime
from genesis.engine.core import Require, System, register_system
from genesis.engine.fem_pairwise_coupling_runtime import kernel_rigid_fem_couple
from genesis.options.solvers import LegacyCouplerOptions

from .coupler_system import CouplerSystem
from .fem_solver_system import FEMSolverSystem
from .rigid_sdf_system import RigidSdfSystem
from .rigid_solver_system import RigidSolverSystem
from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class RigidFEMCouplingSystem(System):
    """Collision and linked-vertex constraints between rigid and FEM state."""

    manager = Require(CouplerSystem)
    fem = Require(FEMSolverSystem)
    rigid = Require(RigidSolverSystem)
    rigid_sdf = Require(RigidSdfSystem)

    @System.action(kind="host")
    def initialize(self):
        return self.rigid_sdf.activate

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    @System.action(kind="host")
    def constraints(self):
        return self.run_constraints

    @System.action(kind="host")
    def couple_grad(self):
        return self.run_couple_grad

    def build(self) -> None:
        self.manager.on_pairwise_couple(action=self.couple)
        self.fem.on_post_pairwise_constraint(action=self.constraints)
        self.manager.on_pairwise_grad(action=self.couple_grad)

    def run_couple(self, f: int) -> None:
        kernel_rigid_fem_couple(
            f,
            self.rigid.dyn_state.geoms,
            self.rigid.dyn_info.geoms,
            self.rigid.dyn_state.links,
            self.rigid.rigid_info,
            self.rigid_sdf.sdf._sdf_info,
            self.fem,
            self.rigid.n_geoms,
            self.rigid.collider_data.collider_config,
        )

    def run_constraints(self, f: int) -> None:
        if self.fem.solver._constraints_initialized:
            fem_runtime._kernel_update_linked_vertex_constraints(self.fem, self.rigid.dyn_state.links)

    def run_couple_grad(self, f: int) -> None:
        kernel_rigid_fem_couple.grad(
            f,
            self.rigid.dyn_state.geoms,
            self.rigid.dyn_info.geoms,
            self.rigid.dyn_state.links,
            self.rigid.rigid_info,
            self.rigid_sdf.sdf._sdf_info,
            self.fem,
            self.rigid.n_geoms,
            self.rigid.collider_data.collider_config,
        )


@register_system(SimulatorEngine)
def create_rigid_fem_coupling_system(scene: "Scene") -> RigidFEMCouplingSystem | None:
    options = scene.options.coupler
    if (
        isinstance(options, LegacyCouplerOptions)
        and scene.sim.fem_solver.is_active
        and scene.sim.rigid_solver.is_active
        and options.rigid_fem
    ):
        return RigidFEMCouplingSystem(scene)
    return None
