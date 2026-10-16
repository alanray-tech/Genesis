"""System owning the FEM-MPM coupling pair."""

from typing import TYPE_CHECKING

from genesis.engine.core import Require, System, register_system
from genesis.engine.fem_pairwise_coupling_runtime import kernel_fem_mpm_couple
from genesis.options.solvers import LegacyCouplerOptions

from .coupler_system import CouplerSystem
from .fem_solver_system import FEMSolverSystem
from .mpm_solver_system import MPMSolverSystem
from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class FEMMPMCouplingSystem(System):
    """Momentum exchange between FEM surfaces and the MPM grid."""

    manager = Require(CouplerSystem)
    fem = Require(FEMSolverSystem)
    mpm = Require(MPMSolverSystem)

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    @System.action(kind="host")
    def couple_grad(self):
        return self.run_couple_grad

    def build(self) -> None:
        self.manager.on_pairwise_fem_mpm_couple(action=self.couple)
        self.manager.on_pairwise_fem_mpm_grad(action=self.couple_grad)

    def run_couple(self, f: int) -> None:
        kernel_fem_mpm_couple(f, self.fem, self.mpm)

    def run_couple_grad(self, f: int) -> None:
        kernel_fem_mpm_couple.grad(f, self.fem, self.mpm)


@register_system(SimulatorEngine)
def create_fem_mpm_coupling_system(scene: "Scene") -> FEMMPMCouplingSystem | None:
    options = scene.options.coupler
    if (
        isinstance(options, LegacyCouplerOptions)
        and scene.sim.fem_solver.is_active
        and scene.sim.mpm_solver.is_active
        and options.fem_mpm
    ):
        return FEMMPMCouplingSystem(scene)
    return None
