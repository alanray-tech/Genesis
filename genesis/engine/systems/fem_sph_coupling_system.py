"""System owning the FEM-SPH coupling pair."""

from typing import TYPE_CHECKING

from genesis.engine.core import Require, System, register_system
from genesis.engine.fem_pairwise_coupling_runtime import kernel_fem_sph_couple
from genesis.options.solvers import LegacyCouplerOptions

from .coupler_system import CouplerSystem
from .fem_solver_system import FEMSolverSystem
from .simulator_engine import SimulatorEngine
from .sph_solver_system import SPHSolverSystem

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class FEMSPHCouplingSystem(System):
    """Momentum exchange between FEM surfaces and SPH particles."""

    manager = Require(CouplerSystem)
    fem = Require(FEMSolverSystem)
    sph = Require(SPHSolverSystem)

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    @System.action(kind="host")
    def couple_grad(self):
        return self.run_couple_grad

    def build(self) -> None:
        self.manager.on_pairwise_fem_sph_couple(action=self.couple)
        self.manager.on_pairwise_fem_sph_grad(action=self.couple_grad)

    def run_couple(self, f: int) -> None:
        kernel_fem_sph_couple(f, self.fem, self.sph)

    def run_couple_grad(self, f: int) -> None:
        kernel_fem_sph_couple.grad(f, self.fem, self.sph)


@register_system(SimulatorEngine)
def create_fem_sph_coupling_system(scene: "Scene") -> FEMSPHCouplingSystem | None:
    options = scene.options.coupler
    if (
        isinstance(options, LegacyCouplerOptions)
        and scene.sim.fem_solver.is_active
        and scene.sim.sph_solver.is_active
        and options.fem_sph
    ):
        return FEMSPHCouplingSystem(scene)
    return None
