"""System owning MPM-Tool collision over the MPM grid."""

from typing import TYPE_CHECKING

import quadrants as qd

from genesis.engine.core import Require, System, register_system
from genesis.engine.pairwise_coupling_runtime import func_mpm_tool_couple, kernel_mpm_tool_couple
from genesis.options.solvers import LegacyCouplerOptions

from .coupler_system import CouplerSystem
from .mpm_solver_system import MPMSolverSystem
from .simulator_engine import SimulatorEngine
from .tool_solver_system import ToolSolverSystem

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class MPMToolCouplingSystem(System):
    """Pairwise MPM-Tool collision over the MPM grid."""

    manager = Require(CouplerSystem)
    mpm = Require(MPMSolverSystem)
    tool = Require(ToolSolverSystem)

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    @System.action(kind="host")
    def couple_grad(self):
        return self.run_couple_grad

    @System.action(kind="stage")
    def graph_couple(self):
        return func_graph_mpm_tool_couple, self

    def build(self) -> None:
        self.manager.on_pairwise_mpm_tool_couple(action=self.couple)
        self.manager.on_pairwise_mpm_tool_grad(action=self.couple_grad)

    def run_couple(self, f: int) -> None:
        kernel_mpm_tool_couple(f, self.mpm, self.tool)

    def run_couple_grad(self, f: int) -> None:
        kernel_mpm_tool_couple.grad(f, self.mpm, self.tool)


@register_system(SimulatorEngine)
def create_mpm_tool_coupling_system(scene: "Scene") -> MPMToolCouplingSystem | None:
    if (
        isinstance(scene.options.coupler, LegacyCouplerOptions)
        and scene.sim.mpm_solver.is_active
        and scene.sim.tool_solver.is_active
    ):
        return MPMToolCouplingSystem(scene)
    return None


@qd.func(requires_top_level=True)
def func_graph_mpm_tool_couple(system: qd.template(), f: int, t: float):
    func_mpm_tool_couple(f, system.mpm, system.tool)
