"""System owning the begin/end lifecycle of the MPM grid's coupling interval."""

from typing import TYPE_CHECKING

import quadrants as qd

from genesis.engine.core import Require, System, register_system
from genesis.engine.pairwise_coupling_runtime import (
    func_mpm_grid_begin,
    func_mpm_grid_end,
    kernel_mpm_grid_begin,
    kernel_mpm_grid_end,
)
from genesis.options.solvers import LegacyCouplerOptions

from .coupler_system import CouplerSystem
from .mpm_solver_system import MPMSolverSystem
from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class MPMGridLifecycleSystem(System):
    """Prepare and finalize MPM grid velocity around the pairwise coupling phases."""

    manager = Require(CouplerSystem)
    mpm = Require(MPMSolverSystem)

    @System.action(kind="host")
    def begin(self):
        return self.run_begin

    @System.action(kind="host")
    def end(self):
        return self.run_end

    @System.action(kind="host")
    def begin_grad(self):
        return self.run_begin_grad

    @System.action(kind="host")
    def end_grad(self):
        return self.run_end_grad

    @System.action(kind="stage")
    def graph_begin(self):
        return func_graph_mpm_grid_begin, self

    @System.action(kind="stage")
    def graph_end(self):
        return func_graph_mpm_grid_end, self

    def build(self) -> None:
        self.manager.on_pairwise_mpm_grid_begin(action=self.begin)
        self.manager.on_pairwise_mpm_grid_end(action=self.end)
        self.manager.on_pairwise_mpm_grid_begin_grad(action=self.begin_grad)
        self.manager.on_pairwise_mpm_grid_end_grad(action=self.end_grad)

    def run_begin(self, f: int) -> None:
        kernel_mpm_grid_begin(f, self._scene.sim.cur_t, self.mpm)

    def run_end(self, f: int) -> None:
        kernel_mpm_grid_end(f, self.mpm)

    def run_begin_grad(self, f: int) -> None:
        kernel_mpm_grid_begin.grad(f, self._scene.sim.cur_t, self.mpm)

    def run_end_grad(self, f: int) -> None:
        kernel_mpm_grid_end.grad(f, self.mpm)


@register_system(SimulatorEngine)
def create_mpm_grid_lifecycle_system(scene: "Scene") -> MPMGridLifecycleSystem | None:
    return (
        MPMGridLifecycleSystem(scene)
        if isinstance(scene.options.coupler, LegacyCouplerOptions) and scene.sim.mpm_solver.is_active
        else None
    )


@qd.func(requires_top_level=True)
def func_graph_mpm_grid_begin(system: qd.template(), f: int, t: float):
    func_mpm_grid_begin(f, t, system.mpm)


@qd.func(requires_top_level=True)
def func_graph_mpm_grid_end(system: qd.template(), f: int, t: float):
    func_mpm_grid_end(f, system.mpm)
