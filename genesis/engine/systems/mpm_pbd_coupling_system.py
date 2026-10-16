"""System owning momentum exchange between the MPM grid and PBD particles."""

from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd

from genesis.engine.core import Require, System, register_system
from genesis.engine.pairwise_coupling_runtime import func_mpm_pbd_couple, kernel_mpm_pbd_couple
from genesis.options.solvers import LegacyCouplerOptions

from .coupler_system import CouplerSystem
from .mpm_solver_system import MPMSolverSystem
from .pbd_solver_system import PBDSolverSystem
from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class MPMPBDCouplingSystem(System):
    """Pairwise momentum exchange between the MPM grid and PBD particles."""

    manager = Require(CouplerSystem)
    mpm = Require(MPMSolverSystem)
    pbd = Require(PBDSolverSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.stencil_size = int(np.floor(scene.sim.mpm_solver.dx / scene.sim.pbd_solver.hash_grid_cell_size) + 2)

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    @System.action(kind="host")
    def couple_grad(self):
        return self.run_couple_grad

    @System.action(kind="stage")
    def graph_couple(self):
        return func_graph_mpm_pbd_couple, self

    def build(self) -> None:
        self.manager.on_pairwise_mpm_pbd_couple(action=self.couple)
        self.manager.on_pairwise_mpm_pbd_grad(action=self.couple_grad)

    def run_couple(self, f: int) -> None:
        kernel_mpm_pbd_couple(f, self.mpm, self.pbd, self.stencil_size)

    def run_couple_grad(self, f: int) -> None:
        kernel_mpm_pbd_couple.grad(f, self.mpm, self.pbd, self.stencil_size)


@register_system(SimulatorEngine)
def create_mpm_pbd_coupling_system(scene: "Scene") -> MPMPBDCouplingSystem | None:
    options = scene.options.coupler
    if (
        isinstance(options, LegacyCouplerOptions)
        and scene.sim.mpm_solver.is_active
        and scene.sim.pbd_solver.is_active
        and options.mpm_pbd
    ):
        return MPMPBDCouplingSystem(scene)
    return None


@qd.func(requires_top_level=True)
def func_graph_mpm_pbd_couple(system: qd.template(), f: int, t: float):
    func_mpm_pbd_couple(f, system.mpm, system.pbd, system.stencil_size)
