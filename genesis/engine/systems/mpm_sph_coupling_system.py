"""System owning momentum exchange between the MPM grid and SPH particles."""

from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd

from genesis.engine.core import Require, System, register_system
from genesis.engine.pairwise_coupling_runtime import func_mpm_sph_couple, kernel_mpm_sph_couple
from genesis.options.solvers import LegacyCouplerOptions

from .coupler_system import CouplerSystem
from .mpm_solver_system import MPMSolverSystem
from .simulator_engine import SimulatorEngine
from .sph_solver_system import SPHSolverSystem

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class MPMSPHCouplingSystem(System):
    """Pairwise momentum exchange between the MPM grid and SPH particles."""

    manager = Require(CouplerSystem)
    mpm = Require(MPMSolverSystem)
    sph = Require(SPHSolverSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.stencil_size = int(np.floor(scene.sim.mpm_solver.dx / scene.sim.sph_solver.hash_grid_cell_size) + 2)

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    @System.action(kind="host")
    def couple_grad(self):
        return self.run_couple_grad

    @System.action(kind="stage")
    def graph_couple(self):
        return func_graph_mpm_sph_couple, self

    def build(self) -> None:
        self.manager.on_pairwise_mpm_sph_couple(action=self.couple)
        self.manager.on_pairwise_mpm_sph_grad(action=self.couple_grad)

    def run_couple(self, f: int) -> None:
        kernel_mpm_sph_couple(f, self.mpm, self.sph, self.stencil_size)

    def run_couple_grad(self, f: int) -> None:
        kernel_mpm_sph_couple.grad(f, self.mpm, self.sph, self.stencil_size)


@register_system(SimulatorEngine)
def create_mpm_sph_coupling_system(scene: "Scene") -> MPMSPHCouplingSystem | None:
    options = scene.options.coupler
    if (
        isinstance(options, LegacyCouplerOptions)
        and scene.sim.mpm_solver.is_active
        and scene.sim.sph_solver.is_active
        and options.mpm_sph
    ):
        return MPMSPHCouplingSystem(scene)
    return None


@qd.func(requires_top_level=True)
def func_graph_mpm_sph_couple(system: qd.template(), f: int, t: float):
    func_mpm_sph_couple(f, system.mpm, system.sph, system.stencil_size)
