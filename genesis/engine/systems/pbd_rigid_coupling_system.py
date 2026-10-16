"""System owning PBD-Rigid collision and link attachment."""

from typing import TYPE_CHECKING

import quadrants as qd

from genesis.engine.coupling_data import get_pbd_rigid_coupling_data
from genesis.engine.core import Require, System, register_system
from genesis.engine.pairwise_coupling_runtime import (
    graph_pbd_rigid_couple,
    graph_solve_pbd_rigid_attachment,
    kernel_attach_pbd_to_rigid_link,
    kernel_clear_pbd_rigid_attachment,
    kernel_pbd_rigid_couple,
    kernel_solve_pbd_rigid_attachment,
)
from genesis.options.solvers import LegacyCouplerOptions

from .coupler_system import CouplerSystem
from .pbd_solver_system import PBDSolverSystem
from .rigid_sdf_system import RigidSdfSystem
from .rigid_solver_system import RigidSolverSystem
from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene

CLAMPED_INV_DT = 50.0


class PBDRigidCouplingSystem(System):
    """Pairwise collision and link animation between PBD particles and rigid geoms."""

    manager = Require(CouplerSystem)
    pbd = Require(PBDSolverSystem)
    rigid = Require(RigidSolverSystem)
    rigid_sdf = Require(RigidSdfSystem)

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    @System.action(kind="stage")
    def graph_couple(self):
        return func_graph_pbd_rigid_couple, self

    def initialize_data(self) -> None:
        self.rigid_sdf.activate()
        self.data = get_pbd_rigid_coupling_data(self.pbd, self.rigid)
        self.rigid_normal_reordered = self.data.rigid_normal_reordered
        self.attachment = self.data.attachment
        self.clamped_inv_dt = min(1.0 / self.pbd._substep_dt, CLAMPED_INV_DT)

    def build(self) -> None:
        self.manager.on_pairwise_couple(action=self.couple)

    def run_couple(self, f: int) -> None:
        kernel_pbd_rigid_couple(
            self.pbd,
            self,
            self.rigid.n_geoms,
            self.rigid._substep_dt,
            self.rigid.dyn_state.geoms,
            self.rigid.dyn_info.geoms,
            self.rigid.dyn_state.links,
            self.rigid_sdf.sdf._sdf_info,
            self.rigid.rigid_info,
            self.rigid.collider_data.collider_config,
        )
        kernel_solve_pbd_rigid_attachment(self.pbd, self, self.clamped_inv_dt, self.rigid.dyn_state.links)

    def attach(self, particles_idx, envs_idx, link_idx: int, links_state=None) -> None:
        kernel_attach_pbd_to_rigid_link(
            self.pbd,
            self,
            particles_idx,
            envs_idx,
            link_idx,
            self.rigid.dyn_state.links if links_state is None else links_state,
        )

    def detach(self, particles_idx, envs_idx) -> None:
        kernel_clear_pbd_rigid_attachment(self.pbd, self, particles_idx, envs_idx)


@register_system(SimulatorEngine)
def create_pbd_rigid_coupling_system(scene: "Scene") -> PBDRigidCouplingSystem | None:
    options = scene.options.coupler
    if (
        isinstance(options, LegacyCouplerOptions)
        and scene.sim.pbd_solver.is_active
        and scene.sim.rigid_solver.is_active
        and options.rigid_pbd
    ):
        return PBDRigidCouplingSystem(scene)
    return None


@qd.func(requires_top_level=True)
def func_graph_pbd_rigid_couple(system: qd.template(), f: int, t: float):
    graph_pbd_rigid_couple(
        system.pbd,
        system,
        system.rigid.n_geoms,
        system.rigid._substep_dt,
        system.rigid.dyn_state.geoms,
        system.rigid.dyn_info.geoms,
        system.rigid.dyn_state.links,
        system.rigid_sdf.sdf._sdf_info,
        system.rigid.rigid_info,
        system.rigid.collider_data.collider_config,
    )
    graph_solve_pbd_rigid_attachment(
        system.pbd,
        system,
        system.clamped_inv_dt,
        system.rigid.dyn_state.links,
    )
