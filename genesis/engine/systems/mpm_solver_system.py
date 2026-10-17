"""System of the MPM solver, which owns its particle/grid data, initialization and substep."""

from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine import mpm_graph_runtime, mpm_runtime
from genesis.engine.core import Require, System, register_system
from genesis.engine.mpm_data import MPMSolverData, get_mpm_material_config, get_mpm_particle_constraints
from genesis.engine.solvers.base_solver import adopt_model
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_torch

from .rigid_sdf_system import RigidSdfSystem
from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


MPM_MODEL = (
    "_entities",
    "_B",
    "_n_particles",
    "_n_vverts",
    "_n_vfaces",
    "_n_vvert_supports",
    "_enable_CPIC",
    "_grid_offset",
    "_grid_res",
    "_dx",
    "_inv_dx",
    "_particle_volume",
    "_particle_volume_scale",
    "_ffs",
    "_sim",
    "_options",
    "boundary",
    "substep_dt",
)


class MPMSolverSystem(System):
    """System owning the particle/grid data and simulation of the MPM solver of the scene."""

    rigid_sdf_system = Require(RigidSdfSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.mpm_solver

    @System.data
    def gravity_data(self):
        return self._gravity

    def build(self) -> None:
        # Imported here to avoid the pair System's reciprocal module dependency.
        from .mpm_rigid_coupling_system import MPMRigidCouplingSystem

        self.rigid_coupling = self.find(MPMRigidCouplingSystem)
        # The graph specializes this build-time capability and compiles out the absent constraint stage.
        self.has_particle_constraints = self.solver._constraints_initialized

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    def initialize_data(self) -> None:
        """Take over the model, then allocate and initialize particles, grid and render data."""
        self.material_config = get_mpm_material_config(self.solver)
        adopt_model(self, MPM_MODEL)
        self.solver._system = self
        self.data = MPMSolverData(self)
        self.grid_res = self._grid_res
        self.grid_offset = self._grid_offset
        self.dx = self._dx
        self.particles = self.data.particles
        self.particles_ng = self.data.particles_ng
        self.particles_info = self.data.particles_info
        self.particles_render = self.data.particles_render
        self.grid = self.data.grid
        self.grid_dirty_flag = self.data.grid_dirty.flag
        self.grid_dirty_list = self.data.grid_dirty.cells
        self.grid_dirty_count = self.data.grid_dirty.count
        self.vverts_info = self.data.visual_info
        self.vverts_render = self.data.visual_state
        self.particle_constraints = None
        self._grid_total = int(np.prod(self._grid_res))
        self.solver._build_gravity(gravity=V_VEC(3, dtype=gs.qd_float, shape=(self._B,)))

        for entity in self._entities:
            entity._add_to_solver()

        suggested_dt = 2e-2 * self._dx
        if self.substep_dt > suggested_dt:
            gs.logger.warning(
                f"Current `substep_dt` ({self.substep_dt:.6g}) is greater than suggested_dt ({suggested_dt:.6g}, "
                "calculated based on `grid_density`). Simulation might be unstable."
            )

        if self._enable_CPIC and self.rigid_sdf_system.rigid_solver_system.n_geoms:
            self.rigid_sdf_system.activate()

    def initialize_constraints(self) -> None:
        """Allocate the particle constraints when the user first enables one."""
        self.particle_constraints = get_mpm_particle_constraints(self)

    @System.action(kind="host")
    def process_input(self):
        return self.solver.process_input

    @System.action(kind="host")
    def pre_coupling(self):
        return self.substep_pre_coupling

    @System.action(kind="host")
    def post_coupling(self):
        return self.substep_post_coupling

    @System.action(kind="stage")
    def graph_pre_coupling(self):
        rigid = self.rigid_sdf_system.rigid_solver_system
        if self.rigid_coupling is not None and self.solver.enable_CPIC:
            return func_graph_mpm_pre_cpic, self, rigid, self.rigid_sdf_system, self.rigid_coupling
        return func_graph_mpm_pre, self, rigid, self.rigid_sdf_system

    @System.action(kind="stage")
    def graph_post_coupling(self):
        rigid = self.rigid_sdf_system.rigid_solver_system
        if self.rigid_coupling is not None and self.solver.enable_CPIC:
            return func_graph_mpm_post_cpic, self, rigid, self.rigid_coupling
        return func_graph_mpm_post, self, rigid

    def substep_pre_coupling(self, f: int) -> None:
        """Update deformation gradients and transfer particle state to the grid."""
        rigid_system = self.rigid_sdf_system.rigid_solver_system
        coupler = self._scene.sim.coupler
        cpic_active = self.rigid_coupling is not None and self.rigid_coupling.data is not None
        mpm_rigid_coupling = self.rigid_coupling if cpic_active else None
        if self._sim.requires_grad:
            mpm_runtime.reset_grid_and_grad(self, f)
            mpm_runtime.compute_F_tmp(self, f)
            mpm_runtime.svd(self, f)
        elif self.material_config.needs_svd:
            mpm_runtime.compute_F_tmp_and_svd(self, f)
        else:
            mpm_runtime.compute_F_tmp_only(self, f)
        mpm_runtime.p2g(
            self,
            f,
            rigid_system.dyn_state.geoms,
            rigid_system.dyn_info.geoms,
            rigid_system.dyn_state.links,
            rigid_system.rigid_info,
            self.rigid_sdf_system.sdf._sdf_info,
            rigid_system.collider_data.collider_config,
            cpic_active,
            mpm_rigid_coupling,
            rigid_system.n_geoms,
        )

    def substep_post_coupling(self, f: int) -> None:
        """Transfer the coupled grid velocity back to particles and clear the touched grid cells."""
        rigid_system = self.rigid_sdf_system.rigid_solver_system
        cpic_active = self.rigid_coupling is not None and self.rigid_coupling.data is not None
        mpm_rigid_coupling = self.rigid_coupling if cpic_active else None
        mpm_runtime.g2p(
            self,
            f,
            rigid_system.dyn_info.geoms,
            rigid_system.dyn_state.links,
            rigid_system.rigid_info,
            cpic_active,
            mpm_rigid_coupling,
        )
        if self.solver._constraints_initialized:
            mpm_runtime.apply_particle_constraints(self, f, rigid_system.dyn_state.links)
        if not self._sim.requires_grad:
            mpm_runtime.reset_dirty_cells(self, f)
            if gs.use_zerocopy:
                qd_to_torch(self.grid_dirty_count, copy=False).zero_()
            else:
                self.grid_dirty_count.fill(0)
        if self._sim._cur_substep_global % gs.engine.simulator.RATE_CHECK_ERRNO == 0:
            if not mpm_runtime._is_state_valid(self, f):
                gs.raise_exception(
                    "NaN detected in MPM states. Try reducing the time step size or adjusting simulation parameters."
                )


@register_system(SimulatorEngine)
def create_mpm_solver_system(scene: "Scene") -> MPMSolverSystem | None:
    """Create the System owning the MPM solver whenever it simulates an entity."""
    return MPMSolverSystem(scene) if scene.sim.mpm_solver.is_active else None


@qd.func(requires_top_level=True)
def func_graph_mpm_pre(system: qd.template(), rigid: qd.template(), rigid_sdf: qd.template(), f: int, t: float):
    if qd.static(system.material_config.needs_svd):
        mpm_graph_runtime.graph_compute_F_tmp_and_svd(system, f)
    else:
        mpm_graph_runtime.graph_compute_F_tmp_only(system, f)
    mpm_graph_runtime.graph_p2g(
        system,
        f,
        rigid.dyn_state.geoms,
        rigid.dyn_info.geoms,
        rigid.dyn_state.links,
        rigid.rigid_info,
        rigid_sdf.sdf._sdf_info,
        rigid.collider_data.collider_config,
        False,
        None,
        0,
    )


@qd.func(requires_top_level=True)
def func_graph_mpm_pre_cpic(
    system: qd.template(),
    rigid: qd.template(),
    rigid_sdf: qd.template(),
    coupling: qd.template(),
    f: int,
    t: float,
):
    if qd.static(system.material_config.needs_svd):
        mpm_graph_runtime.graph_compute_F_tmp_and_svd(system, f)
    else:
        mpm_graph_runtime.graph_compute_F_tmp_only(system, f)
    mpm_graph_runtime.graph_p2g(
        system,
        f,
        rigid.dyn_state.geoms,
        rigid.dyn_info.geoms,
        rigid.dyn_state.links,
        rigid.rigid_info,
        rigid_sdf.sdf._sdf_info,
        rigid.collider_data.collider_config,
        True,
        coupling,
        rigid.n_geoms,
    )


@qd.func(requires_top_level=True)
def func_graph_mpm_post(system: qd.template(), rigid: qd.template(), f: int, t: float):
    mpm_graph_runtime.graph_g2p(
        system,
        f,
        rigid.dyn_info.geoms,
        rigid.dyn_state.links,
        rigid.rigid_info,
        False,
        None,
    )
    if qd.static(system.has_particle_constraints):
        mpm_graph_runtime.graph_apply_particle_constraints(system, f, rigid.dyn_state.links)
    mpm_graph_runtime.graph_reset_dirty_cells(system, f)
    mpm_graph_runtime.graph_reset_dirty_count(system)


@qd.func(requires_top_level=True)
def func_graph_mpm_post_cpic(
    system: qd.template(),
    rigid: qd.template(),
    coupling: qd.template(),
    f: int,
    t: float,
):
    mpm_graph_runtime.graph_g2p(
        system,
        f,
        rigid.dyn_info.geoms,
        rigid.dyn_state.links,
        rigid.rigid_info,
        True,
        coupling,
    )
    if qd.static(system.has_particle_constraints):
        mpm_graph_runtime.graph_apply_particle_constraints(system, f, rigid.dyn_state.links)
    mpm_graph_runtime.graph_reset_dirty_cells(system, f)
    mpm_graph_runtime.graph_reset_dirty_count(system)
