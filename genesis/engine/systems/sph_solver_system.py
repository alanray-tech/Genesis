"""System of the SPH solver, which owns its particle data, initialization and substep."""

from typing import TYPE_CHECKING

import quadrants as qd

import genesis as gs
from genesis.engine import sph_runtime
from genesis.engine.core import System, register_system
from genesis.engine.sph_data import SPHSolverData
from genesis.engine.solvers.base_solver import adopt_model
from genesis.utils.array_class import V, V_VEC
from genesis.utils.misc import qd_to_numpy

from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


SPH_MODEL = (
    "_entities",
    "_B",
    "_n_particles",
    "_particle_size",
    "_support_radius",
    "_particle_volume",
    "_pressure_solver",
    "_df_max_error_div",
    "_df_max_error_den",
    "_df_max_div_iters",
    "_df_max_den_iters",
    "_df_eps",
    "_stable_dt_coef",
    "_substep_dt",
    "_ffs",
    "_sim",
    "_options",
    "sh",
    "boundary",
)


class SPHSolverSystem(System):
    """System owning the particle data and simulation of the SPH solver of the scene."""

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.sph_solver

    @System.data
    def gravity_data(self):
        return self._gravity

    @System.data
    def density_error_data(self):
        return self.density_error

    @System.data
    def graph_condition(self):
        return self.graph_counter

    def build(self) -> None:
        from .sph_rigid_coupling_system import SPHRigidCouplingSystem

        self.rigid_coupling = self.find(SPHRigidCouplingSystem)

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    def initialize_data(self) -> None:
        """Take over the model, then allocate and initialize the particles and spatial hash."""
        adopt_model(self, SPH_MODEL)
        self.solver._system = self
        self.data = SPHSolverData(self)
        self.particles = self.data.particles
        self.particles_ng = self.data.particles_ng
        self.particles_info = self.data.particles_info
        self.particles_reordered = self.data.particles_reordered
        self.particles_ng_reordered = self.data.particles_ng_reordered
        self.particles_info_reordered = self.data.particles_info_reordered
        self.particles_render = self.data.particles_render
        self.spatial_hasher = self.data.spatial_hasher
        self.density_error = V(dtype=gs.qd_float, shape=())
        self.graph_counter = qd.ndarray(qd.i32, shape=())
        self.solver._build_gravity(gravity=V_VEC(3, dtype=gs.qd_float, shape=(self._B,)))

        for entity in self._entities:
            entity._add_to_solver()

        mu = qd_to_numpy(self.particles_info.mu, transpose=True).max()
        stable_dt = self._stable_dt_coef * (0.01 * self._support_radius**2) / mu
        if self._substep_dt > stable_dt:
            gs.logger.warning(
                f"Current timestep ({self._substep_dt:.4g}) is larger than the stable timestep "
                f"({stable_dt:.4g}) for SPH solver. "
                "Please consider reducing the timestep, liquid viscosity (mu), or increasing the particle size."
            )

        # TODO: Support per-particle density
        self._density0 = float(qd_to_numpy(self.particles_info.rho, transpose=True)[0])

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
        if self.rigid_coupling is None:
            return func_graph_sph_pre, self
        return func_graph_sph_pre_rigid, self, self.rigid_coupling

    @System.action(kind="stage")
    def graph_post_coupling(self):
        if self.rigid_coupling is None:
            return func_graph_sph_post, self
        return func_graph_sph_post_rigid, self, self.rigid_coupling

    def substep_pre_coupling(self, f: int) -> None:
        """Reorder the particles and advance their velocity through the pressure solve."""
        if self.rigid_coupling is not None:
            has_rigid_coupling = True
            rigid_coupling = self.rigid_coupling
            n_geoms = self.rigid_coupling.rigid.n_geoms
        else:
            has_rigid_coupling = False
            rigid_coupling = None
            n_geoms = 0
        if self._pressure_solver == "WCSPH":
            sph_runtime.kernel_wcsph_pre_coupling(
                self,
                f,
                self._sim.cur_t,
                has_rigid_coupling,
                rigid_coupling,
                n_geoms,
            )
        else:
            sph_runtime.kernel_dfsph_pre_coupling(
                self,
                f,
                self._sim.cur_t,
                self.graph_counter,
                has_rigid_coupling,
                rigid_coupling,
                n_geoms,
            )

    def substep_post_coupling(self, f: int) -> None:
        """Advect the particle positions after coupling and restore their authored order."""
        if self.rigid_coupling is not None:
            sph_runtime.kernel_post_coupling(
                self,
                f,
                True,
                self.rigid_coupling,
                self.rigid_coupling.rigid.n_geoms,
            )
        else:
            sph_runtime.kernel_post_coupling(self, f, False, None, 0)


@register_system(SimulatorEngine)
def create_sph_solver_system(scene: "Scene") -> SPHSolverSystem | None:
    """Create the System owning the SPH solver whenever it simulates an entity."""
    return SPHSolverSystem(scene) if scene.sim.sph_solver.is_active else None


@qd.func(requires_top_level=True)
def func_graph_sph_pre(system: qd.template(), f: int, t: float):
    sph_runtime.func_reorder_particles(system, f, False, None, 0)
    sph_runtime.func_compute_rho(system, f)
    if qd.static(system._pressure_solver == "WCSPH"):
        sph_runtime.func_compute_non_pressure_forces(system, f, t)
        sph_runtime.func_compute_pressure_forces(system, f)
        sph_runtime.func_advect_velocity(system, f)
    else:
        sph_runtime.func_compute_DFSPH_factor(system, f)
        sph_runtime.func_divergence_solve(system, system.graph_counter)
        sph_runtime.func_advect_velocity(system, f)
        sph_runtime.func_compute_non_pressure_forces(system, f, t)
        sph_runtime.func_predict_velocity(system, f)
        sph_runtime.func_density_solve(system, system.graph_counter)


@qd.func(requires_top_level=True)
def func_graph_sph_pre_rigid(system: qd.template(), coupling: qd.template(), f: int, t: float):
    sph_runtime.func_reorder_particles(system, f, True, coupling, coupling.rigid.n_geoms)
    sph_runtime.func_compute_rho(system, f)
    if qd.static(system._pressure_solver == "WCSPH"):
        sph_runtime.func_compute_non_pressure_forces(system, f, t)
        sph_runtime.func_compute_pressure_forces(system, f)
        sph_runtime.func_advect_velocity(system, f)
    else:
        sph_runtime.func_compute_DFSPH_factor(system, f)
        sph_runtime.func_divergence_solve(system, system.graph_counter)
        sph_runtime.func_advect_velocity(system, f)
        sph_runtime.func_compute_non_pressure_forces(system, f, t)
        sph_runtime.func_predict_velocity(system, f)
        sph_runtime.func_density_solve(system, system.graph_counter)


@qd.func(requires_top_level=True)
def func_graph_sph_post(system: qd.template(), f: int, t: float):
    sph_runtime.func_advect_position(system, f)
    sph_runtime.func_copy_from_reordered(system, f, False, None, 0)


@qd.func(requires_top_level=True)
def func_graph_sph_post_rigid(system: qd.template(), coupling: qd.template(), f: int, t: float):
    sph_runtime.func_advect_position(system, f)
    sph_runtime.func_copy_from_reordered(system, f, True, coupling, coupling.rigid.n_geoms)
