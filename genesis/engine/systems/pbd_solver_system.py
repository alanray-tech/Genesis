"""System of the PBD solver, which owns its particle and constraint data, initialization and substep."""

from typing import TYPE_CHECKING

import quadrants as qd

import genesis as gs
from genesis.engine import pbd_runtime
from genesis.engine.core import System, register_system
from genesis.engine.pbd_data import PBDSolverData
from genesis.engine.solvers.base_solver import adopt_model
from genesis.utils.array_class import V_VEC

from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


PBD_MODEL = (
    "_entities",
    "_B",
    "_n_particles",
    "_n_fluid_particles",
    "_n_edges",
    "_n_inner_edges",
    "_n_elems",
    "_n_vverts",
    "_n_vfaces",
    "_n_vvert_supports",
    "_particle_size",
    "_max_stretch_solver_iterations",
    "_max_bending_solver_iterations",
    "_max_volume_solver_iterations",
    "_max_density_solver_iterations",
    "_max_viscosity_solver_iterations",
    "_substep_dt",
    "_ffs",
    "_sim",
    "dist_scale",
    "h",
    "h_2",
    "h_6",
    "h_9",
    "poly6_Coe",
    "spiky_Coe",
    "lambda_epsilon",
    "S_Corr_delta_q",
    "S_Corr_k",
    "g_del",
    "vorqd_epsilon",
    "sh",
    "boundary",
    "MATERIAL",
)


@qd.data_oriented
class PBDSolverSystem(System):
    """System owning the particle and constraint data and simulation of the PBD solver of the scene."""

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.pbd_solver

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    def initialize_data(self) -> None:
        """Take over the model, then allocate and initialize the particles, constraints and spatial hash."""
        adopt_model(self, PBD_MODEL)
        self.solver._system = self
        self.data = PBDSolverData(self)
        self.particles_info = self.data.particles_info
        self.particles_info_reordered = self.data.particles_info_reordered
        self.particles = self.data.particles
        self.particles_reordered = self.data.particles_reordered
        self.particles_ng = self.data.particles_ng
        self.particles_ng_reordered = self.data.particles_ng_reordered
        self.particles_render = self.data.particles_render
        self.edges_info = self.data.edges_info
        self.inner_edges_info = self.data.inner_edges_info
        self.elems_info = self.data.elems_info
        self.vverts_info = self.data.visual_info
        self.vverts_render = self.data.visual_state
        self.vverts_uvs = self.data.visual_info.uvs
        self.vfaces_indices = self.data.visual_info.faces
        self.spatial_hasher = self.data.spatial_hasher
        self.solver._build_gravity(gravity=V_VEC(3, dtype=gs.qd_float, shape=(self._B,)))
        for entity in self._entities:
            entity._add_to_solver()

    @System.action(kind="host")
    def process_input(self):
        return self.solver.process_input

    @System.action(kind="host")
    def pre_coupling(self):
        return self.substep_pre_coupling

    @System.action(kind="host")
    def post_coupling(self):
        return self.substep_post_coupling

    def substep_pre_coupling(self, f: int) -> None:
        """Apply external forces, solve the topology and spatial constraints, and compute effective velocity."""
        pbd_runtime.kernel_pre_coupling(self, f, self._sim.cur_t)

    def substep_post_coupling(self, f: int) -> None:
        """Restore authored particle order and apply the boundary collision."""
        pbd_runtime.kernel_post_coupling(self, f)


@register_system(SimulatorEngine)
def create_pbd_solver_system(scene: "Scene") -> PBDSolverSystem | None:
    """Create the System owning the PBD solver whenever it simulates an entity."""
    return PBDSolverSystem(scene) if scene.sim.pbd_solver.is_active else None
