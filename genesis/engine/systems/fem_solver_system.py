"""System of the FEM solver, which owns its model, data, initialization and substep."""

from typing import TYPE_CHECKING

import igl
import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine import fem_runtime
from genesis.engine.core import HostAction, System, register_system
from genesis.engine.fem_pairwise_coupling_runtime import kernel_fem_boundary
from genesis.engine.fem_data import (
    FEMSolverData,
    FEMSurfaceMeta,
    get_fem_material_config,
    get_fem_vertex_constraints,
)
from genesis.engine.solvers.base_solver import adopt_model
from genesis.utils.array_class import V, V_VEC
from genesis.utils.misc import qd_to_numpy

from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


FEM_MODEL = (
    "_entities",
    "_B",
    "_n_elements_max",
    "_n_vertices_max",
    "_n_vverts",
    "_n_vfaces",
    "_damping",
    "_use_implicit_solver",
    "_n_newton_iterations",
    "_newton_dx_threshold",
    "_n_pcg_iterations",
    "_pcg_threshold",
    "_n_linesearch_iterations",
    "_linesearch_c",
    "_linesearch_tau",
    "_damping_alpha",
    "_damping_beta",
    "_enable_vertex_constraints",
    "_vol_scale",
    "_ffs",
    "_sim",
    "_substep_dt",
    "boundary",
    "substep_dt",
)


class FEMSolverSystem(System):
    """System owning the FEM model, typed data and explicit/implicit integration of the scene."""

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.fem_solver

    @System.action_collection(kind="host", call_args=("f",))
    def post_pairwise_constraints(self):
        """Pair-owned constraints applied after all pairwise interactions and before FEM integration."""

    @System.protocol
    def on_post_pairwise_constraint(self, *, action: HostAction) -> None:
        self.post_pairwise_constraints += action

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    def initialize_data(self) -> None:
        """Take over the FEM model, allocate typed data and let entities fill it."""
        self.material_config = get_fem_material_config(self.solver)
        self.materials = tuple(self.solver._mats)
        adopt_model(self, FEM_MODEL)
        self.solver._system = self
        self.entities = self._entities
        self.is_active = True
        self.n_vertices = self._n_vertices_max
        self.n_elements = self._n_elements_max
        self.n_surfaces = sum(entity.n_surfaces for entity in self._entities)
        self.damping = self._damping
        self.vol_scale = self._vol_scale

        self.data = FEMSolverData(self)
        self.batch_active = self.data.batch.active
        self.batch_pcg_active = self.data.batch.pcg_active
        self.batch_linesearch_active = self.data.batch.linesearch_active
        self.pcg_state = self.data.pcg
        self.linesearch_state = self.data.linesearch
        self.elements_v = self.data.vertices
        self.elements_el = self.data.elements
        self.elements_el_ng = self.data.elements_ng
        self.elements_i = self.data.elements_info
        self.elements_el_energy = self.data.element_energy
        # Preserve the (B, 3, 3, n_elements) block layout expected by FEM and SAP.
        self.elements_el_hessian = self.data.element_hessian
        self.elements_v_energy = self.data.vertex_energy
        self.elements_v_info = self.data.vertices_info
        self.pcg_state_v = self.data.pcg_vertices
        self.linesearch_state_v = self.data.linesearch_vertices
        self.surface = self.data.surface
        self.vverts_info = self.data.visual_info
        self.vverts_render = self.data.visual_state
        self.vverts_uvs = self.data.visual_info.uvs
        self.vfaces_indices = self.data.visual_info.faces
        self.tet_wrong_order = self.data.tet_wrong_order
        self.vertices_on_surface = self.data.vertices_on_surface
        self.elements_on_surface = self.data.elements_on_surface

        self.solver._build_gravity(gravity=V_VEC(3, dtype=gs.qd_float, shape=(self._B,)))
        self._gravity = self.solver._gravity
        self.solver.init_ckpt()
        for entity in self._entities:
            entity._add_to_solver()
        for material in self.materials:
            material.build(self)

        self._init_surface_info()
        if bool(qd_to_numpy(self.tet_wrong_order)):
            raise RuntimeError(
                "The order of vertices in the tetrahedral elements is not correct. "
                "Please check the input mesh or the FEM solver implementation."
            )
        if self._enable_vertex_constraints and not self.solver._constraints_initialized:
            self.initialize_constraints()

    def _init_surface_info(self) -> None:
        fem_runtime.compute_surface_vertices(self)
        fem_runtime.compute_surface_elements(self)
        vertices_on_surface_np = qd_to_numpy(self.vertices_on_surface)
        elements_on_surface_np = qd_to_numpy(self.elements_on_surface)
        (surface_vertices_np,) = vertices_on_surface_np.nonzero()
        (surface_elements_np,) = elements_on_surface_np.nonzero()

        surface_vertices = V(dtype=qd.i32, shape=(len(surface_vertices_np),))
        surface_vertices.from_numpy(surface_vertices_np.astype(np.int32, copy=False))
        surface_elements = V(dtype=qd.i32, shape=(len(surface_elements_np),))
        surface_elements.from_numpy(surface_elements_np.astype(np.int32, copy=False))

        surface_triangles_np = qd_to_numpy(self.surface.tri2v)
        pos_np = qd_to_numpy(self.elements_v.pos)[0, :, 0, :][surface_vertices_np]
        surface_vertices_mapping = np.full(self.n_vertices, -1, dtype=np.int32)
        surface_vertices_mapping[surface_vertices_np] = np.arange(len(surface_vertices_np))
        mass = igl.massmatrix(pos_np, surface_vertices_mapping[surface_triangles_np])
        surface_vert_mass_np = mass.diagonal().astype(gs.np_float, copy=False)
        surface_vert_mass = V(dtype=gs.qd_float, shape=(len(surface_vertices_np),))
        surface_vert_mass.from_numpy(surface_vert_mass_np)

        self.data.surface_meta = FEMSurfaceMeta(surface_vertices, surface_elements, surface_vert_mass)
        self.surface_vertices = surface_vertices
        self.surface_elements = surface_elements
        self.surface_vert_mass = surface_vert_mass

    def initialize_constraints(self) -> None:
        """Allocate vertex constraints when enabled at build or first requested by an entity."""
        self.data.vertex_constraints = get_fem_vertex_constraints(self)
        self.vertex_constraints = self.data.vertex_constraints
        self.solver._constraints_initialized = True

    @System.action(kind="host")
    def process_input(self):
        return self.solver.process_input

    @System.action(kind="host")
    def pre_coupling(self):
        return self.substep_pre_coupling

    @System.action(kind="host")
    def post_coupling(self):
        return self.substep_post_coupling

    def pcg_solve(self) -> None:
        fem_runtime.init_pcg_solve(self)
        for _ in range(self._n_pcg_iterations):
            fem_runtime.one_pcg_iter(self)

    def linesearch(self, f: int) -> None:
        if self._n_linesearch_iterations <= 0:
            fem_runtime.skip_linesearch(self, f)
            return
        fem_runtime.init_linesearch(self, f)
        for _ in range(self._n_linesearch_iterations):
            fem_runtime.one_linesearch_iter(self, f)

    def batch_solve(self, f: int) -> None:
        self.batch_active.fill(True)
        for _ in range(self._n_newton_iterations):
            fem_runtime.compute_ele_hessian_gradient(self, f)
            for material in self.materials:
                if material.hessian_invariant:
                    material._hessian_ready = True
            fem_runtime.accumulate_vertex_force_preconditioner(self, f)
            self.pcg_solve()
            self.linesearch(f)

    def substep_pre_coupling(self, f: int) -> None:
        from genesis.engine.couplers import IPCCoupler

        if isinstance(self._scene.sim._coupler, IPCCoupler):
            return
        if self._use_implicit_solver:
            fem_runtime.precompute_material_data(self, f)
            fem_runtime.init_pos_and_inertia(self, f)
            self.batch_solve(f)
            fem_runtime.setup_pos_vel(self, f)
        else:
            fem_runtime.init_pos_and_vel(self, f)
            fem_runtime.compute_vel(self, f)
            fem_runtime.apply_uniform_force(self, f)
            if self.solver._constraints_initialized:
                fem_runtime.apply_soft_constraints(self, f)

    def substep_post_coupling(self, f: int) -> None:
        for action in self.post_pairwise_constraints.actions:
            action.invoke((f,))
        kernel_fem_boundary(f, self)
        fem_runtime.compute_pos(self, f)
        if self.solver._constraints_initialized and not self._use_implicit_solver:
            fem_runtime.apply_hard_constraints(self, f)

    @property
    def n_vertices_max(self) -> int:
        return self._n_vertices_max

    @property
    def n_elements_max(self) -> int:
        return self._n_elements_max

    @property
    def n_surface_vertices(self) -> int:
        return self.surface_vertices.shape[0]

    @property
    def n_surface_elements(self) -> int:
        return self.surface_elements.shape[0]

    @property
    def floor_height(self) -> float:
        return self.boundary.height


@register_system(SimulatorEngine)
def create_fem_solver_system(scene: "Scene") -> FEMSolverSystem | None:
    """Create the System owning the FEM solver whenever it simulates an entity."""
    return FEMSolverSystem(scene) if scene.sim.fem_solver.is_active else None
