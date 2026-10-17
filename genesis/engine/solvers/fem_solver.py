# pylint: disable=no-value-for-parameter

from typing import TYPE_CHECKING

import numpy as np
import igl
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine import fem_runtime
from genesis.engine.boundaries import FloorBoundary
from genesis.engine.entities.fem_entity import FEMEntity
from genesis.engine.materials import FEM
from genesis.engine.states.solvers import FEMSolverState
from genesis.utils.geom import qd_transform_by_quat, qd_transform_quat_by_quat
from genesis.utils.misc import qd_to_torch, qd_zero_grad

from .base_solver import GravityMixin, Solver, TimeBasedMixin, forward_to_system

if TYPE_CHECKING:
    from genesis.engine.entities import FEMEntity


@qd.data_oriented
class FEMSolver(GravityMixin, TimeBasedMixin, Solver):
    material_cls = FEM.Base
    data = forward_to_system("data")
    batch_active = forward_to_system("batch_active")
    batch_pcg_active = forward_to_system("batch_pcg_active")
    batch_linesearch_active = forward_to_system("batch_linesearch_active")
    pcg_state = forward_to_system("pcg_state")
    linesearch_state = forward_to_system("linesearch_state")
    elements_v = forward_to_system("elements_v")
    elements_el = forward_to_system("elements_el")
    elements_el_ng = forward_to_system("elements_el_ng")
    elements_i = forward_to_system("elements_i")
    elements_el_energy = forward_to_system("elements_el_energy")
    elements_el_hessian = forward_to_system("elements_el_hessian")
    elements_v_energy = forward_to_system("elements_v_energy")
    elements_v_info = forward_to_system("elements_v_info")
    pcg_state_v = forward_to_system("pcg_state_v")
    linesearch_state_v = forward_to_system("linesearch_state_v")
    surface = forward_to_system("surface")
    vverts_info = forward_to_system("vverts_info")
    vverts_render = forward_to_system("vverts_render")
    vverts_uvs = forward_to_system("vverts_uvs")
    vfaces_indices = forward_to_system("vfaces_indices")
    tet_wrong_order = forward_to_system("tet_wrong_order")
    vertices_on_surface = forward_to_system("vertices_on_surface")
    elements_on_surface = forward_to_system("elements_on_surface")
    surface_vertices = forward_to_system("surface_vertices")
    surface_elements = forward_to_system("surface_elements")
    surface_vert_mass = forward_to_system("surface_vert_mass")
    vertex_constraints = forward_to_system("vertex_constraints")
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._floor_height = options.floor_height
        self._damping = options.damping
        self._use_implicit_solver = options.use_implicit_solver
        self._n_newton_iterations = options.n_newton_iterations
        self._newton_dx_threshold = options.newton_dx_threshold
        self._n_pcg_iterations = options.n_pcg_iterations
        self._pcg_threshold = options.pcg_threshold
        self._n_linesearch_iterations = options.n_linesearch_iterations
        self._linesearch_c = options.linesearch_c
        self._linesearch_tau = options.linesearch_tau
        self._damping_alpha = options.damping_alpha
        self._damping_beta = options.damping_beta
        self._enable_vertex_constraints = options.enable_vertex_constraints

        # use scaled volume for better numerical stability, similar to p_vol_scale in mpm
        self._vol_scale = float(1e4)

        # materials
        self._mats = list()
        self._mats_idx = list()
        self._mats_update_stress = list()
        self._mats_compute_energy_gradient_hessian = list()
        self._mats_compute_energy = list()

        # boundary
        self.setup_boundary()

        # lazy initialization
        self._constraints_initialized = False

    def setup_boundary(self):
        self.boundary = FloorBoundary(height=self._floor_height)

    def init_ckpt(self):
        self._ckpt = dict()

    def init_constraints(self):
        self._system.initialize_constraints()

    def reset_grad(self):
        qd_zero_grad(self.elements_v)
        qd_zero_grad(self.elements_el)

        for entity in self._entities:
            entity.reset_grad()

    def build(self):
        super().build()

        self.n_envs = self.sim.n_envs
        self._B = self.sim._B

        # elements and bodies
        self._n_elements_max = self.n_elements
        self._n_vertices_max = self.n_vertices
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

        # Device data and material scratch are initialized by FEMSolverSystem after the Engine has resolved Systems.

    @property
    def is_active(self):
        # Counted from the entities rather than from what build recorded, so the answer stands before it: the rate
        # every solver integrates at is settled from the active ones, and that is settled before any of them builds.
        return self.n_elements > 0

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "FEMEntity":
        # add material's update methods if not matching any existing material
        exist = False
        for mat in self._mats:
            if material == mat:
                material.idx = mat.idx
                exist = True
                break
        self._mats.append(material)
        if not exist:
            material.idx = len(self._mats_idx)
            self._mats_idx.append(material.idx)
            self._mats_update_stress.append(material.update_stress)
            self._mats_compute_energy_gradient_hessian.append(material.compute_energy_gradient_hessian)
            self._mats_compute_energy.append(material.compute_energy)

        # create entity
        entity = FEMEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            v_start=self.n_vertices,
            el_start=self.n_elements,
            s_start=self.n_surfaces,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            name=name,
        )

        self._entities.append(entity)
        return entity

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    # Other options for preconditioner:
    # Uncomment one of the following lines to test different preconditioners
    # Use identity for preconditioner
    # self.pcg_state_v[i_b, i_v].prec = qd.Matrix.identity(gs.qd_float, 3)

    # Use diagonal for preconditioner
    # self.pcg_state_v[i_b, i_v].prec = qd.Matrix([[1 / self.pcg_state_v[i_b, i_v].diag3x3[0, 0], 0, 0],
    #                                            [0, 1 / self.pcg_state_v[i_b, i_v].diag3x3[1, 1], 0],
    #                                            [0, 0, 1 / self.pcg_state_v[i_b, i_v].diag3x3[2, 2]]])

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self._entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        self._system.substep_pre_coupling(f)

    def substep_pre_coupling_grad(self, f):
        if self.is_active:
            if self._use_implicit_solver:
                gs.raise_exception("Gradient computation is not supported for implicit solver.")
            fem_runtime.apply_uniform_force.grad(self._system, f)
            fem_runtime.compute_vel.grad(self._system, f)
            fem_runtime.init_pos_and_vel.grad(self._system, f)

    def substep_post_coupling(self, f):
        self._system.substep_post_coupling(f)

    def substep_post_coupling_grad(self, f):
        if self.is_active:
            fem_runtime.compute_pos.grad(self._system, f)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- gradient ---------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        for entity in self._entities:
            entity.collect_output_grads()

    def add_grad_from_state(self, state):
        if self.is_active:
            if state.pos.grad is not None:
                state.pos.assert_contiguous()
                fem_runtime._kernel_add_grad_from_pos(self._system, self._sim.cur_substep_local, state.pos.grad)

            if state.vel.grad is not None:
                state.vel.assert_contiguous()
                fem_runtime._kernel_add_grad_from_vel(self._system, self._sim.cur_substep_local, state.vel.grad)

    def save_ckpt(self, ckpt_name):
        if self.is_active:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = {
                    "pos": torch.zeros((self._B, self.n_vertices, 3), dtype=gs.tc_float, device=gs.device),
                    "vel": torch.zeros((self._B, self.n_vertices, 3), dtype=gs.tc_float, device=gs.device),
                    "active": torch.zeros((self._B, self.n_elements), dtype=gs.tc_int, device=gs.device),
                }

            fem_runtime._kernel_get_state(
                self._system,
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["active"],
            )

            fem_runtime.copy_frame(self._system, self.sim.substeps_local, 0)

    def load_ckpt(self, ckpt_name):
        fem_runtime.copy_frame(self._system, 0, self._sim.substeps_local)
        if gs.use_ndarray:
            fem_runtime.copy_grad_ndarray(
                0,
                self._sim.substeps_local,
                self.n_vertices_max,
                self.n_elements_max,
                self._B,
                self.elements_v.pos.grad,
                self.elements_v.vel.grad,
                self.elements_el_ng.active,
            )
        else:
            fem_runtime.copy_grad(self._system, 0, self._sim.substeps_local)

        if self._sim.requires_grad:
            fem_runtime.reset_grad_till_frame(self._system, self._sim.substeps_local)

            fem_runtime._kernel_set_state(
                self._system,
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["active"],
            )

            for entity in self._entities:
                entity.load_ckpt(ckpt_name=ckpt_name)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            fem_runtime._kernel_set_state(self._system, f, state.pos, state.vel, state.active)

    def get_state(self, f):
        if self.is_active:
            state = FEMSolverState(self._scene)
            fem_runtime._kernel_get_state(self._system, f, state.pos, state.vel, state.active)
        else:
            state = None
        return state

    def get_state_render(self, f):
        """
        Refresh and return the render geometry of every visual geom, laid out contiguously.

        Returns
        -------
        tuple
            (vverts_pos, vverts_uvs, vfaces_indices) - environment-offset render vertex positions with shape
            (n_vverts, B), their UV coordinates, and the render triangles in global render vertex space.
        """
        if not self.is_active or self._n_vverts == 0:
            return None, None, None

        fem_runtime._kernel_get_state_render(self._system, f)
        return self.vverts_render.pos, self.vverts_uvs, self.vfaces_indices

    def get_forces(self):
        """
        Get forces on all vertices.

        Returns:
            torch.Tensor : shape (B, n_vertices, 3) where B is batch size
        """
        if not self.is_active:
            return None

        return qd_to_torch(self.elements_v_energy.force, copy=True)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def floor_height(self):
        return self._floor_height

    @property
    def damping(self):
        return self._damping

    @property
    def n_vertices(self):
        return sum([entity.n_vertices for entity in self._entities])

    @property
    def n_elements(self):
        return sum([entity.n_elements for entity in self._entities])

    @property
    def n_surfaces(self):
        return sum([entity.n_surfaces for entity in self.entities])

    @property
    def n_vverts(self):
        return sum([entity.n_vverts for entity in self._entities])

    @property
    def n_vfaces(self):
        return sum([entity.n_vfaces for entity in self._entities])

    @property
    def n_vertices_max(self):
        return self._n_vertices_max

    @property
    def n_elements_max(self):
        return self._n_elements_max

    @property
    def vol_scale(self):
        return self._vol_scale

    @property
    def n_surface_vertices(self):
        return self.surface_vertices.shape[0]

    @property
    def n_surface_elements(self):
        return self.surface_elements.shape[0]

    # ------------------------------------------------------------------------------------
    # -------------------------------- vertex constraints --------------------------------
    # ------------------------------------------------------------------------------------
