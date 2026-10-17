from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine import mpm_runtime
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import MPMEntity
from genesis.engine.materials import MPM
from genesis.engine.states.solvers import MPMSolverState
from genesis.options.solvers import MPMOptions
from genesis.utils.misc import DeprecationError, qd_to_torch, qd_zero_grad

from .base_solver import GravityMixin, Solver, TimeBasedMixin, forward_to_system

if TYPE_CHECKING:
    from genesis.engine.entities import MPMEntity
    from genesis.engine.scene import Scene
    from genesis.engine.solvers.base_solver import Solver
    from genesis.engine.simulator import Simulator


class MPMSolver(GravityMixin, TimeBasedMixin, Solver):
    material_cls = MPM.Base
    data = forward_to_system("data")
    particles = forward_to_system("particles")
    particles_ng = forward_to_system("particles_ng")
    particles_info = forward_to_system("particles_info")
    particles_render = forward_to_system("particles_render")
    grid = forward_to_system("grid")
    grid_dirty_flag = forward_to_system("grid_dirty_flag")
    grid_dirty_list = forward_to_system("grid_dirty_list")
    grid_dirty_count = forward_to_system("grid_dirty_count")
    vverts_info = forward_to_system("vverts_info")
    vverts_render = forward_to_system("vverts_render")
    particle_constraints = forward_to_system("particle_constraints")
    _gravity = forward_to_system("_gravity", None, settable=True)

    def __init__(self, scene: "Scene", sim: "Simulator", options: "MPMOptions"):
        super().__init__(scene, sim, options)

        # options
        self._grid_density = options.grid_density
        self._particle_size = options.particle_size
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)
        self._enable_CPIC = options.enable_CPIC
        self._constraints_initialized = False

        self._n_vvert_supports = self.scene.options.vis.n_support_neighbors

        # `_particle_volume_scale` is used to avoid potential numerical instability, as the actual `_particle_volume` may be very small.
        # Note that the magnitude of `_particle_volume` doesn't affect MPM simulation itself, but it is used to compute particle
        # mass. We need to account for this scale when handling coupling.
        self._particle_volume_real = float(self._particle_size**3)
        self._particle_volume_scale = 1e3
        self._particle_volume = self._particle_volume_real * self._particle_volume_scale

        # Other derived parameters
        self._dx = float(1.0 / self._grid_density)
        self._inv_dx = float(self._grid_density)
        self._lower_bound_cell = np.round(self._grid_density * self._lower_bound).astype(gs.np_int)
        self._upper_bound_cell = np.round(self._grid_density * self._upper_bound).astype(gs.np_int)
        self._grid_res = self._upper_bound_cell - self._lower_bound_cell + 1  # +1 to include both corner
        self._grid_offset = qd.Vector(self._lower_bound_cell)
        if np.prod(self._grid_res) > 1e9:
            gs.raise_exception(
                "Grid size larger than 1e9 not supported by MPM solver. Please reduce 'grid_density', or set tighter "
                "boundaries via 'lower_bound' / 'upper_bound'."
            )

        # materials
        self._materials = list()
        self._materials_idx = list()
        self._materials_update_F_S_Jp = list()
        self._materials_update_stress = list()

        # boundary
        self.setup_boundary()

    def setup_boundary(self):
        # safety padding
        self.boundary_padding = 3 * self._dx
        self.boundary = CubeBoundary(
            lower=self._lower_bound + self.boundary_padding, upper=self._upper_bound - self.boundary_padding
        )

    def init_ckpt(self):
        self._ckpt = dict()

    def init_constraints(self):
        """Lazy initialization of particle constraint fields."""
        # Memory check: ensure index fits in int32
        if self._n_particles * self._B * 3 > np.iinfo(np.int32).max:
            gs.raise_exception(
                f"Particle constraint shape (n_envs={self._B}, n_particles={self._n_particles}, 3) is too large. "
                "Consider reducing n_envs or n_particles."
            )

        self._constraints_initialized = True

        self._system.initialize_constraints()

    def reset_grad(self):
        qd_zero_grad(self.particles)
        qd_zero_grad(self.grid)

        for entity in self._entities:
            entity.reset_grad()

    def build(self):
        super().build()

        # particles and entities
        self._B = self._sim._B
        self._n_particles = self.n_particles
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

        self._coupler = self.sim._coupler

        # Aggregate SVD requirement across registered materials. If no material reads U/V/S, the solver can skip the SVD
        # kernel entirely on the forward pass and derive J from det(F_tmp) directly. Compile-time constant consumed by
        # p2g via qd.static. Forced True under requires_grad so the backward path stays exactly as upstream.
        # TODO: validate the J = det(F_tmp) shortcut composes with quadrants autograd before enabling it there.
        self.needs_svd = self._sim.requires_grad or any(m.needs_svd for m in self._materials)

        if self.is_active:
            if self._enable_CPIC and self._sim.requires_grad:
                gs.raise_exception(
                    "CPIC is not supported in differentiable mode yet. Submit a feature request if you need it."
                )

            self.init_ckpt()

    @property
    def is_active(self):
        return self.n_particles > 0

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "MPMEntity":
        self.add_material(material)

        # create entity
        entity = MPMEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            name=name,
        )
        self._entities.append(entity)

        return entity

    def _kernel_get_mass(self, particle_start, n_particles, mass, envs_idx):
        mpm_runtime._kernel_get_mass(self._system, particle_start, n_particles, mass, envs_idx)

    def add_material(self, material):
        # Register material update methods if and only if the provided material is not already registered
        for material_i in self._materials:
            if material == material_i:
                material.idx = material_i.idx
                break
        else:
            material.idx = len(self._materials_idx)
            self._materials_idx.append(material.idx)
            self._materials_update_F_S_Jp.append(material.update_F_S_Jp)
            self._materials_update_stress.append(material.update_stress)
        self._materials.append(material)

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self._entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        self._system.substep_pre_coupling(f)

    def substep_pre_coupling_grad(self, f):
        cpic_active = self._system.rigid_coupling is not None and self._system.rigid_coupling.data is not None
        mpm_rigid_coupling = self._system.rigid_coupling if cpic_active else None
        mpm_runtime.p2g.grad(
            self._system,
            f,
            self.sim.rigid_solver.dyn_state.geoms,
            self.sim.rigid_solver.dyn_info.geoms,
            self.sim.rigid_solver.dyn_state.links,
            self.sim.rigid_solver.rigid_info,
            self.sim.rigid_solver.sdf._sdf_info,
            self.sim.rigid_solver.collider.collider_config,
            cpic_active,
            mpm_rigid_coupling,
            self.sim.rigid_solver.n_geoms,
        )
        mpm_runtime.svd_grad(self._system, f)
        mpm_runtime.compute_F_tmp.grad(self._system, f)

    def substep_post_coupling(self, f):
        self._system.substep_post_coupling(f)

    def substep_post_coupling_grad(self, f):
        cpic_active = self._system.rigid_coupling is not None and self._system.rigid_coupling.data is not None
        mpm_rigid_coupling = self._system.rigid_coupling if cpic_active else None
        mpm_runtime.g2p.grad(
            self._system,
            f,
            self.sim.rigid_solver.dyn_info.geoms,
            self.sim.rigid_solver.dyn_state.links,
            self.sim.rigid_solver.rigid_info,
            cpic_active,
            mpm_rigid_coupling,
        )

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        for entity in self._entities:
            entity.collect_output_grads()

    def add_grad_from_state(self, state):
        if self.is_active:
            if state.pos.grad is not None:
                state.pos.assert_contiguous()
                mpm_runtime.add_grad_from_pos(self._system, self._sim.cur_substep_local, state.pos.grad)

            if state.vel.grad is not None:
                state.vel.assert_contiguous()
                mpm_runtime.add_grad_from_vel(self._system, self._sim.cur_substep_local, state.vel.grad)

            if state.C.grad is not None:
                state.C.assert_contiguous()
                mpm_runtime.add_grad_from_C(self._system, self._sim.cur_substep_local, state.C.grad)

            if state.F.grad is not None:
                state.F.assert_contiguous()
                mpm_runtime.add_grad_from_F(self._system, self._sim.cur_substep_local, state.F.grad)

            if state.Jp.grad is not None:
                state.Jp.assert_contiguous()
                mpm_runtime.add_grad_from_Jp(self._system, self._sim.cur_substep_local, state.Jp.grad)

    def save_ckpt(self, ckpt_name):
        if self._sim.requires_grad:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = {
                    "pos": torch.zeros((self._B, self._n_particles, 3), dtype=gs.tc_float, device=gs.device),
                    "vel": torch.zeros((self._B, self._n_particles, 3), dtype=gs.tc_float, device=gs.device),
                    "C": torch.zeros((self._B, self._n_particles, 3, 3), dtype=gs.tc_float, device=gs.device),
                    "F": torch.zeros((self._B, self._n_particles, 3, 3), dtype=gs.tc_float, device=gs.device),
                    "Jp": torch.zeros((self._B, self._n_particles), dtype=gs.tc_float, device=gs.device),
                    "active": torch.zeros((self._B, self._n_particles), dtype=gs.tc_bool, device=gs.device),
                }

            mpm_runtime._kernel_get_state(
                self._system,
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["C"],
                self._ckpt[ckpt_name]["F"],
                self._ckpt[ckpt_name]["Jp"],
                self._ckpt[ckpt_name]["active"],
            )

            for entity in self._entities:
                entity.save_ckpt(ckpt_name)

        # restart from frame 0 in memory
        mpm_runtime.copy_frame(self._system, self._sim.substeps_local, 0)

    def load_ckpt(self, ckpt_name):
        mpm_runtime.copy_frame(self._system, 0, self._sim.substeps_local)
        if gs.use_ndarray:
            mpm_runtime.copy_grad_ndarray(
                0,
                self._sim.substeps_local,
                self._n_particles,
                self._B,
                self.particles.pos.grad,
                self.particles.vel.grad,
                self.particles.F.grad,
                self.particles.C.grad,
                self.particles.Jp.grad,
                self.particles_ng.active,
            )
        else:
            mpm_runtime.copy_grad(self._system, 0, self._sim.substeps_local)

        if self._sim.requires_grad:
            mpm_runtime.reset_grad_till_frame(self._system, self._sim.substeps_local)

            mpm_runtime._kernel_set_state(
                self._system,
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["C"],
                self._ckpt[ckpt_name]["F"],
                self._ckpt[ckpt_name]["Jp"],
                self._ckpt[ckpt_name]["active"],
            )

            for entity in self._entities:
                entity.load_ckpt(ckpt_name=ckpt_name)

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            mpm_runtime._kernel_set_state(
                self._system, f, state.pos, state.vel, state.C, state.F, state.Jp, state.active
            )

    def get_state(self, f):
        if not self.is_active:
            return None

        state = MPMSolverState(self._scene)
        mpm_runtime._kernel_get_state(self._system, f, state.pos, state.vel, state.C, state.F, state.Jp, state.active)
        return state

    def update_render_fields(self):
        mpm_runtime._kernel_update_render_fields(self._system, self.sim.cur_substep_local)

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        return sum(entity.n_particles for entity in self._entities)

    @property
    def n_vverts(self):
        if self.is_built:
            return self._n_vverts
        return sum(entity.n_vverts for entity in self._entities)

    @property
    def n_vfaces(self):
        if self.is_built:
            return self._n_vfaces
        return sum(entity.n_vfaces for entity in self._entities)

    @property
    def grid_density(self):
        return self._grid_density

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def leaf_block_size(self):
        raise DeprecationError("This property has been removed.")

    @property
    def use_sparse_grid(self):
        return DeprecationError("This property has been removed.")

    @property
    def dx(self):
        return self._dx

    @property
    def inv_dx(self):
        return self._inv_dx

    @property
    def particle_volume_real(self):
        return self._particle_volume_real

    @property
    def particle_volume(self):
        return self._particle_volume

    @property
    def particle_volume_scale(self):
        return self._particle_volume_scale

    @property
    def is_built(self):
        return self._scene._is_built

    @property
    def lower_bound_cell(self):
        return self._lower_bound_cell

    @property
    def upper_bound_cell(self):
        return self._upper_bound_cell

    @property
    def grid_res(self):
        return self._grid_res

    @property
    def grid_offset(self):
        return self._grid_offset

    @property
    def enable_CPIC(self):
        return self._enable_CPIC

    @property
    def enable_particle_constraints(self):
        return self._enable_particle_constraints
