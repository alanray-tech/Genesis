from typing import TYPE_CHECKING

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import SPHEntity
from genesis.engine.materials import SPH
from genesis.engine import sph_runtime
from genesis.engine.states.solvers import SPHSolverState

from .base_solver import GravityMixin, Solver, TimeBasedMixin, forward_to_system

if TYPE_CHECKING:
    from genesis.engine.entities import SPHEntity


class SPHSolver(GravityMixin, TimeBasedMixin, Solver):
    material_cls = SPH.Base
    data = forward_to_system("data")
    particles = forward_to_system("particles")
    particles_ng = forward_to_system("particles_ng")
    particles_info = forward_to_system("particles_info")
    particles_reordered = forward_to_system("particles_reordered")
    particles_ng_reordered = forward_to_system("particles_ng_reordered")
    particles_info_reordered = forward_to_system("particles_info_reordered")
    particles_render = forward_to_system("particles_render")
    spatial_hasher = forward_to_system("spatial_hasher")
    _gravity = forward_to_system("_gravity", None, settable=True)
    _density0 = forward_to_system("_density0", 0.0, settable=True)

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._particle_size = options.particle_size
        self._support_radius = options._support_radius
        self._pressure_solver = options.pressure_solver

        # DFSPH parameters
        self._df_max_error_div = options.max_divergence_error
        self._df_max_error_den = options.max_density_error_percent
        self._df_max_div_iters = options.max_divergence_solver_iterations
        self._df_max_den_iters = options.max_density_solver_iterations
        self._df_eps = 1e-5

        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)

        self._particle_volume = 0.8 * self._particle_size**3  # 0.8 is an empirical value

        # spatial hasher
        self.sh = gu.SpatialHasher(
            cell_size=options.hash_grid_cell_size,
            grid_res=options._hash_grid_res,
        )
        # boundary
        self.setup_boundary()

        # Coefficient for computing stable timestep
        self._stable_dt_coef = 1.0

    def setup_boundary(self):
        self.boundary = CubeBoundary(
            lower=self._lower_bound,
            upper=self._upper_bound,
            # restitution=0.5,
        )

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        pass

    def build(self):
        super().build()

        self._B = self._sim._B

        # particles and entities
        self._n_particles = self.n_particles

        self._coupler = self.sim._coupler

        if self.is_active:
            self.init_ckpt()

    @property
    def is_active(self):
        return self.n_particles > 0

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "SPHEntity":
        entity = SPHEntity(
            scene=self.scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
            name=name,
        )

        self.entities.append(entity)
        return entity

    def _kernel_add_particles(
        self,
        f,
        active,
        particle_start,
        n_particles,
        mat_rho,
        mat_stiffness,
        mat_exponent,
        mat_mu,
        mat_gamma,
        pos,
    ):
        sph_runtime._kernel_add_particles(
            self._system,
            f,
            active,
            particle_start,
            n_particles,
            mat_rho,
            mat_stiffness,
            mat_exponent,
            mat_mu,
            mat_gamma,
            pos,
        )

    def _kernel_reorder_particles(self, f):
        if self._system.rigid_coupling is not None:
            sph_runtime._kernel_reorder_particles(
                self._system,
                f,
                True,
                self._system.rigid_coupling,
                self._system.rigid_coupling.rigid.n_geoms,
            )
        else:
            sph_runtime._kernel_reorder_particles(self._system, f, False, None, 0)

    def _kernel_compute_rho(self, f):
        sph_runtime._kernel_compute_rho(self._system, f)

    def _kernel_get_entity_frame(self, f, particle_start, n_particles, pos, vel):
        sph_runtime.kernel_get_entity_frame(self._system, f, particle_start, n_particles, pos, vel)

    def _kernel_set_particles_pos(self, particles_idx, envs_idx, poss):
        sph_runtime._kernel_set_particles_pos(self._system, particles_idx, envs_idx, poss)

    def _kernel_get_particles_pos(self, particle_start, n_particles, envs_idx, poss):
        sph_runtime._kernel_get_particles_pos(self._system, particle_start, n_particles, envs_idx, poss)

    def _kernel_set_particles_vel(self, particles_idx, envs_idx, vels):
        sph_runtime._kernel_set_particles_vel(self._system, particles_idx, envs_idx, vels)

    def _kernel_get_particles_vel(self, particle_start, n_particles, envs_idx, vels):
        sph_runtime._kernel_get_particles_vel(self._system, particle_start, n_particles, envs_idx, vels)

    def _kernel_set_particles_active(self, particles_idx, envs_idx, actives):
        sph_runtime._kernel_set_particles_active(self._system, particles_idx, envs_idx, actives)

    def _kernel_get_particles_active(self, particle_start, n_particles, envs_idx, actives):
        sph_runtime._kernel_get_particles_active(self._system, particle_start, n_particles, envs_idx, actives)

    def _kernel_get_mass(self, particle_start, n_particles, mass, envs_idx):
        sph_runtime._kernel_get_mass(self._system, particle_start, n_particles, mass, envs_idx)

    def process_input(self, in_backward=False):
        for entity in self.entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self.entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        self._system.substep_pre_coupling(f)

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        self._system.substep_post_coupling(f)

    def substep_post_coupling_grad(self, f):
        pass

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        pass

    def add_grad_from_state(self, state):
        pass

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            sph_runtime._kernel_set_state(self._system, f, state.pos, state.vel, state.active)

    def get_state(self, f):
        if self.is_active:
            state = SPHSolverState(self.scene)
            sph_runtime._kernel_get_state(self._system, f, state.pos, state.vel, state.active)
        else:
            state = None
        return state

    def update_render_fields(self):
        sph_runtime._kernel_update_render_fields(self._system, self.sim.cur_substep_local)

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        else:
            return sum([entity.n_particles for entity in self._entities])

    @property
    def particle_volume(self):
        return self._particle_volume

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def support_radius(self):
        return self._support_radius

    @property
    def hash_grid_res(self):
        return self.sh.grid_res

    @property
    def hash_grid_cell_size(self):
        return self.sh.cell_size

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound
