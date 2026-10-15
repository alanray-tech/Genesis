import math
from typing import TYPE_CHECKING

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine import pbd_runtime
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import PBD2DEntity, PBD3DEntity, PBDFreeParticleEntity, PBDParticleEntity
from genesis.engine.entities.pbd_entity import PBDTetEntity
from genesis.engine.materials import PBD
from genesis.engine.states.solvers import PBDSolverState
from genesis.utils.array_class import LinksState
from genesis.utils.geom import SpatialHasher

from .base_solver import GravityMixin, Solver, TimeBasedMixin, forward_to_system

if TYPE_CHECKING:
    from genesis.engine.entities import PBD2DEntity, PBD3DEntity, PBDFreeParticleEntity, PBDParticleEntity


class PBDSolver(GravityMixin, TimeBasedMixin, Solver):
    material_cls = PBD.Base
    data = forward_to_system("data")
    particles_info = forward_to_system("particles_info")
    particles_info_reordered = forward_to_system("particles_info_reordered")
    particles = forward_to_system("particles")
    particles_reordered = forward_to_system("particles_reordered")
    particles_ng = forward_to_system("particles_ng")
    particles_ng_reordered = forward_to_system("particles_ng_reordered")
    particles_render = forward_to_system("particles_render")
    edges_info = forward_to_system("edges_info")
    inner_edges_info = forward_to_system("inner_edges_info")
    elems_info = forward_to_system("elems_info")
    vverts_info = forward_to_system("vverts_info")
    vverts_render = forward_to_system("vverts_render")
    vverts_uvs = forward_to_system("vverts_uvs")
    vfaces_indices = forward_to_system("vfaces_indices")
    spatial_hasher = forward_to_system("spatial_hasher")
    _gravity = forward_to_system("_gravity", None, settable=True)

    class MATERIAL(gs.IntEnum):
        CLOTH = 0
        ELASTIC = 1
        LIQUID = 2
        PARTICLE = 3  # non-physics particles

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)
        self._particle_size = options.particle_size
        self._max_stretch_solver_iterations = options.max_stretch_solver_iterations
        self._max_bending_solver_iterations = options.max_bending_solver_iterations
        self._max_volume_solver_iterations = options.max_volume_solver_iterations
        self._max_density_solver_iterations = options.max_density_solver_iterations
        self._max_viscosity_solver_iterations = options.max_viscosity_solver_iterations

        self._n_vvert_supports = self.scene.options.vis.n_support_neighbors

        # -Neighbours_Setting-
        self.dist_scale = self.particle_radius / 0.4  # @Zhenjia: double check this
        self.h = 1.0
        self.h_2 = self.h**2
        self.h_6 = self.h**6
        self.h_9 = self.h**9

        # -POLY6_KERNEL-
        self.poly6_Coe = 315.0 / (64 * math.pi)

        # -SPIKY_KERNEL-
        self.spiky_Coe = -45.0 / math.pi

        # -LAMBDAS-
        self.lambda_epsilon = 100.0

        # -S_CORR-
        self.S_Corr_delta_q = 0.3
        self.S_Corr_k = 0.0001

        # -Gradient Approx. delta difference-
        self.g_del = 0.01

        self.vorqd_epsilon = 0.01

        # spatial hasher
        self.sh = SpatialHasher(
            cell_size=options.hash_grid_cell_size,
            grid_res=options._hash_grid_res,
        )

        # boundary
        self.setup_boundary()

    def setup_boundary(self):
        self.boundary = CubeBoundary(
            lower=self._lower_bound,
            upper=self._upper_bound,
        )

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        pass

    def build(self):
        super().build()

        self._B = self._sim._B
        self._n_particles = self.n_particles
        self._n_fluid_particles = self.n_fluid_particles
        self._n_edges = self.n_edges
        self._n_inner_edges = self.n_inner_edges
        self._n_elems = self.n_elems
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

        if self.is_active:
            self.init_ckpt()

    @property
    def is_active(self):
        return self.n_particles > 0

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "PBD2DEntity | PBD3DEntity | PBDParticleEntity | PBDFreeParticleEntity":
        if isinstance(material, gs.materials.PBD.Cloth):
            entity = PBD2DEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
                edge_start=self.n_edges,
                inner_edge_start=self.n_inner_edges,
                vvert_start=self.n_vverts,
                vface_start=self.n_vfaces,
                name=name,
            )

        elif isinstance(material, gs.materials.PBD.Elastic):
            entity = PBD3DEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
                edge_start=self.n_edges,
                elem_start=self.n_elems,
                vvert_start=self.n_vverts,
                vface_start=self.n_vfaces,
                name=name,
            )

        elif isinstance(material, gs.materials.PBD.Liquid):
            entity = PBDParticleEntity(
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

        elif isinstance(material, gs.materials.PBD.Particle):
            entity = PBDFreeParticleEntity(
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

        else:
            raise NotImplementedError()

        self._entities.append(entity)

        return entity

    def _kernel_set_particles_pos(self, particles_idx, envs_idx, poss):
        pbd_runtime._kernel_set_particles_pos(self._system, particles_idx, envs_idx, poss)

    def _kernel_get_particles_pos(self, particle_start, n_particles, envs_idx, poss):
        pbd_runtime._kernel_get_particles_pos(self._system, particle_start, n_particles, envs_idx, poss)

    def _kernel_set_particles_vel(self, particles_idx, envs_idx, vels):
        pbd_runtime._kernel_set_particles_vel(self._system, particles_idx, envs_idx, vels)

    def _kernel_get_particles_vel(self, particle_start, n_particles, envs_idx, vels):
        pbd_runtime._kernel_get_particles_vel(self._system, particle_start, n_particles, envs_idx, vels)

    def _kernel_set_particles_active(self, particles_idx, envs_idx, actives):
        pbd_runtime._kernel_set_particles_active(self._system, particles_idx, envs_idx, actives)

    def _kernel_get_particles_active(self, particle_start, n_particles, envs_idx, actives):
        pbd_runtime._kernel_get_particles_active(self._system, particle_start, n_particles, envs_idx, actives)

    def _kernel_fix_particles(self, particles_idx, envs_idx):
        pbd_runtime._kernel_fix_particles(self._system, particles_idx, envs_idx)

    def _kernel_release_particle(self, particles_idx, envs_idx):
        pbd_runtime._kernel_release_particle(self._system, particles_idx, envs_idx)

    def _kernel_get_mass(self, particle_start, n_particles, mass, envs_idx):
        pbd_runtime._kernel_get_mass(self._system, particle_start, n_particles, mass, envs_idx)

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        pass

    def substep_pre_coupling(self, f):
        self._system.substep_pre_coupling(f)

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        self._system.substep_post_coupling(f)

    def substep_post_coupling_grad(self, f):
        pass

    def collect_output_grads(self):
        pass

    def add_grad_from_state(self, state):
        pass

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            pbd_runtime._kernel_set_state(self._system, f, state.pos, state.vel, state.free)

    def get_state(self, f):
        if self.is_active:
            state = PBDSolverState(self.scene)
            pbd_runtime._kernel_get_state(self._system, f, state.pos, state.vel, state.free)
        else:
            state = None
        return state

    def get_state_render(self):
        """
        Get visual vertex positions, UVs, and face indices for rendering.

        Returns
        -------
        tuple
            (vverts_pos, vverts_uvs, vfaces_indices) - vertex positions, UV coords, and triangle indices
        """
        if not self.is_active or self._n_vverts == 0:
            return None, None, None

        # Make sure render fields are up to date
        self.update_render_fields()

        # Return the Quadrants fields directly for GPU access
        return self.vverts_render.pos, self.vverts_uvs, self.vfaces_indices

    def update_render_fields(self):
        pbd_runtime._kernel_update_render_fields(self._system, self.sim.cur_substep_local)

    @gs.assert_built
    def set_animate_particles_by_link(
        self,
        particles_idx,
        link_idx: int,
        links_state: LinksState,
        envs_idx=None,
    ) -> None:
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        self._sim._coupler.kernel_attach_pbd_to_rigid_link(self._system, particles_idx, envs_idx, link_idx, links_state)

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        return sum([entity.n_particles for entity in self._entities])

    @property
    def n_fluid_particles(self):
        if self.is_built:
            return self._n_fluid_particles
        return sum(entity.n_fluid_particles for entity in self._entities if isinstance(entity, PBDParticleEntity))

    @property
    def n_edges(self):
        if self.is_built:
            return self._n_edges
        return sum(entity.n_edges for entity in self._entities if isinstance(entity, PBDTetEntity))

    @property
    def n_inner_edges(self):
        if self.is_built:
            return self._n_inner_edges
        return sum(entity.n_inner_edges for entity in self._entities if isinstance(entity, PBD2DEntity))

    @property
    def n_elems(self):
        if self.is_built:
            return self._n_elems
        return sum(entity.n_elems for entity in self._entities if isinstance(entity, PBD3DEntity))

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
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

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
