from __future__ import annotations

from enum import IntEnum
from typing import TypeVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.misc import qd_to_numpy

from .contact import CONTACT_CONFIG_DEFAULTS
from .contact_system import ContactSystem
from .finite_element import FEMDiagPreconditioner, FiniteElementMethod
from .finite_element.fem_contact_assemble import (
    distribute_fem_fem_kernel,
    distribute_fem_gradient_kernel,
)
from .global_body_manager import GlobalBodyManager
from .global_linear_system import GlobalLinearSystem
from .global_surface_manager import GlobalSurfaceManager
from .global_vertex_manager import GlobalVertexManager
from .rigid_contact_assemble import RigidContactAssemble
from .rigid_contact_proxy import RigidContactProxySystem
from .rigid_joint_forest import RigidJointForestSystem
from .rigid_system import RigidSystem
from .sim_config import SimConfig
from .sim_system import SimSystem
from .standard_pcg_solver import StandardPCGSolver

T = TypeVar("T", bound=SimSystem)


class ContactCheckpoint(IntEnum):
    FRAME = 0
    FRICTION = 1
    COUNT = 2
    FILTER = 3
    SORT = 4
    SOLVE = 5
    QUERY = 6
    CCD = 7
    LINE_SEARCH = 8
    FINALIZE = 9


@qd.data_oriented
class SimEngine:
    """GPU graph-native Newton engine composed explicitly from CGQ systems."""

    def __init__(self) -> None:
        if gs.backend == gs.cpu:
            raise RuntimeError("SimEngine does not support the CPU backend")
        if not gs.use_ndarray:
            raise RuntimeError("SimEngine requires the ndarray backend")
        # The mixed Rigid + FEM graph compiles about 22% faster on Windows
        # (28.44 s -> 22.14 s after the Quadrants pass fixes), with a measured
        # ~3.5% steady-step cost. Prefer development iteration speed.
        qd.cfg.advanced_optimization = False

        self.systems: dict[type, SimSystem] = {}
        self.is_built_host = False
        self.is_initialized_host = False
        self.params_wired_host = False

        self.newton_cond = qd.ndarray(qd.i32, shape=())
        self.ls_cond = qd.ndarray(qd.i32, shape=())
        self.converged = qd.ndarray(qd.i32, shape=())
        self.frame_failed = qd.ndarray(qd.i32, shape=())
        self.newton_iter = qd.ndarray(qd.i32, shape=())
        self.ls_iter = qd.ndarray(qd.i32, shape=())
        self.alpha = qd.ndarray(qd.f64, shape=())
        self.max_disp = qd.ndarray(qd.f64, shape=())
        self.energy_delta = qd.ndarray(qd.f64, shape=())
        self.checkpoint_never_yield = qd.ndarray(qd.i32, shape=())
        self.checkpoint_never_yield.from_numpy(np.array(0, dtype=np.int32))

        self.sim_config = SimConfig()
        self.add_system(self.sim_config)

    def add_system(self, system: SimSystem) -> None:
        if self.is_built_host:
            raise RuntimeError("add_system() is only valid before build_systems()")
        system_type = type(system)
        if system_type in self.systems:
            raise RuntimeError(f"SimSystem {system_type.__name__} is already registered")
        system._set_engine(self)
        self.systems[system_type] = system

    def find(self, system_type: type[T]) -> T | None:
        system = self.systems.get(system_type)
        if system is None or not system.is_valid:
            return None
        return system

    def require(self, system_type: type[T]) -> T:
        system = self.find(system_type)
        if system is None:
            raise RuntimeError(f"Required system {system_type.__name__} is not registered")
        return system

    def build_systems(self) -> None:
        if self.is_built_host:
            raise RuntimeError("SimEngine systems are already built")
        for system in self.systems.values():
            system.do_build()

        self.rigid = self.find(RigidSystem)
        self.fem = self.find(FiniteElementMethod)
        self.global_body_manager = self.find(GlobalBodyManager)
        self.global_vertex_manager = self.find(GlobalVertexManager)
        self.global_surface_manager = self.find(GlobalSurfaceManager)
        self.contact = self.find(ContactSystem)
        self.rigid_contact_proxy = self.find(RigidContactProxySystem)
        self.rigid_contact_assemble = self.find(RigidContactAssemble)
        self.rigid_forest = self.find(RigidJointForestSystem)
        self.global_linear_system = self.require(GlobalLinearSystem)
        self.pcg_solver = self.require(StandardPCGSolver)
        self.fem_preconditioner = self.find(FEMDiagPreconditioner)
        if self.fem is not None and self.fem_preconditioner is None:
            raise RuntimeError("FiniteElementMethod requires FEMDiagPreconditioner")
        if self.fem is not None and (
            self.global_body_manager is None
            or self.global_vertex_manager is None
            or self.global_surface_manager is None
        ):
            raise RuntimeError("FiniteElementMethod requires global body, vertex, and surface managers")

        self.has_rigid = self.rigid is not None
        self.has_fem = self.fem is not None
        self.has_contact = self.contact is not None
        self.has_rigid_contact_proxy = self.rigid_contact_proxy is not None
        self.has_rigid_contact_assemble = self.rigid_contact_assemble is not None
        self.has_rigid_forest = self.rigid_forest is not None
        if self.has_rigid_contact_proxy != self.has_rigid_forest:
            raise RuntimeError(
                "RigidContactProxySystem and RigidJointForestSystem must be registered together"
            )
        if self.has_rigid_contact_proxy != self.has_rigid_contact_assemble:
            raise RuntimeError(
                "RigidContactProxySystem and RigidContactAssemble must be registered together"
            )
        if self.has_contact and not self.has_fem:
            raise RuntimeError("The current ContactSystem milestone requires FiniteElementMethod")
        if not self.has_rigid and not self.has_fem:
            raise RuntimeError("SimEngine requires RigidSystem or FiniteElementMethod")
        self.is_built_host = True

    def wire_solver_params(
        self,
        dt: float,
        tol: float,
        max_newton_iter: int,
        max_pcg_iter: int,
        max_ls_iter: int,
        pcg_tol_rate: float,
    ) -> None:
        if not self.is_built_host:
            raise RuntimeError("build_systems() must run before wire_solver_params()")
        self.sim_config.dt.from_numpy(np.array(dt, dtype=np.float64))
        self.sim_config.tol.from_numpy(np.array(tol, dtype=np.float64))
        self.sim_config.max_newton_iter.from_numpy(np.array(max_newton_iter, dtype=np.int64))
        self.sim_config.max_pcg_iter.from_numpy(np.array(max_pcg_iter, dtype=np.int64))
        self.sim_config.max_ls_iter.from_numpy(np.array(max_ls_iter, dtype=np.int64))
        self.max_ls_iter_host = max_ls_iter
        self.pcg_tol_rate_host = pcg_tol_rate
        self.energy_buf = qd.ndarray(qd.f64, shape=(max_ls_iter + 1,))
        self.params_wired_host = True

    def init(self) -> None:
        if self.is_initialized_host:
            raise RuntimeError("SimEngine is already initialized")
        if not self.params_wired_host:
            raise RuntimeError("wire_solver_params() must run before init()")

        dof_offset = 0
        if self.rigid is not None:
            self.rigid.init(dof_offset)
            dof_offset += self.rigid.storage_dof_count_host
        dof_block_base = dof_offset // 3

        n_elastic_triplets = 0
        if self.fem is not None:
            self.fem.init(dof_offset)
            dof_offset += self.fem.vert_capacity_host * 3
            n_elastic_triplets = self.fem.n_elastic_triplets()

        proxy_dof_offset = dof_offset
        if self.rigid_contact_proxy is not None:
            dof_offset += self.rigid_contact_proxy.n_pairs_host * 6

        n_block_rows = dof_offset // 3
        max_contact_body_triplets = self.contact.unique_triplet_rows.shape[0] if self.contact is not None else 0
        if self.rigid_contact_assemble is not None:
            max_contact_body_triplets = (
                self.contact.unique_triplet_rows.shape[0] * 4
                + self.contact.unique_doublet_vertices.shape[0]
            )
        self.global_linear_system.init(
            n_block_rows,
            n_elastic_triplets,
            max_contact_body_triplets,
            dof_block_base,
            self.pcg_tol_rate_host,
        )
        self.pcg_solver.init(dof_offset, n_block_rows, self.pcg_tol_rate_host)
        if self.rigid_forest is not None:
            self.rigid_forest.init(
                dof_offset,
                self.rigid_contact_proxy.n_bodies_host,
                proxy_dof_offset,
            )
            self.rigid_contact_assemble.init()
        if self.fem_preconditioner is not None:
            self.fem_preconditioner.init()
        self._initialize_global_resources()
        if self.contact is not None:
            self._initialize_contact()
        self.is_initialized_host = True

    @qd.kernel(fastcache=True)
    def _initialize_global_resources(self):
        if qd.static(self.has_fem):
            self.fem.initialize_global_vertices(self.global_vertex_manager)
            self.global_body_manager.compute_vertex_offsets(self.fem)
        if qd.static(self.has_rigid_contact_proxy):
            self.rigid_contact_proxy.initialize_proxy_state()
            self.rigid_contact_proxy.prepare_metric()
            self.rigid_contact_proxy.prepare_constraint()
            self.rigid_contact_proxy.initialize_global_vertices(self.global_vertex_manager)

    @qd.kernel(graph=True, checkpoints=True, fastcache=True)
    def _init_contact_kernel(
        self,
        pair_overflow: qd.types.ndarray(qd.i32, ndim=0),
    ):
        with qd.checkpoint(ContactCheckpoint.FRAME, yield_on=self.checkpoint_never_yield):
            if qd.static(True):
                self.fem.forward_global_vertices(self.global_vertex_manager)
                self.global_vertex_manager.reset_trajectory()
                self.contact.bvh_triangle_build()
                self.contact.bvh_edge_build()
        with qd.checkpoint(ContactCheckpoint.QUERY, yield_on=pair_overflow):
            if qd.static(True):
                self.contact.reset_collision_counts()
                self.contact.trajectory_query()
        with qd.checkpoint(ContactCheckpoint.COUNT, yield_on=self.checkpoint_never_yield):
            if qd.static(True):
                self.contact.count_active()

    def _handle_pair_overflow(self) -> None:
        self.contact.handle_broad_phase_overflow()
        self.contact.realloc_pair_buffers(
            pt=int(qd_to_numpy(self.contact.n_pairs_pt)),
            ee=int(qd_to_numpy(self.contact.n_pairs_ee)),
            pe=int(qd_to_numpy(self.contact.n_pairs_pe)),
            pp=int(qd_to_numpy(self.contact.n_pairs_pp)),
            ph=int(qd_to_numpy(self.contact.n_pairs_ph)),
        )

    def _initialize_contact(self) -> None:
        pair_overflow = self.contact.overflow_flag
        status = self._init_contact_kernel(pair_overflow)
        while status.yielded:
            if status.checkpoint != ContactCheckpoint.QUERY:
                raise RuntimeError(f"Unexpected contact init checkpoint {status.checkpoint}")
            self._handle_pair_overflow()
            status = self._init_contact_kernel.resume(
                pair_overflow,
                from_checkpoint=ContactCheckpoint.QUERY,
            )
        if bool(qd_to_numpy(self.contact.intersection_flag)):
            raise RuntimeError("ContactSystem initial state contains an intersection")

    @qd.kernel(graph=True, checkpoints=True, fastcache=True)
    def _step_kernel(
        self,
        pair_overflow: qd.types.ndarray(qd.i32, ndim=0),
        assembly_overflow: qd.types.ndarray(qd.i32, ndim=0),
        triplet_overflow: qd.types.ndarray(qd.i32, ndim=0),
        friction_overflow: qd.types.ndarray(qd.i32, ndim=0),
    ):
        with qd.checkpoint(ContactCheckpoint.FRAME, yield_on=self.checkpoint_never_yield):
            if qd.static(self.has_contact):
                self.contact.adaptive_kappa_update()
            if qd.static(self.has_rigid):
                self.rigid.predict()
                self.rigid.assemble_candidate_rows()
            if qd.static(self.has_rigid_contact_proxy):
                self.rigid_contact_proxy.mark_mechanism_constrained()
            if qd.static(self.has_rigid):
                self.rigid.initialize_newton()
            if qd.static(self.has_rigid_contact_proxy):
                self.rigid_contact_proxy.reset_frame()
                self.rigid_contact_proxy.prepare_metric()
            if qd.static(self.has_fem):
                self.fem.predict(self.sim_config)
                self.fem.forward_global_vertices(self.global_vertex_manager)
            for _ in range(1):
                self.newton_iter[()] = 0
                self.ls_iter[()] = 0
                self.frame_failed[()] = 0
                self.newton_cond[()] = 1

        with qd.checkpoint(ContactCheckpoint.FRICTION, yield_on=friction_overflow):
            if qd.static(self.has_contact):  # noqa: SIM102
                if qd.static(self.contact.has_friction):
                    self.contact.friction_snapshot()

        while qd.graph.do_while(self.newton_cond):
            with qd.checkpoint(ContactCheckpoint.COUNT, yield_on=assembly_overflow):
                if qd.static(self.has_rigid_contact_proxy):
                    self.rigid_contact_proxy.prepare_constraint()
                if qd.static(self.has_contact):
                    self.contact.count_active()

            with qd.checkpoint(ContactCheckpoint.FILTER, yield_on=self.checkpoint_never_yield):
                if qd.static(self.has_contact):
                    self.contact.adaptive_kappa_newton_tick()
                    self.contact.filter_assemble()

            with qd.checkpoint(ContactCheckpoint.SORT, yield_on=triplet_overflow):
                if qd.static(self.has_rigid_contact_assemble):
                    self.contact.sort_reduce()
                    self.fem.report_extent(self.global_linear_system)
                    self.rigid_contact_assemble.classify()
                    self.global_linear_system.derive_extents()
                else:
                    if qd.static(self.has_fem):
                        self.fem.report_extent(self.global_linear_system)
                    if qd.static(True):
                        self.global_linear_system.derive_extents()
                    if qd.static(self.has_contact):
                        self.contact.sort_reduce()
                        self.global_linear_system.compute_n_triplets(self.contact)

            with qd.checkpoint(ContactCheckpoint.SOLVE, yield_on=self.checkpoint_never_yield):
                if qd.static(True):
                    self.global_linear_system.zero_rhs()
                    self.global_linear_system.zero_triplet()

                if qd.static(self.has_rigid):
                    self.rigid.assemble(self.sim_config, self.global_linear_system)
                if qd.static(self.has_fem):
                    self.fem.assemble(self.sim_config, self.global_linear_system)
                if qd.static(self.has_rigid_contact_assemble):
                    self.rigid_contact_assemble.distribute()
                elif qd.static(self.has_contact):
                    distribute_fem_gradient_kernel(self.contact, self.fem, self.global_linear_system)
                    distribute_fem_fem_kernel(self.contact, self.fem, self.global_linear_system)

                if qd.static(self.has_fem):
                    self.global_linear_system.body_sort_reduce()
                    self.global_linear_system.traverse(self.fem_preconditioner, self.has_fem)
                if qd.static(self.has_rigid_forest):
                    self.rigid_forest.prepare_particular()
                    self.rigid_forest.particular_spmv(self.global_linear_system)
                    self.rigid_forest.project_physical_rhs(self.global_linear_system)

                if qd.static(True):
                    self.pcg_solver.solve(
                        self.global_linear_system,
                        self.rigid,
                        self.rigid_forest,
                        self.fem_preconditioner,
                        self.has_rigid,
                        self.has_rigid_forest,
                        self.has_fem,
                        self.sim_config.max_pcg_iter[()],
                    )

                if qd.static(self.has_rigid):
                    self.rigid.negate_dq(self.global_linear_system)
                if qd.static(self.has_fem):
                    self.fem.negate_dx(self.global_linear_system)
                if qd.static(self.has_rigid_forest):
                    self.rigid_forest.expand_solution(self.global_linear_system.x_sol)

                for _ in range(1):
                    self.max_disp[()] = qd.f64(0.0)
                if qd.static(self.has_fem):
                    self.fem.contribute_newton_max_disp(self.max_disp)

                for _ in range(1):
                    rigid_converged = True
                    fem_converged = True
                    if qd.static(self.has_rigid):
                        rigid_converged = self.rigid.gradient_squared[()] <= self.sim_config.tol[()] ** 2
                    if qd.static(self.has_fem):
                        fem_converged = self.max_disp[()] <= self.sim_config.tol[()] * self.sim_config.dt[()]
                    self.converged[()] = qd.i32(rigid_converged and fem_converged)

                if qd.static(self.has_rigid):
                    self.rigid.record_start_point()
                    self.rigid.energy(self.sim_config)
                if qd.static(self.has_fem):
                    self.fem.record_start_point()
                    self.fem.energy(self.sim_config)
                    self.global_vertex_manager.record_safe_positions()
                    self.fem.publish_trajectory_end_positions(self.global_vertex_manager)
                if qd.static(self.has_rigid_contact_proxy):
                    self.rigid_contact_proxy.record_start_point()
                    self.rigid_contact_proxy.forward_global_vertices(self.global_vertex_manager)
                    self.rigid_contact_proxy.publish_trajectory_end_positions(
                        self.global_vertex_manager
                    )
                if qd.static(self.has_contact):
                    self.contact.contact_energy()

                for _ in range(1):
                    baseline = qd.f64(0.0)
                    if qd.static(self.has_rigid):
                        baseline = baseline + self.rigid.rigid_energy[()]
                    if qd.static(self.has_fem):
                        baseline = baseline + self.fem.fem_energy[()]
                    if qd.static(self.has_contact):
                        baseline = baseline + self.contact.contact_energy_value[()]
                    self.energy_buf[0] = baseline

                if qd.static(self.has_contact):
                    self.contact.bvh_triangle_build()
                    self.contact.bvh_edge_build()

            with qd.checkpoint(ContactCheckpoint.QUERY, yield_on=pair_overflow):
                if qd.static(self.has_contact):
                    self.contact.reset_collision_counts()
                    self.contact.trajectory_query()

            with qd.checkpoint(ContactCheckpoint.CCD, yield_on=self.checkpoint_never_yield):
                if qd.static(self.has_contact):
                    self.contact.init_ccd()
                    self.contact.ccd()

            with qd.checkpoint(ContactCheckpoint.LINE_SEARCH, yield_on=self.checkpoint_never_yield):
                for _ in range(1):
                    initial_alpha = qd.f64(1.0)
                    if qd.static(self.has_contact):
                        initial_alpha = qd.min(initial_alpha, self.contact.ccd_alpha[()])
                    self.alpha[()] = initial_alpha
                    self.ls_cond[()] = qd.i32(self.converged[()] == 0)
                    self.ls_iter[()] = 0

                while qd.graph.do_while(self.ls_cond):
                    trial_alpha = self.alpha[()]
                    if self.converged[()] != 0 or self.pcg_solver.linear_pcg.is_failed[()] != 0:
                        trial_alpha = qd.f64(0.0)

                    if qd.static(self.has_rigid):
                        self.rigid.step_forward(trial_alpha)
                        self.rigid.energy(self.sim_config)
                    if qd.static(self.has_fem):
                        self.fem.step_forward(trial_alpha)
                        self.fem.forward_global_vertices(self.global_vertex_manager)
                        self.fem.energy(self.sim_config)
                    if qd.static(self.has_rigid_contact_proxy):
                        self.rigid_contact_proxy.step_forward(trial_alpha)
                        self.rigid_contact_proxy.forward_global_vertices(
                            self.global_vertex_manager
                        )
                    if qd.static(self.has_contact):
                        self.contact.contact_energy()

                    for _ in range(1):
                        trial_energy = qd.f64(0.0)
                        if qd.static(self.has_rigid):
                            trial_energy = trial_energy + self.rigid.rigid_energy[()]
                        if qd.static(self.has_fem):
                            trial_energy = trial_energy + self.fem.fem_energy[()]
                        if qd.static(self.has_contact):
                            trial_energy = trial_energy + self.contact.contact_energy_value[()]
                        self.energy_delta[()] = trial_energy - self.energy_buf[0]

                    for _ in range(1):
                        accepted = self.energy_delta[()] <= 0.0
                        if accepted:
                            self.ls_cond[()] = 0
                        else:
                            self.alpha[()] = self.alpha[()] * 0.5
                            self.ls_iter[()] = self.ls_iter[()] + 1
                            if self.ls_iter[()] >= self.sim_config.max_ls_iter[()]:
                                self.alpha[()] = qd.f64(0.0)
                                self.ls_cond[()] = 0

                for _ in range(1):
                    if self.converged[()] != 0 or self.pcg_solver.linear_pcg.is_failed[()] != 0:
                        self.alpha[()] = qd.f64(0.0)
                if qd.static(self.has_fem):
                    self.fem.step_forward(self.alpha[()])
                    self.fem.forward_global_vertices(self.global_vertex_manager)
                if qd.static(self.has_rigid):
                    self.rigid.step_forward(self.alpha[()])
                if qd.static(self.has_rigid_contact_proxy):
                    self.rigid_contact_proxy.step_forward(self.alpha[()])
                    self.rigid_contact_proxy.forward_global_vertices(
                        self.global_vertex_manager
                    )

                for _ in range(1):
                    rejected = self.converged[()] == 0 and self.alpha[()] == 0.0
                    pcg_failed = self.pcg_solver.linear_pcg.is_failed[()] != 0
                    linear_failed = self.global_linear_system.triplet_overflow[()] != 0
                    if qd.static(self.has_fem):
                        linear_failed = linear_failed or self.global_linear_system.bcoo_valid[()] == 0
                    if qd.static(self.has_contact):
                        linear_failed = linear_failed or self.contact.intersection_flag[()] != 0

                    if self.converged[()] == 0 and not rejected and not pcg_failed and not linear_failed:
                        self.newton_iter[()] = self.newton_iter[()] + 1
                    exhausted = self.newton_iter[()] >= self.sim_config.max_newton_iter[()]
                    failed = rejected or pcg_failed or linear_failed or (exhausted and self.converged[()] == 0)
                    self.frame_failed[()] = qd.i32(failed)
                    self.newton_cond[()] = qd.i32(self.converged[()] == 0 and not failed and not exhausted)

                if qd.static(self.has_rigid):
                    self.rigid.set_newton_active(self.newton_cond[()])
                    self.rigid.build_preconditioner(compute_envelope=False)

        with qd.checkpoint(ContactCheckpoint.FINALIZE, yield_on=self.checkpoint_never_yield):
            if qd.static(self.has_rigid):
                self.rigid.update_velocity()
            if qd.static(self.has_fem):
                self.fem.update_velocity(self.sim_config)
                self.fem.copy_x_prev()
            if qd.static(self.has_rigid_contact_proxy):
                self.rigid_contact_proxy.copy_previous_state()

    def step(self) -> None:
        if not self.is_initialized_host:
            raise RuntimeError("SimEngine.init() must run before step()")
        if self.contact is None:
            pair_overflow = self.checkpoint_never_yield
            assembly_overflow = self.checkpoint_never_yield
            friction_overflow = self.checkpoint_never_yield
        else:
            pair_overflow = self.contact.overflow_flag
            assembly_overflow = self.contact.count_overflow_flag
            friction_overflow = self.contact.friction_overflow_flag
        triplet_overflow = self.global_linear_system.triplet_overflow

        status = self._step_kernel(
            pair_overflow,
            assembly_overflow,
            triplet_overflow,
            friction_overflow,
        )
        while status.yielded:
            checkpoint = status.checkpoint
            if self.contact is not None and checkpoint == ContactCheckpoint.FRICTION:
                required = {
                    channel: int(qd_to_numpy(getattr(self.contact, f"n_friction_pairs_{channel}")))
                    for channel in ("pt", "ee", "pe", "pp", "ph")
                }
                self.contact.realloc_friction_pair_buffers(required)
                friction_overflow.from_numpy(np.array(0, dtype=np.int32))
                resume_from = ContactCheckpoint.FRICTION
            elif self.contact is not None and checkpoint == ContactCheckpoint.COUNT:
                required_doublets = int(qd_to_numpy(self.contact.n_counted_doublets)) + int(
                    qd_to_numpy(self.contact.n_friction_demand_doublets)
                )
                required_triplets = int(qd_to_numpy(self.contact.n_counted_triplets)) + int(
                    qd_to_numpy(self.contact.n_friction_demand_triplets)
                )
                self.contact.realloc_assembly_buffers(required_doublets, required_triplets)
                assembly_overflow.from_numpy(np.array(0, dtype=np.int32))
                resume_from = ContactCheckpoint.FILTER
            elif checkpoint == ContactCheckpoint.SORT:
                required = int(qd_to_numpy(self.global_linear_system.n_triplets))
                grow_factor = CONTACT_CONFIG_DEFAULTS["extras/capacity_grow_factor"]
                self.global_linear_system.realloc_triplet_buffers(
                    max(
                        int(np.ceil(required * grow_factor)),
                        self.global_linear_system.triplet_row.shape[0] + 1,
                    ),
                    live_size=required,
                )
                triplet_overflow.from_numpy(np.array(0, dtype=np.int32))
                resume_from = ContactCheckpoint.SOLVE
            elif self.contact is not None and checkpoint == ContactCheckpoint.QUERY:
                self._handle_pair_overflow()
                resume_from = ContactCheckpoint.QUERY
            else:
                raise RuntimeError(f"Unexpected timestep checkpoint {checkpoint}")

            status = self._step_kernel.resume(
                pair_overflow,
                assembly_overflow,
                triplet_overflow,
                friction_overflow,
                from_checkpoint=resume_from,
            )
        if self.fem is not None:
            self.fem._forward_scene_vertices()
        if bool(qd_to_numpy(self.frame_failed)):
            details = [
                f"newton={self.get_newton_iters()}",
                f"pcg={self.get_max_pcg_iters()}",
                f"line_search={self.get_max_ls_iters()}",
                f"triplet_overflow={int(qd_to_numpy(self.global_linear_system.triplet_overflow))}",
                f"bcoo_valid={int(qd_to_numpy(self.global_linear_system.bcoo_valid))}",
            ]
            if self.contact is not None:
                details.extend(
                    (
                        f"intersection={int(qd_to_numpy(self.contact.intersection_flag))}",
                        f"pairs_pt={int(qd_to_numpy(self.contact.n_pairs_pt))}",
                        f"pairs_ee={int(qd_to_numpy(self.contact.n_pairs_ee))}",
                        f"pairs_ph={int(qd_to_numpy(self.contact.n_pairs_ph))}",
                        f"active_pairs={int(qd_to_numpy(self.contact.n_active_pairs))}",
                        f"ccd_alpha={float(qd_to_numpy(self.contact.ccd_alpha)):.6g}",
                    )
                )
                for channel in ("pt", "ee", "ph"):
                    count = int(qd_to_numpy(getattr(self.contact, f"n_pairs_{channel}")))
                    if count:
                        alphas = qd_to_numpy(getattr(self.contact, f"ccd_alpha_{channel}"))[:count]
                        minimum_index = int(np.argmin(alphas))
                        pair = qd_to_numpy(getattr(self.contact, f"pairs_{channel}"))[minimum_index]
                        details.append(
                            f"ccd_{channel}=({float(alphas[minimum_index]):.6g},"
                            f" pair={tuple(int(value) for value in pair)})"
                        )
                        if channel == "ph":
                            surface_vertex = int(pair[0])
                            vertex_id = int(qd_to_numpy(self.global_surface_manager.surf_verts)[surface_vertex])
                            current = qd_to_numpy(self.global_vertex_manager.positions)[vertex_id]
                            endpoint = qd_to_numpy(self.global_vertex_manager.trajectory_end_positions)[vertex_id]
                            thickness = float(qd_to_numpy(self.global_vertex_manager.thicknesses)[vertex_id])
                            details.append(
                                f"ph_path=(current={tuple(float(value) for value in current)},"
                                f" endpoint={tuple(float(value) for value in endpoint)},"
                                f" thickness={thickness:.6g})"
                            )
            raise RuntimeError(f"SimEngine Newton solve failed ({', '.join(details)})")

    def get_newton_iters(self) -> int:
        return int(qd_to_numpy(self.newton_iter))

    def get_max_pcg_iters(self) -> int:
        return int(qd_to_numpy(self.pcg_solver.linear_pcg.n_iterations))

    def get_max_ls_iters(self) -> int:
        return int(qd_to_numpy(self.ls_iter))
