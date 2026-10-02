from __future__ import annotations

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.abd.forward_kinematics import (
    func_forward_velocity,
    func_update_cartesian_space,
    func_update_geom_aabbs,
)
from genesis.engine.solvers.rigid.collider import broadphase, contact, narrowphase
from genesis.engine.solvers.rigid.constraint import linesearch, solver
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver, func_step_1, func_step_2
from genesis.utils import geom as gu

from .sim_system import SimSystem

RIGID_DYNAMICS_BACKENDS = ("genesis", "cgq_mincoo")


def validate_rigid_dynamics_backend(value: object) -> str:
    backend = str(value)
    if backend not in RIGID_DYNAMICS_BACKENDS:
        choices = ", ".join(repr(item) for item in RIGID_DYNAMICS_BACKENDS)
        raise ValueError(f"Unsupported rigid/dynamics_backend {backend!r}; expected one of {choices}")
    return backend


@qd.func
def _cgq_wrap_revolute(value):
    return value - qd.floor(value / 6.283185307179586 + 0.5) * 6.283185307179586


@qd.func
def _cgq_free_flow_rate(quaternion, omega_body, inertia, inertia_inverse):
    omega_quaternion = qd.Vector(
        [0.0, omega_body[0], omega_body[1], omega_body[2]],
        dt=qd.f64,
    )
    quaternion_rate = 0.5 * gu.qd_quat_mul(quaternion, omega_quaternion)
    angular_momentum = inertia @ omega_body
    omega_rate = inertia_inverse @ angular_momentum.cross(omega_body)
    return quaternion_rate, omega_rate


@qd.func
def _cgq_free_flow(quaternion_start, omega_world, inertia, inertia_inverse, dt):
    quaternion = quaternion_start
    omega_body = gu.qd_inv_transform_by_quat(omega_world, quaternion_start)
    substep_dt = dt / 4.0
    for _ in qd.static(range(4)):
        k1_q, k1_w = _cgq_free_flow_rate(
            quaternion,
            omega_body,
            inertia,
            inertia_inverse,
        )
        k2_q, k2_w = _cgq_free_flow_rate(
            (quaternion + 0.5 * substep_dt * k1_q).normalized(),
            omega_body + 0.5 * substep_dt * k1_w,
            inertia,
            inertia_inverse,
        )
        k3_q, k3_w = _cgq_free_flow_rate(
            (quaternion + 0.5 * substep_dt * k2_q).normalized(),
            omega_body + 0.5 * substep_dt * k2_w,
            inertia,
            inertia_inverse,
        )
        k4_q, k4_w = _cgq_free_flow_rate(
            (quaternion + substep_dt * k3_q).normalized(),
            omega_body + substep_dt * k3_w,
            inertia,
            inertia_inverse,
        )
        quaternion = (quaternion + substep_dt / 6.0 * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)).normalized()
        omega_body = omega_body + substep_dt / 6.0 * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)
    return quaternion, gu.qd_transform_by_quat(omega_body, quaternion)


@qd.func
def _cgq_second_moment(inertia):
    return 0.5 * inertia.trace() * qd.Matrix.identity(qd.f64, 3) - inertia


@qd.func
def _cgq_is_spd_3x3(matrix):
    minor_1 = matrix[0, 0]
    minor_2 = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    determinant = (
        matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
        - matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
        + matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
    )
    return minor_1 > 0.0 and minor_2 > 0.0 and determinant > 0.0


@qd.data_oriented
class RigidSystem(SimSystem):
    """Expose the Genesis minimal-coordinate Rigid numerical core to the global Newton runtime."""

    def __init__(
        self,
        rigid_solver: RigidSolver,
        *,
        dynamics_backend: str = "genesis",
        joint_limit_kappa: float = 1.0e6,
    ) -> None:
        super().__init__()
        self.dynamics_backend = validate_rigid_dynamics_backend(dynamics_backend)
        self.uses_cgq_mincoo = self.dynamics_backend == "cgq_mincoo"
        self.cgq_joint_limit_kappa = float(joint_limit_kappa)
        if self.cgq_joint_limit_kappa < 0.0:
            raise ValueError("rigid/joint_limit_kappa must be non-negative")
        if gs.qd_float != qd.f64:
            raise RuntimeError("The Rigid Newton framework requires double precision")
        if rigid_solver._requires_grad:
            raise RuntimeError("The Rigid Newton framework does not support differentiable simulation")
        if rigid_solver.rigid_config.solver_type != gs.constraint_solver.Newton:
            raise RuntimeError("The Rigid Newton framework requires the native Newton constraint formulation")
        if rigid_solver._options.noslip_iterations > 0:
            raise RuntimeError("The Rigid Newton framework does not support the noslip post-processing solve")
        if rigid_solver.rigid_config.use_hibernation:
            raise RuntimeError("The Rigid Newton framework does not support hibernation")

        self.dyn_state = rigid_solver.dyn_state
        self.constraint_state = rigid_solver.constraint_solver.constraint_state
        self.dyn_info = rigid_solver.dyn_info
        self.rigid_info = rigid_solver.rigid_info
        self.rigid_config = rigid_solver.rigid_config
        self.collider_state = rigid_solver.collider.collider_state
        self.collider_info = rigid_solver.collider.collider_info
        self.collider_config = rigid_solver.collider.collider_config
        self.geoms_init_AABB = rigid_solver.geoms_init_AABB
        self.errno = rigid_solver._errno

        self.mpr_state = rigid_solver.collider._mpr.mpr_state
        self.gjk_state = rigid_solver.collider._gjk.gjk_state
        self.gjk_config = rigid_solver.collider._gjk.gjk_config

        self.has_constraints = not self.uses_cgq_mincoo and not rigid_solver._disable_constraint
        self.has_collision = not self.uses_cgq_mincoo and rigid_solver._enable_collision
        self.has_split_narrowphase = rigid_solver.collider._use_split_narrowphase
        self.has_cooperative_contact_pruning = (
            gs.backend != gs.cpu
            and rigid_solver.collider.collider_config.has_prunable_contacts
            and (rigid_solver._options.contact_pruning_tolerance or 0.0) > 0.0
            and rigid_solver._B * 2 <= rigid_solver.collider._gpu_cores
        )
        if self.has_split_narrowphase:
            self.contact0_mpr_state = rigid_solver.collider.contact0_mpr_state
            self.contact0_gjk_state = rigid_solver.collider.contact0_gjk_state
            self.multicontact_mpr_state = rigid_solver.collider.multicontact_mpr_state
            self.multicontact_gjk_state = rigid_solver.collider.multicontact_gjk_state
            self.contact0_n_chunks = rigid_solver.collider._contact0_n_chunks
            self.multicontact_n_total_threads = rigid_solver.collider._multicontact_n_total_threads
            self.multicontact_max_items_per_thread = rigid_solver.collider._multicontact_max_items_per_thread

        self.n_dofs_per_instance_host = rigid_solver.n_dofs
        self.n_instances_host = rigid_solver._B
        self.dof_count_host = self.n_dofs_per_instance_host * self.n_instances_host
        self.storage_dof_count_host = ((self.dof_count_host + 2) // 3) * 3
        self.n_dofs_per_instance = qd.ndarray(qd.i32, shape=())
        self.n_instances = qd.ndarray(qd.i32, shape=())
        self.n_links = qd.ndarray(qd.i32, shape=())
        self.n_dofs = qd.ndarray(qd.i32, shape=())
        self.dof_offset = qd.ndarray(qd.i32, shape=())
        self.n_storage_dofs = qd.ndarray(qd.i32, shape=())
        self.gradient_squared = qd.ndarray(qd.f64, shape=())
        self.rigid_energy = qd.ndarray(qd.f64, shape=())
        self.qacc_temp = qd.ndarray(qd.f64, shape=self.constraint_state.qacc.shape)
        self.Ma_temp = qd.ndarray(qd.f64, shape=self.constraint_state.Ma.shape)
        self.Jaref_temp = qd.ndarray(qd.f64, shape=self.constraint_state.Jaref.shape)
        body_capacity = max(rigid_solver.n_links * rigid_solver._B, 1)
        self.cgq_t_prev = qd.Vector.tensor(3, qd.f64, shape=(body_capacity,))
        self.cgq_quat_prev = qd.Vector.tensor(4, qd.f64, shape=(body_capacity,))
        self.cgq_velocity = qd.Vector.tensor(3, qd.f64, shape=(body_capacity,))
        self.cgq_omega = qd.Vector.tensor(3, qd.f64, shape=(body_capacity,))
        self.cgq_t_tilde = qd.Vector.tensor(3, qd.f64, shape=(body_capacity,))
        self.cgq_quat_tilde = qd.Vector.tensor(4, qd.f64, shape=(body_capacity,))
        self.cgq_quat_free_tilde = qd.Vector.tensor(4, qd.f64, shape=(body_capacity,))
        self.cgq_omega_tilde = qd.Vector.tensor(3, qd.f64, shape=(body_capacity,))
        self.cgq_t_temp = qd.Vector.tensor(3, qd.f64, shape=(body_capacity,))
        self.cgq_quat_temp = qd.Vector.tensor(4, qd.f64, shape=(body_capacity,))
        self.cgq_inertia_inverse = qd.ndarray(qd.f64, shape=(body_capacity, 3, 3))
        self.cgq_qpos_temp = qd.tensor(qd.f64, shape=self.rigid_info.qpos.shape)
        self.cgq_qpos_prev = qd.tensor(qd.f64, shape=self.rigid_info.qpos.shape)
        self.n_dofs_per_instance.from_numpy(np.array(self.n_dofs_per_instance_host, dtype=np.int32))
        self.n_instances.from_numpy(np.array(self.n_instances_host, dtype=np.int32))
        self.n_links.from_numpy(np.array(rigid_solver.n_links, dtype=np.int32))
        self.n_dofs.from_numpy(np.array(self.dof_count_host, dtype=np.int32))
        self.dof_offset.from_numpy(np.array(0, dtype=np.int32))
        self.n_storage_dofs.from_numpy(np.array(self.storage_dof_count_host, dtype=np.int32))
        self.h = rigid_solver._substep_dt
        self.h4 = self.h**4
        self.is_forward_pos_updated = rigid_solver._is_forward_pos_updated
        self.is_forward_vel_updated = rigid_solver._is_forward_vel_updated
        self.forest = None

    def configure_genesis_collision(self, enabled: bool) -> None:
        if self.uses_cgq_mincoo and enabled:
            raise ValueError("cgq_mincoo does not support Genesis native rigid collision")
        self.has_collision = self.has_collision and enabled

    def do_build(self) -> None:
        if self.uses_cgq_mincoo:
            from .rigid_joint_forest import RigidJointForestSystem

            self.forest = self.require(RigidJointForestSystem)

    def init(self, dof_offset: int) -> None:
        self.dof_offset.from_numpy(np.array(dof_offset, dtype=np.int32))

    @qd.func(requires_top_level=True)
    def initialize_cgq_state(self):
        if qd.static(self.uses_cgq_mincoo):
            for body in range(self.forest.n_mechanism_bodies[()]):
                environment = body // self.n_links[()]
                link = body - environment * self.n_links[()]
                link_index = [link, environment] if qd.static(self.rigid_config.batch_links_info) else link
                position = self.forest.endpoint_t[body]
                quaternion = self.forest.endpoint_quat[body]
                angular_velocity = self.dyn_state.links.cd_ang[link, environment]
                offset = position - self.dyn_state.links.root_COM[link, environment]
                linear_velocity = self.dyn_state.links.cd_vel[
                    link,
                    environment,
                ] + angular_velocity.cross(offset)
                inertia = self.dyn_info.links.inertial_i[link_index]
                inertia_inverse = inertia.inverse()
                for axis in qd.static(range(3)):
                    self.cgq_t_prev[body][axis] = position[axis]
                    self.cgq_velocity[body][axis] = linear_velocity[axis]
                    self.cgq_omega[body][axis] = angular_velocity[axis]
                    self.cgq_t_tilde[body][axis] = position[axis]
                    self.cgq_t_temp[body][axis] = position[axis]
                    self.cgq_omega_tilde[body][axis] = angular_velocity[axis]
                    for column in qd.static(range(3)):
                        self.cgq_inertia_inverse[body, axis, column] = inertia_inverse[axis, column]
                for axis in qd.static(range(4)):
                    self.cgq_quat_prev[body][axis] = quaternion[axis]
                    self.cgq_quat_tilde[body][axis] = quaternion[axis]
                    self.cgq_quat_free_tilde[body][axis] = quaternion[axis]
                    self.cgq_quat_temp[body][axis] = quaternion[axis]
            for q, environment in qd.ndrange(
                self.rigid_info.qpos.shape[0],
                self.n_instances[()],
            ):
                value = self.rigid_info.qpos[q, environment]
                self.cgq_qpos_prev[q, environment] = value
                self.cgq_qpos_temp[q, environment] = value

    @qd.func(requires_top_level=True)
    def _cgq_predict(self):
        for body in range(self.forest.n_mechanism_bodies[()]):
            environment = body // self.n_links[()]
            link = body - environment * self.n_links[()]
            link_index = [link, environment] if qd.static(self.rigid_config.batch_links_info) else link
            position = self.forest.endpoint_t[body]
            quaternion = self.forest.endpoint_quat[body]
            for axis in qd.static(range(3)):
                self.cgq_t_prev[body][axis] = position[axis]
            for axis in qd.static(range(4)):
                self.cgq_quat_prev[body][axis] = quaternion[axis]

            fixed = self.forest.parent_body[body] < 0 and self.forest.root_dof_index[body] < 0
            predicted_position = position
            predicted_quaternion = quaternion
            predicted_omega = qd.Vector.zero(qd.f64, 3)
            if not fixed:
                entity = self.dyn_info.links.entity_idx[link_index]
                gravity_scale = 1.0 - self.dyn_info.entities.gravity_compensation[entity]
                predicted_position = (
                    position
                    + self.h * self.cgq_velocity[body]
                    + self.h * self.h * gravity_scale * self.rigid_info.gravity[environment]
                )
                inertia = self.dyn_info.links.inertial_i[link_index]
                inertia_inverse = qd.Matrix.zero(qd.f64, 3, 3)
                for row in qd.static(range(3)):
                    for column in qd.static(range(3)):
                        inertia_inverse[row, column] = self.cgq_inertia_inverse[body, row, column]
                predicted_quaternion, predicted_omega = _cgq_free_flow(
                    quaternion,
                    self.cgq_omega[body],
                    inertia,
                    inertia_inverse,
                    self.h,
                )
            for axis in qd.static(range(3)):
                self.cgq_t_tilde[body][axis] = predicted_position[axis]
                self.cgq_omega_tilde[body][axis] = predicted_omega[axis]
            for axis in qd.static(range(4)):
                self.cgq_quat_tilde[body][axis] = predicted_quaternion[axis]
                self.cgq_quat_free_tilde[body][axis] = predicted_quaternion[axis]

        for q, environment in qd.ndrange(
            self.rigid_info.qpos.shape[0],
            self.n_instances[()],
        ):
            self.cgq_qpos_prev[q, environment] = self.rigid_info.qpos[q, environment]

    @qd.func(requires_top_level=True)
    def predict(self):
        if qd.static(self.uses_cgq_mincoo):
            self._cgq_predict()
        else:
            func_step_1(
                self.dyn_state,
                self.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                self.is_forward_pos_updated,
                self.is_forward_vel_updated,
                False,
            )

    @qd.func(requires_top_level=True)
    def assemble_candidate_rows(self):
        if qd.static(self.has_constraints):
            solver.func_add_equality_constraints(
                self.dyn_state,
                self.collider_state,
                self.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )

        if qd.static(self.has_collision):
            func_update_geom_aabbs(self.geoms_init_AABB, self.dyn_state, self.rigid_config)
            broadphase.func_broad_phase_device(
                self.dyn_state,
                self.collider_state,
                self.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.collider_info,
                self.rigid_config,
                self.errno,
            )

            if qd.static(self.has_split_narrowphase):
                narrowphase.func_reset_narrowphase_work_queues(self.collider_state)
                narrowphase.func_narrowphase_contact0(
                    self.geoms_init_AABB,
                    self.dyn_state,
                    self.collider_state,
                    self.contact0_mpr_state,
                    self.contact0_gjk_state,
                    self.dyn_info,
                    self.rigid_info,
                    self.collider_info,
                    self.rigid_config,
                    self.collider_config,
                    self.n_instances[()],
                    self.contact0_n_chunks,
                    self.errno,
                )
                narrowphase.func_narrowphase_multicontact(
                    self.geoms_init_AABB,
                    self.dyn_state,
                    self.collider_state,
                    self.multicontact_mpr_state,
                    self.multicontact_gjk_state,
                    self.dyn_info,
                    self.rigid_info,
                    self.collider_info,
                    self.rigid_config,
                    self.collider_config,
                    self.gjk_config,
                    self.multicontact_n_total_threads,
                    self.multicontact_max_items_per_thread,
                    self.errno,
                )
            elif qd.static(self.collider_config.has_non_box_plane_convex_convex):
                narrowphase.func_narrow_phase_convex_vs_convex(
                    self.geoms_init_AABB,
                    self.dyn_state,
                    self.collider_state,
                    self.mpr_state,
                    self.gjk_state,
                    self.gjk_state.diff_contact_input,
                    self.dyn_info,
                    self.rigid_info,
                    self.collider_info,
                    self.rigid_config,
                    self.collider_config,
                    self.gjk_config,
                    self.errno,
                )

            if qd.static(self.collider_config.has_convex_specialization):
                narrowphase.func_narrow_phase_convex_specializations(
                    self.geoms_init_AABB,
                    self.dyn_state,
                    self.collider_state,
                    self.dyn_info,
                    self.rigid_info,
                    self.collider_info,
                    self.rigid_config,
                    self.collider_config,
                    self.errno,
                )
            if qd.static(self.collider_config.has_terrain):
                narrowphase.func_narrow_phase_any_vs_terrain(
                    self.geoms_init_AABB,
                    self.dyn_state,
                    self.collider_state,
                    self.mpr_state,
                    self.dyn_info,
                    self.rigid_info,
                    self.collider_info,
                    self.rigid_config,
                    self.collider_config,
                    self.errno,
                )
            if qd.static(self.collider_config.has_nonconvex_nonterrain):
                narrowphase.func_narrow_phase_nonconvex_vs_nonterrain(
                    self.geoms_init_AABB,
                    self.dyn_state,
                    self.collider_state,
                    self.dyn_info,
                    self.rigid_info,
                    self.collider_info,
                    self.rigid_config,
                    self.collider_config,
                    self.errno,
                )

            if qd.static(self.has_cooperative_contact_pruning):
                contact.func_clamp_prune_contacts_coop(
                    self.dyn_state,
                    self.collider_state,
                    self.rigid_info,
                    self.collider_info,
                    self.errno,
                )
            else:
                contact.func_clamp_prune_contacts(
                    self.dyn_state,
                    self.collider_state,
                    self.rigid_info,
                    self.collider_info,
                    self.rigid_config,
                    self.collider_config,
                    self.errno,
                )

        if qd.static(not self.has_collision):
            for i_b in range(self.n_instances[()]):
                self.collider_state.n_contacts[i_b] = 0
                self.collider_state.n_contacts_hibernated[i_b] = 0

        if qd.static(self.has_constraints):
            solver.func_add_inequality_constraints(
                self.dyn_state,
                self.collider_state,
                self.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                self.collider_config,
                include_collision=self.has_collision,
            )

    @qd.func(requires_top_level=True)
    def initialize_newton(self):
        if qd.static(self.has_constraints):
            solver.func_solve_init(
                self.dyn_state,
                self.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                True,
            )
            self.set_newton_active(1)
            self.build_preconditioner(compute_envelope=True)
            solver.func_update_gradient_no_solve(
                self.dyn_state,
                self.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )

    @qd.func(requires_top_level=True)
    def set_newton_active(self, is_active):
        for i_b in range(self.n_instances[()]):
            has_constraints = self.constraint_state.n_constraints[i_b] > 0 and is_active != 0
            self.constraint_state.improved[i_b] = has_constraints
            for i_island in range(self.constraint_state.island.n_islands[i_b]):
                self.constraint_state.island.improved[i_island, i_b] = has_constraints

    @qd.func(requires_top_level=True)
    def build_preconditioner(self, compute_envelope: qd.template()):
        if qd.static(not self.uses_cgq_mincoo):
            solver.func_hessian_and_cholesky_factor_direct(
                self.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                compute_envelope=compute_envelope,
            )

    @qd.func(requires_top_level=True)
    def _cgq_assemble(self, global_linear_system: qd.template()):
        for _ in range(1):
            self.gradient_squared[()] = qd.f64(0.0)

        for body in range(self.forest.n_mechanism_bodies[()]):
            environment = body // self.n_links[()]
            link = body - environment * self.n_links[()]
            link_index = [link, environment] if qd.static(self.rigid_config.batch_links_info) else link
            fixed = self.forest.parent_body[body] < 0 and self.forest.root_dof_index[body] < 0
            gradient = qd.Vector.zero(qd.f64, 6)
            hessian = qd.Matrix.zero(qd.f64, 6, 6)
            if fixed:
                hessian = qd.Matrix.identity(qd.f64, 6)
            else:
                mass = self.dyn_info.links.inertial_mass[link_index]
                inertia = self.dyn_info.links.inertial_i[link_index]
                rotation = gu.qd_quat_to_R(
                    self.forest.endpoint_quat[body],
                    qd.f64(1.0e-12),
                )
                predicted_rotation = gu.qd_quat_to_R(
                    self.cgq_quat_tilde[body],
                    qd.f64(1.0e-12),
                )
                second_moment = _cgq_second_moment(inertia)
                potential_rotation = predicted_rotation @ second_moment @ rotation.transpose()
                position_delta = self.forest.endpoint_t[body] - self.cgq_t_tilde[body]
                for axis in qd.static(range(3)):
                    gradient[axis] = mass * position_delta[axis]
                    hessian[axis, axis] = mass
                gradient[3] = potential_rotation[1, 2] - potential_rotation[2, 1]
                gradient[4] = potential_rotation[2, 0] - potential_rotation[0, 2]
                gradient[5] = potential_rotation[0, 1] - potential_rotation[1, 0]

                rotational_hessian = potential_rotation.trace() * qd.Matrix.identity(qd.f64, 3) - 0.5 * (
                    potential_rotation + potential_rotation.transpose()
                )
                if not _cgq_is_spd_3x3(rotational_hessian):
                    rotational_hessian = rotation @ inertia @ rotation.transpose()
                for row in qd.static(range(3)):
                    for column in qd.static(range(3)):
                        hessian[row + 3, column + 3] = rotational_hessian[row, column]

            for row in qd.static(range(6)):
                self.forest.body_wrench[body, row] = gradient[row]
                for column in qd.static(range(6)):
                    self.forest.body_inertia[body, row, column] = hessian[row, column]

        self.forest.restrict_body_wrenches(global_linear_system.b_rhs)
        self.forest.cgq_control_gradient(global_linear_system.b_rhs)
        for i_d, i_b in qd.ndrange(
            self.n_dofs_per_instance[()],
            self.n_instances[()],
        ):
            i_global = self.dof_offset[()] + i_b * self.n_dofs_per_instance[()] + i_d
            gradient = global_linear_system.b_rhs[i_global]
            qd.atomic_add(self.gradient_squared[()], gradient * gradient)
        for i_padding in range(self.n_dofs[()], self.n_storage_dofs[()]):
            global_linear_system.b_rhs[self.dof_offset[()] + i_padding] = qd.f64(0.0)

    @qd.func(requires_top_level=True)
    def assemble(
        self,
        sim_config: qd.template(),
        global_linear_system: qd.template(),
        displacement_coordinates: qd.template(),
    ):
        if qd.static(self.uses_cgq_mincoo):
            self._cgq_assemble(global_linear_system)
        else:
            for _ in range(1):
                self.gradient_squared[()] = qd.f64(0.0)
            gradient_scale = self.h4
            if qd.static(displacement_coordinates):
                gradient_scale = self.h * self.h
            for i_d, i_b in qd.ndrange(self.n_dofs_per_instance[()], self.n_instances[()]):
                i_global = self.dof_offset[()] + i_b * self.n_dofs_per_instance[()] + i_d
                gradient = qd.f64(0.0)
                gradient_unscaled = qd.f64(0.0)
                has_live_constraints = False
                if qd.static(self.has_constraints):
                    has_live_constraints = self.constraint_state.n_constraints[i_b] > 0
                if has_live_constraints:
                    gradient_unscaled = self.constraint_state.grad[i_d, i_b]
                    gradient = gradient_scale * gradient_unscaled
                global_linear_system.b_rhs[i_global] = gradient
                qd.atomic_add(self.gradient_squared[()], gradient_unscaled * gradient_unscaled)

            for i_padding in range(self.n_dofs[()], self.n_storage_dofs[()]):
                global_linear_system.b_rhs[self.dof_offset[()] + i_padding] = qd.f64(0.0)

    @qd.func(requires_top_level=True)
    def energy(self, sim_config: qd.template()):
        for _ in range(1):
            self.rigid_energy[()] = qd.f64(0.0)
        if qd.static(self.uses_cgq_mincoo):
            for body in range(self.forest.n_mechanism_bodies[()]):
                environment = body // self.n_links[()]
                link = body - environment * self.n_links[()]
                link_index = [link, environment] if qd.static(self.rigid_config.batch_links_info) else link
                fixed = self.forest.parent_body[body] < 0 and self.forest.root_dof_index[body] < 0
                if not fixed:
                    position_delta = self.forest.endpoint_t[body] - self.cgq_t_tilde[body]
                    value = 0.5 * self.dyn_info.links.inertial_mass[link_index] * position_delta.norm_sqr()
                    relative = gu.qd_quat_mul(
                        gu.qd_inv_quat(self.forest.endpoint_quat[body]),
                        self.cgq_quat_tilde[body],
                    )
                    vector = qd.Vector(
                        [relative[1], relative[2], relative[3]],
                        dt=qd.f64,
                    )
                    second_moment = _cgq_second_moment(self.dyn_info.links.inertial_i[link_index])
                    value = value + 2.0 * (
                        vector.norm_sqr() * second_moment.trace() - vector.dot(second_moment @ vector)
                    )
                    qd.atomic_add(self.rigid_energy[()], value)
            self.forest.cgq_control_energy(self.rigid_energy)
        elif qd.static(self.has_constraints):
            for i_b in range(self.n_instances[()]):
                qd.atomic_add(self.rigid_energy[()], self.h4 * self.constraint_state.cost[i_b])

    @qd.func(requires_top_level=True)
    def _genesis_apply_hessian_impl(
        self,
        x: qd.template(),
        y: qd.template(),
        displacement_coordinates: qd.template(),
        component_labels: qd.template(),
        converged: qd.template(),
        masked: qd.template(),
    ):
        for i_d, i_b in qd.ndrange(self.n_dofs_per_instance[()], self.n_instances[()]):
            i_global = self.dof_offset[()] + i_b * self.n_dofs_per_instance[()] + i_d
            is_active = True
            if qd.static(masked):
                block_row = (self.dof_offset[()] + i_b * self.n_dofs_per_instance[()]) // 3
                is_active = converged[component_labels[block_row]] == 0
            if is_active:
                self.constraint_state.search[i_d, i_b] = x[i_global]

        for i_b in range(self.n_instances[()]):
            is_active = self.n_dofs_per_instance[()] > 0
            if qd.static(masked):
                if is_active:
                    block_row = (self.dof_offset[()] + i_b * self.n_dofs_per_instance[()]) // 3
                    is_active = converged[component_labels[block_row]] == 0
            if is_active:
                if qd.static(self.has_constraints) and self.constraint_state.n_constraints[i_b] > 0:
                    for i_island in range(self.constraint_state.island.n_islands[i_b]):
                        n_dofs = self.constraint_state.island.dof_slices.n[i_island, i_b]
                        if qd.static(self.rigid_config.is_single_island):
                            n_dofs = self.n_dofs_per_instance[()]
                        i_dof_start = self.constraint_state.island.dof_slices.start[i_island, i_b]

                        for j_d_local in range(n_dofs):
                            j_d = j_d_local
                            if qd.static(not self.rigid_config.is_single_island or self.rigid_config.sparse_solve):
                                j_d = self.constraint_state.island.dof_id[i_dof_start + j_d_local, i_b]
                            value = gs.qd_float(0.0)
                            for i_d_local in range(j_d_local, n_dofs):
                                i_d = i_d_local
                                if qd.static(not self.rigid_config.is_single_island or self.rigid_config.sparse_solve):
                                    i_d = self.constraint_state.island.dof_id[i_dof_start + i_d_local, i_b]
                                scale = gs.qd_float(1.0)
                                if qd.static(self.rigid_config.enable_jacobi_equilibration):
                                    scale = self.constraint_state.nt_jacobi[i_d, i_b]
                                value = value + (
                                    self.constraint_state.nt_H[i_b, i_d, j_d]
                                    * self.constraint_state.search[i_d, i_b]
                                    / scale
                                )
                            self.constraint_state.Mgrad[j_d, i_b] = value

                        for i_d_local in range(n_dofs):
                            i_d = i_d_local
                            if qd.static(not self.rigid_config.is_single_island or self.rigid_config.sparse_solve):
                                i_d = self.constraint_state.island.dof_id[i_dof_start + i_d_local, i_b]
                            value = gs.qd_float(0.0)
                            for j_d_local in range(i_d_local + 1):
                                j_d = j_d_local
                                if qd.static(not self.rigid_config.is_single_island or self.rigid_config.sparse_solve):
                                    j_d = self.constraint_state.island.dof_id[i_dof_start + j_d_local, i_b]
                                value = value + (
                                    self.constraint_state.nt_H[i_b, i_d, j_d] * self.constraint_state.Mgrad[j_d, i_b]
                                )
                            scale = gs.qd_float(1.0)
                            if qd.static(self.rigid_config.enable_jacobi_equilibration):
                                scale = self.constraint_state.nt_jacobi[i_d, i_b]
                            self.constraint_state.grad[i_d, i_b] = value / scale
                else:
                    for i_d in range(self.n_dofs_per_instance[()]):
                        value = gs.qd_float(0.0)
                        for j_d in range(self.n_dofs_per_instance[()]):
                            value = value + (
                                self.rigid_info.mass_mat[i_d, j_d, i_b] * self.constraint_state.search[j_d, i_b]
                            )
                        self.constraint_state.grad[i_d, i_b] = value

        hessian_scale = self.h4
        if qd.static(displacement_coordinates):
            hessian_scale = 1.0
        for i_d, i_b in qd.ndrange(self.n_dofs_per_instance[()], self.n_instances[()]):
            i_global = self.dof_offset[()] + i_b * self.n_dofs_per_instance[()] + i_d
            is_active = True
            if qd.static(masked):
                block_row = (self.dof_offset[()] + i_b * self.n_dofs_per_instance[()]) // 3
                is_active = converged[component_labels[block_row]] == 0
            if is_active:
                y[i_global] = y[i_global] + hessian_scale * self.constraint_state.grad[i_d, i_b]

        for i_padding in range(self.n_dofs[()], self.n_storage_dofs[()]):
            i_global = self.dof_offset[()] + i_padding
            is_active = True
            if qd.static(masked):
                is_active = converged[component_labels[i_global // 3]] == 0
            if is_active:
                y[i_global] = y[i_global] + x[i_global]

    @qd.func(requires_top_level=True)
    def _cgq_apply_hessian(
        self,
        x: qd.template(),
        y: qd.template(),
    ):
        self.forest.forest_inertia_wrench()
        self.forest.restrict_body_wrenches(y)
        self.forest.cgq_control_matvec(x, y)
        for i_padding in range(self.n_dofs[()], self.n_storage_dofs[()]):
            i_global = self.dof_offset[()] + i_padding
            y[i_global] = y[i_global] + x[i_global]

    @qd.func(requires_top_level=True)
    def _cgq_apply_hessian_masked(
        self,
        x: qd.template(),
        y: qd.template(),
        component_labels: qd.template(),
        converged: qd.template(),
    ):
        self.forest.forest_inertia_wrench_masked(
            component_labels,
            converged,
        )
        self.forest.restrict_body_wrenches_masked(
            y,
            component_labels,
            converged,
        )
        self.forest.cgq_control_matvec_masked(
            x,
            y,
            component_labels,
            converged,
        )
        for i_padding in range(self.n_dofs[()], self.n_storage_dofs[()]):
            i_global = self.dof_offset[()] + i_padding
            if converged[component_labels[i_global // 3]] == 0:
                y[i_global] = y[i_global] + x[i_global]

    @qd.func(requires_top_level=True)
    def apply_hessian(
        self,
        x: qd.template(),
        y: qd.template(),
        displacement_coordinates: qd.template(),
    ):
        if qd.static(self.uses_cgq_mincoo):
            self._cgq_apply_hessian(x, y)
        else:
            self._genesis_apply_hessian_impl(
                x,
                y,
                displacement_coordinates,
                x,
                x,
                False,
            )

    @qd.func(requires_top_level=True)
    def apply_hessian_masked(
        self,
        x: qd.template(),
        y: qd.template(),
        displacement_coordinates: qd.template(),
        component_labels: qd.template(),
        converged: qd.template(),
    ):
        if qd.static(self.uses_cgq_mincoo):
            self._cgq_apply_hessian_masked(
                x,
                y,
                component_labels,
                converged,
            )
        else:
            self._genesis_apply_hessian_impl(
                x,
                y,
                displacement_coordinates,
                component_labels,
                converged,
                True,
            )

    @qd.func(requires_top_level=True)
    def _apply_preconditioner_impl(
        self,
        residual: qd.template(),
        result: qd.template(),
        displacement_coordinates: qd.template(),
        component_labels: qd.template(),
        converged: qd.template(),
        masked: qd.template(),
    ):
        if qd.static(self.has_constraints):
            for i_d, i_b in qd.ndrange(self.n_dofs_per_instance[()], self.n_instances[()]):
                i_global = self.dof_offset[()] + i_b * self.n_dofs_per_instance[()] + i_d
                is_active = True
                if qd.static(masked):
                    block_row = (self.dof_offset[()] + i_b * self.n_dofs_per_instance[()]) // 3
                    is_active = converged[component_labels[block_row]] == 0
                if is_active:
                    self.constraint_state.grad[i_d, i_b] = residual[i_global]

            for i_b in range(self.n_instances[()]):
                is_active = self.n_dofs_per_instance[()] > 0
                if qd.static(masked):
                    if is_active:
                        block_row = (self.dof_offset[()] + i_b * self.n_dofs_per_instance[()]) // 3
                        is_active = converged[component_labels[block_row]] == 0
                if is_active and self.constraint_state.n_constraints[i_b] > 0:
                    for i_island in range(self.constraint_state.island.n_islands[i_b]):
                        solver.func_cholesky_solve_batch(
                            i_b,
                            i_island,
                            rhs=self.constraint_state.grad,
                            out=self.constraint_state.Mgrad,
                            constraint_state=self.constraint_state,
                            rigid_config=self.rigid_config,
                        )

            preconditioner_scale = 1.0 / self.h4
            if qd.static(displacement_coordinates):
                preconditioner_scale = 1.0
            for i_d, i_b in qd.ndrange(self.n_dofs_per_instance[()], self.n_instances[()]):
                i_global = self.dof_offset[()] + i_b * self.n_dofs_per_instance[()] + i_d
                is_active = True
                if qd.static(masked):
                    block_row = (self.dof_offset[()] + i_b * self.n_dofs_per_instance[()]) // 3
                    is_active = converged[component_labels[block_row]] == 0
                if is_active and self.constraint_state.n_constraints[i_b] > 0:
                    result[i_global] = preconditioner_scale * self.constraint_state.Mgrad[i_d, i_b]

        for i_padding in range(self.n_dofs[()], self.n_storage_dofs[()]):
            i_global = self.dof_offset[()] + i_padding
            is_active = True
            if qd.static(masked):
                is_active = converged[component_labels[i_global // 3]] == 0
            if is_active:
                result[i_global] = residual[i_global]

    @qd.func(requires_top_level=True)
    def apply_preconditioner(
        self,
        residual: qd.template(),
        result: qd.template(),
        displacement_coordinates: qd.template(),
    ):
        self._apply_preconditioner_impl(
            residual,
            result,
            displacement_coordinates,
            residual,
            residual,
            False,
        )

    @qd.func(requires_top_level=True)
    def apply_preconditioner_masked(
        self,
        residual: qd.template(),
        result: qd.template(),
        displacement_coordinates: qd.template(),
        component_labels: qd.template(),
        converged: qd.template(),
    ):
        self._apply_preconditioner_impl(
            residual,
            result,
            displacement_coordinates,
            component_labels,
            converged,
            True,
        )

    @qd.func(requires_top_level=True)
    def negate_dq(
        self,
        global_linear_system: qd.template(),
        displacement_coordinates: qd.template(),
    ):
        if qd.static(self.uses_cgq_mincoo):
            for i_d, i_b in qd.ndrange(
                self.n_dofs_per_instance[()],
                self.n_instances[()],
            ):
                i_global = self.dof_offset[()] + i_b * self.n_dofs_per_instance[()] + i_d
                self.constraint_state.search[i_d, i_b] = -global_linear_system.x_sol[i_global]
        elif qd.static(self.has_constraints):
            for i_d, i_b in qd.ndrange(self.n_dofs_per_instance[()], self.n_instances[()]):
                i_global = self.dof_offset[()] + i_b * self.n_dofs_per_instance[()] + i_d
                direction = global_linear_system.x_sol[i_global]
                if qd.static(displacement_coordinates):
                    direction = direction / (self.h * self.h)
                self.constraint_state.search[i_d, i_b] = -direction
                self.constraint_state.Mgrad[i_d, i_b] = direction
            for i_b in range(self.n_instances[()]):
                if self.constraint_state.n_constraints[i_b] > 0:
                    linesearch.func_mv_jv_dense(i_b, self.constraint_state, self.rigid_info)

    @qd.func(requires_top_level=True)
    def record_start_point(self):
        if qd.static(self.uses_cgq_mincoo):
            for q, environment in qd.ndrange(
                self.rigid_info.qpos.shape[0],
                self.n_instances[()],
            ):
                self.cgq_qpos_temp[q, environment] = self.rigid_info.qpos[q, environment]
            for body in range(self.forest.n_mechanism_bodies[()]):
                for axis in qd.static(range(3)):
                    self.cgq_t_temp[body][axis] = self.forest.endpoint_t[body][axis]
                for axis in qd.static(range(4)):
                    self.cgq_quat_temp[body][axis] = self.forest.endpoint_quat[body][axis]
        elif qd.static(self.has_constraints):
            for i_d, i_b in qd.ndrange(self.n_dofs_per_instance[()], self.n_instances[()]):
                self.qacc_temp[i_d, i_b] = self.constraint_state.qacc[i_d, i_b]
                self.Ma_temp[i_d, i_b] = self.constraint_state.Ma[i_d, i_b]
            for i_c, i_b in qd.ndrange(self.constraint_state.Jaref.shape[0], self.n_instances[()]):
                if i_c < self.constraint_state.n_constraints[i_b]:
                    self.Jaref_temp[i_c, i_b] = self.constraint_state.Jaref[i_c, i_b]

    @qd.func(requires_top_level=True)
    def _cgq_step_forward(self, alpha):
        for q, environment in qd.ndrange(
            self.rigid_info.qpos.shape[0],
            self.n_instances[()],
        ):
            self.rigid_info.qpos[q, environment] = self.cgq_qpos_temp[q, environment]

        for link, environment in qd.ndrange(
            self.n_links[()],
            self.n_instances[()],
        ):
            link_index = [link, environment] if qd.static(self.rigid_config.batch_links_info) else link
            for joint in range(
                self.dyn_info.links.joint_start[link_index],
                self.dyn_info.links.joint_end[link_index],
            ):
                joint_index = [joint, environment] if qd.static(self.rigid_config.batch_joints_info) else joint
                joint_type = self.dyn_info.joints.type[joint_index]
                q_start = self.dyn_info.joints.q_start[joint_index]
                dof_start = self.dyn_info.joints.dof_start[joint_index]
                if joint_type == gs.JOINT_TYPE.FREE:
                    body = environment * self.n_links[()] + link
                    endpoint_position = qd.Vector.zero(qd.f64, 3)
                    for axis in qd.static(range(3)):
                        endpoint_position[axis] = (
                            self.cgq_t_temp[body][axis] - alpha * self.forest.body_twist[body, axis]
                        )
                    rotation = qd.Vector(
                        [
                            -alpha * self.forest.body_twist[body, 3],
                            -alpha * self.forest.body_twist[body, 4],
                            -alpha * self.forest.body_twist[body, 5],
                        ],
                        dt=qd.f64,
                    )
                    delta = gu.qd_rotvec_to_quat(
                        rotation,
                        self.rigid_info.EPS[None],
                    )
                    endpoint_quaternion = gu.qd_transform_quat_by_quat(
                        self.cgq_quat_temp[body],
                        delta,
                    )
                    inertial_quaternion = self.dyn_info.links.inertial_quat[link_index]
                    quaternion = gu.qd_quat_mul(
                        endpoint_quaternion,
                        gu.qd_inv_quat(inertial_quaternion),
                    ).normalized()
                    position = endpoint_position - gu.qd_transform_by_quat(
                        self.dyn_info.links.inertial_pos[link_index],
                        quaternion,
                    )
                    for axis in qd.static(range(3)):
                        self.rigid_info.qpos[q_start + axis, environment] = position[axis]
                    for axis in qd.static(range(4)):
                        self.rigid_info.qpos[q_start + 3 + axis, environment] = quaternion[axis]
                elif joint_type != gs.JOINT_TYPE.FIXED:
                    for local_dof in range(
                        self.dyn_info.joints.dof_end[joint_index] - dof_start,
                    ):
                        self.rigid_info.qpos[q_start + local_dof, environment] = (
                            self.cgq_qpos_temp[q_start + local_dof, environment]
                            + alpha * self.constraint_state.search[dof_start + local_dof, environment]
                        )

    @qd.func(requires_top_level=True)
    def step_forward(self, alpha):
        if qd.static(self.uses_cgq_mincoo):
            self._cgq_step_forward(alpha)
        elif qd.static(self.has_constraints):
            for i_d, i_b in qd.ndrange(self.n_dofs_per_instance[()], self.n_instances[()]):
                self.constraint_state.qacc[i_d, i_b] = (
                    self.qacc_temp[i_d, i_b] + alpha * self.constraint_state.search[i_d, i_b]
                )
                self.constraint_state.Ma[i_d, i_b] = self.Ma_temp[i_d, i_b] + alpha * self.constraint_state.mv[i_d, i_b]
            for i_c, i_b in qd.ndrange(self.constraint_state.Jaref.shape[0], self.n_instances[()]):
                if i_c < self.constraint_state.n_constraints[i_b]:
                    self.constraint_state.Jaref[i_c, i_b] = (
                        self.Jaref_temp[i_c, i_b] + alpha * self.constraint_state.jv[i_c, i_b]
                    )

            solver.func_update_constraint(
                qacc=self.constraint_state.qacc,
                Ma=self.constraint_state.Ma,
                cost=self.constraint_state.cost,
                dyn_state=self.dyn_state,
                constraint_state=self.constraint_state,
                rigid_config=self.rigid_config,
            )
            self.set_newton_active(1)
            solver.func_update_gradient_no_solve(
                self.dyn_state,
                self.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )

    @qd.func(requires_top_level=True)
    def _cgq_update_velocity(self):
        inverse_dt = 1.0 / self.h
        for body in range(self.forest.n_mechanism_bodies[()]):
            fixed = self.forest.parent_body[body] < 0 and self.forest.root_dof_index[body] < 0
            linear_velocity = qd.Vector.zero(qd.f64, 3)
            angular_velocity = qd.Vector.zero(qd.f64, 3)
            if not fixed:
                linear_velocity = inverse_dt * (self.forest.endpoint_t[body] - self.cgq_t_prev[body])
                relative = gu.qd_quat_mul(
                    self.forest.endpoint_quat[body],
                    gu.qd_inv_quat(self.cgq_quat_free_tilde[body]),
                )
                angular_velocity = self.cgq_omega_tilde[body] + inverse_dt * gu.qd_quat_to_rotvec(
                    relative,
                    self.rigid_info.EPS[None],
                )
            for axis in qd.static(range(3)):
                self.cgq_velocity[body][axis] = linear_velocity[axis]
                self.cgq_omega[body][axis] = angular_velocity[axis]
                self.cgq_t_prev[body][axis] = self.forest.endpoint_t[body][axis]
            for axis in qd.static(range(4)):
                self.cgq_quat_prev[body][axis] = self.forest.endpoint_quat[body][axis]

        func_update_cartesian_space(
            self.dyn_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            force_update_all_geoms=False,
            is_backward=False,
        )

        for link, environment in qd.ndrange(
            self.n_links[()],
            self.n_instances[()],
        ):
            link_index = [link, environment] if qd.static(self.rigid_config.batch_links_info) else link
            body = environment * self.n_links[()] + link
            for joint in range(
                self.dyn_info.links.joint_start[link_index],
                self.dyn_info.links.joint_end[link_index],
            ):
                joint_index = [joint, environment] if qd.static(self.rigid_config.batch_joints_info) else joint
                joint_type = self.dyn_info.joints.type[joint_index]
                q_start = self.dyn_info.joints.q_start[joint_index]
                dof_start = self.dyn_info.joints.dof_start[joint_index]
                if joint_type == gs.JOINT_TYPE.FREE:
                    angular_velocity = self.cgq_omega[body]
                    offset = self.forest.endpoint_t[body] - self.forest.endpoint_joint_xanchor[joint, environment]
                    joint_velocity = self.cgq_velocity[body] - angular_velocity.cross(offset)
                    for axis in qd.static(range(3)):
                        old_velocity = self.dyn_state.dofs.vel[dof_start + axis, environment]
                        new_velocity = joint_velocity[axis]
                        self.dyn_state.dofs.vel_prev[dof_start + axis, environment] = old_velocity
                        self.dyn_state.dofs.vel_next[dof_start + axis, environment] = new_velocity
                        self.dyn_state.dofs.vel[dof_start + axis, environment] = new_velocity
                        self.dyn_state.dofs.acc[dof_start + axis, environment] = (
                            new_velocity - old_velocity
                        ) * inverse_dt
                        angular_dof = dof_start + axis + 3
                        old_angular_velocity = self.dyn_state.dofs.vel[angular_dof, environment]
                        new_angular_velocity = angular_velocity.dot(
                            self.dyn_state.dofs.cdof_ang[angular_dof, environment]
                        )
                        self.dyn_state.dofs.vel_prev[angular_dof, environment] = old_angular_velocity
                        self.dyn_state.dofs.vel_next[angular_dof, environment] = new_angular_velocity
                        self.dyn_state.dofs.vel[angular_dof, environment] = new_angular_velocity
                        self.dyn_state.dofs.acc[angular_dof, environment] = (
                            new_angular_velocity - old_angular_velocity
                        ) * inverse_dt
                elif joint_type != gs.JOINT_TYPE.FIXED:
                    for local_dof in range(
                        self.dyn_info.joints.dof_end[joint_index] - dof_start,
                    ):
                        delta = (
                            self.rigid_info.qpos[q_start + local_dof, environment]
                            - self.cgq_qpos_prev[q_start + local_dof, environment]
                        )
                        if joint_type == gs.JOINT_TYPE.REVOLUTE:
                            delta = _cgq_wrap_revolute(delta)
                        dof = dof_start + local_dof
                        old_velocity = self.dyn_state.dofs.vel[dof, environment]
                        new_velocity = inverse_dt * delta
                        self.dyn_state.dofs.vel_prev[dof, environment] = old_velocity
                        self.dyn_state.dofs.vel_next[dof, environment] = new_velocity
                        self.dyn_state.dofs.vel[dof, environment] = new_velocity
                        self.dyn_state.dofs.acc[dof, environment] = (new_velocity - old_velocity) * inverse_dt

        for q, environment in qd.ndrange(
            self.rigid_info.qpos.shape[0],
            self.n_instances[()],
        ):
            value = self.rigid_info.qpos[q, environment]
            self.cgq_qpos_prev[q, environment] = value
            self.rigid_info.qpos_next[q, environment] = value

        func_forward_velocity(
            self.dyn_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            False,
        )

    @qd.func(requires_top_level=True)
    def update_velocity(self):
        if qd.static(self.uses_cgq_mincoo):
            self._cgq_update_velocity()
        else:
            if qd.static(self.has_constraints):
                solver.func_update_qacc(self.dyn_state, self.constraint_state, self.rigid_config, self.errno)
                if qd.static(self.has_collision):
                    solver.func_update_contact_force(
                        self.dyn_state,
                        self.collider_state,
                        self.constraint_state,
                        self.dyn_info,
                        self.rigid_info,
                        self.rigid_config,
                    )
            func_step_2(
                self.dyn_state,
                self.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                False,
                self.errno,
            )
