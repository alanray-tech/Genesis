from __future__ import annotations

import numpy as np
import quadrants as qd

from .rigid_contact_proxy import RigidContactProxySystem
from .rigid_contact_proxy_kkt import (
    expand_proxy_twist,
    expand_slack_twist,
    restrict_proxy_wrench,
    restrict_slack_wrench,
)
from .rigid_system import RigidSystem
from .sim_system import SimSystem


@qd.data_oriented
class RigidJointForestSystem(SimSystem):
    """Genesis minimal-coordinate link/proxy prolongation and restriction."""

    def __init__(self, rigid_solver) -> None:
        super().__init__()
        self.is_initialized_host = False
        self.n_links_host = rigid_solver.n_links
        self.n_instances_host = rigid_solver._B
        parent_link = []
        depth = []
        for link in rigid_solver.links:
            parent_link.append(link.parent_idx)
            depth.append(0 if link.parent_idx < 0 else depth[link.parent_idx] + 1)
        self.parent_link_host = tuple(parent_link)
        self.depth_host = tuple(depth)
        self.n_levels_host = max(depth, default=0) + 1
        self.has_contact_proxy = False

    def do_build(self) -> None:
        self.rigid = self.require(RigidSystem)
        self.contact_proxy = self.find(RigidContactProxySystem)
        self.has_contact_proxy = self.contact_proxy is not None

    def init(
        self,
        total_dof: int,
        n_rigid_bodies: int,
        proxy_dof_offset: int,
    ) -> None:
        if self.is_initialized_host:
            raise RuntimeError("RigidJointForestSystem is already initialized")
        n_links = self.n_links_host
        n_instances = self.n_instances_host
        n_mechanism_bodies = n_links * n_instances
        if n_rigid_bodies < n_mechanism_bodies:
            raise ValueError("RigidJointForestSystem body count is smaller than Genesis link count")
        if total_dof < 0 or proxy_dof_offset < 0 or proxy_dof_offset > total_dof:
            raise ValueError("RigidJointForestSystem global DOF layout is invalid")
        if self.has_contact_proxy:
            required = proxy_dof_offset + self.contact_proxy.n_pairs_host * 6
            if required > total_dof:
                raise ValueError("RigidJointForestSystem proxy dummy rows exceed global DOF layout")

        parents = np.full(n_mechanism_bodies, -1, dtype=np.int32)
        depths = np.zeros(n_mechanism_bodies, dtype=np.int32)
        tree_ids = np.full(n_rigid_bodies, -1, dtype=np.int32)
        for environment in range(n_instances):
            body_offset = environment * n_links
            for link, parent_link in enumerate(self.parent_link_host):
                body = body_offset + link
                if parent_link >= 0:
                    parents[body] = body_offset + parent_link
                    depths[body] = self.depth_host[link]
                    tree_ids[body] = tree_ids[body_offset + parent_link]
                else:
                    tree_ids[body] = body

        capacity = max(n_rigid_bodies, 1)
        self.n_bodies = qd.ndarray(qd.i32, shape=())
        self.n_mechanism_bodies = qd.ndarray(qd.i32, shape=())
        self.n_levels = qd.ndarray(qd.i32, shape=())
        self.total_dof = qd.ndarray(qd.i32, shape=())
        self.proxy_dof_offset = qd.ndarray(qd.i32, shape=())
        self.parent_body = qd.ndarray(qd.i32, shape=(capacity,))
        self.depth = qd.ndarray(qd.i32, shape=(capacity,))
        self.tree_id = qd.ndarray(qd.i32, shape=(capacity,))
        self.body_twist = qd.ndarray(qd.f64, shape=(capacity, 6))
        self.body_wrench = qd.ndarray(qd.f64, shape=(capacity, 6))
        self.physical_p = qd.ndarray(qd.f64, shape=(max(total_dof, 1),))
        self.physical_Ap = qd.ndarray(qd.f64, shape=(max(total_dof, 1),))
        self.n_bodies.from_numpy(np.array(n_rigid_bodies, dtype=np.int32))
        self.n_mechanism_bodies.from_numpy(np.array(n_mechanism_bodies, dtype=np.int32))
        self.n_levels.from_numpy(np.array(int(depths.max(initial=0)) + 1, dtype=np.int32))
        self.total_dof.from_numpy(np.array(total_dof, dtype=np.int32))
        self.proxy_dof_offset.from_numpy(np.array(proxy_dof_offset, dtype=np.int32))
        self.parent_body.from_numpy(
            np.pad(parents, (0, capacity - len(parents)), constant_values=-1)
        )
        self.depth.from_numpy(np.pad(depths, (0, capacity - len(depths))))
        self.tree_id.from_numpy(tree_ids)
        self.body_twist.from_numpy(np.zeros((capacity, 6), dtype=np.float64))
        self.body_wrench.from_numpy(np.zeros((capacity, 6), dtype=np.float64))
        self.physical_p.from_numpy(np.zeros(max(total_dof, 1), dtype=np.float64))
        self.physical_Ap.from_numpy(np.zeros(max(total_dof, 1), dtype=np.float64))

        self.configuration_scale = self.rigid.h * self.rigid.h
        self.is_initialized_host = True

    @qd.func
    def _body_point_twist(self, link, environment, dof):
        angular = self.rigid.dyn_state.dofs.cdof_ang[dof, environment]
        linear = self.rigid.dyn_state.dofs.cdof_vel[dof, environment] + angular.cross(
            self.rigid.dyn_state.links.i_pos[link, environment]
            - self.rigid.dyn_state.links.root_COM[link, environment]
        )
        result = qd.Vector.zero(qd.f64, 6)
        for axis in qd.static(range(3)):
            result[axis] = linear[axis]
            result[axis + 3] = angular[axis]
        return result

    @qd.func(requires_top_level=True)
    def expand_reduced_direction(self, reduced: qd.template()):
        for body in range(self.n_bodies[()]):
            for component in qd.static(range(6)):
                self.body_twist[body, component] = 0.0

        for level in qd.static(range(self.n_levels_host)):
            for body in range(self.n_mechanism_bodies[()]):
                if self.depth[body] == level:
                    environment = body // self.n_links_host
                    link = body - environment * self.n_links_host
                    parent = self.parent_body[body]
                    twist = qd.Vector.zero(qd.f64, 6)
                    if parent >= 0:
                        parent_linear = qd.Vector.zero(qd.f64, 3)
                        parent_angular = qd.Vector.zero(qd.f64, 3)
                        for axis in qd.static(range(3)):
                            parent_linear[axis] = self.body_twist[parent, axis]
                            parent_angular[axis] = self.body_twist[parent, axis + 3]
                        offset = (
                            self.rigid.dyn_state.links.i_pos[link, environment]
                            - self.rigid.dyn_state.links.i_pos[parent % self.n_links_host, environment]
                        )
                        shifted_linear = parent_linear + parent_angular.cross(offset)
                        for axis in qd.static(range(3)):
                            twist[axis] = shifted_linear[axis]
                            twist[axis + 3] = parent_angular[axis]

                    link_index = (
                        [link, environment]
                        if qd.static(self.rigid.rigid_config.batch_links_info)
                        else link
                    )
                    for joint in range(
                        self.rigid.dyn_info.links.joint_start[link_index],
                        self.rigid.dyn_info.links.joint_end[link_index],
                    ):
                        joint_index = (
                            [joint, environment]
                            if qd.static(self.rigid.rigid_config.batch_joints_info)
                            else joint
                        )
                        for dof in range(
                            self.rigid.dyn_info.joints.dof_start[joint_index],
                            self.rigid.dyn_info.joints.dof_end[joint_index],
                        ):
                            reduced_index = (
                                self.rigid.dof_offset[()]
                                + environment * self.rigid.n_dofs_per_instance[()]
                                + dof
                            )
                            basis = self._body_point_twist(link, environment, dof)
                            coefficient = self.configuration_scale * reduced[reduced_index]
                            twist = twist + coefficient * basis
                    for component in qd.static(range(6)):
                        self.body_twist[body, component] = twist[component]

        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                mechanism = self.contact_proxy.mechanism_body[pair]
                proxy = self.contact_proxy.proxy_body[pair]
                tangent = qd.Matrix.zero(qd.f64, 6, 6)
                mechanism_twist = qd.Vector.zero(qd.f64, 6)
                for row in qd.static(range(6)):
                    mechanism_twist[row] = self.body_twist[mechanism, row]
                    for column in qd.static(range(6)):
                        tangent[row, column] = self.contact_proxy.tangent_map[pair, row, column]
                proxy_twist = expand_proxy_twist(tangent, mechanism_twist)
                for component in qd.static(range(6)):
                    self.body_twist[proxy, component] = proxy_twist[component]

    @qd.func(requires_top_level=True)
    def clear_body_wrench(self):
        for body in range(self.n_bodies[()]):
            for component in qd.static(range(6)):
                self.body_wrench[body, component] = 0.0

    @qd.func(requires_top_level=True)
    def restrict_proxy_wrenches(self):
        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                mechanism = self.contact_proxy.mechanism_body[pair]
                proxy = self.contact_proxy.proxy_body[pair]
                tangent = qd.Matrix.zero(qd.f64, 6, 6)
                proxy_wrench = qd.Vector.zero(qd.f64, 6)
                for row in qd.static(range(6)):
                    proxy_wrench[row] = self.body_wrench[proxy, row]
                    for column in qd.static(range(6)):
                        tangent[row, column] = self.contact_proxy.tangent_map[pair, row, column]
                mapped = restrict_proxy_wrench(tangent, proxy_wrench)
                for component in qd.static(range(6)):
                    qd.atomic_add(self.body_wrench[mechanism, component], mapped[component])

    @qd.func(requires_top_level=True)
    def restrict_body_wrenches(self, reduced: qd.template()):
        for reverse_level in qd.static(range(self.n_levels_host)):
            level = self.n_levels_host - 1 - reverse_level
            for body in range(self.n_mechanism_bodies[()]):
                if self.depth[body] == level:
                    environment = body // self.n_links_host
                    link = body - environment * self.n_links_host
                    wrench = qd.Vector.zero(qd.f64, 6)
                    for component in qd.static(range(6)):
                        wrench[component] = self.body_wrench[body, component]

                    link_index = (
                        [link, environment]
                        if qd.static(self.rigid.rigid_config.batch_links_info)
                        else link
                    )
                    for joint in range(
                        self.rigid.dyn_info.links.joint_start[link_index],
                        self.rigid.dyn_info.links.joint_end[link_index],
                    ):
                        joint_index = (
                            [joint, environment]
                            if qd.static(self.rigid.rigid_config.batch_joints_info)
                            else joint
                        )
                        for dof in range(
                            self.rigid.dyn_info.joints.dof_start[joint_index],
                            self.rigid.dyn_info.joints.dof_end[joint_index],
                        ):
                            basis = self._body_point_twist(link, environment, dof)
                            reduced_index = (
                                self.rigid.dof_offset[()]
                                + environment * self.rigid.n_dofs_per_instance[()]
                                + dof
                            )
                            qd.atomic_add(
                                reduced[reduced_index],
                                self.configuration_scale * basis.dot(wrench),
                            )

                    parent = self.parent_body[body]
                    if parent >= 0:
                        parent_link = parent % self.n_links_host
                        offset = (
                            self.rigid.dyn_state.links.i_pos[link, environment]
                            - self.rigid.dyn_state.links.i_pos[parent_link, environment]
                        )
                        force = qd.Vector([wrench[0], wrench[1], wrench[2]])
                        torque = qd.Vector([wrench[3], wrench[4], wrench[5]])
                        parent_torque = torque + offset.cross(force)
                        for axis in qd.static(range(3)):
                            qd.atomic_add(self.body_wrench[parent, axis], force[axis])
                            qd.atomic_add(self.body_wrench[parent, axis + 3], parent_torque[axis])

    @qd.func(requires_top_level=True)
    def prepare_particular(self):
        for dof in range(self.total_dof[()]):
            self.physical_p[dof] = 0.0
        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                for component in qd.static(range(6)):
                    self.physical_p[
                        self.proxy_dof_offset[()] + pair * 6 + component
                    ] = self.contact_proxy.particular[pair, component]

    @qd.func(requires_top_level=True)
    def particular_spmv(self, linear_system: qd.template()):
        for dof in range(self.total_dof[()]):
            self.physical_Ap[dof] = 0.0
        linear_system.spmv(self.physical_p, self.physical_Ap)

    @qd.func(requires_top_level=True)
    def project_physical_rhs(self, linear_system: qd.template()):
        for dof in range(self.proxy_dof_offset[()]):
            linear_system.b_rhs[dof] = linear_system.b_rhs[dof] + self.physical_Ap[dof]

        self.clear_body_wrench()
        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                proxy = self.contact_proxy.proxy_body[pair]
                offset = self.proxy_dof_offset[()] + pair * 6
                proxy_wrench = qd.Vector.zero(qd.f64, 6)
                normal = qd.Matrix.zero(qd.f64, 6, 6)
                for row in qd.static(range(6)):
                    proxy_wrench[row] = (
                        linear_system.b_rhs[offset + row] + self.physical_Ap[offset + row]
                    )
                    self.body_wrench[proxy, row] = proxy_wrench[row]
                    for column in qd.static(range(6)):
                        normal[row, column] = self.contact_proxy.normal_map[pair, row, column]

                if self.contact_proxy.restoration_active[()] != 0:
                    slack_rhs = (
                        restrict_slack_wrench(normal, proxy_wrench)
                        + qd.Vector(
                            [
                                self.contact_proxy.lambda_[pair, 0],
                                self.contact_proxy.lambda_[pair, 1],
                                self.contact_proxy.lambda_[pair, 2],
                                self.contact_proxy.lambda_[pair, 3],
                                self.contact_proxy.lambda_[pair, 4],
                                self.contact_proxy.lambda_[pair, 5],
                            ]
                        )
                    )
                    for component in qd.static(range(6)):
                        linear_system.b_rhs[offset + component] = slack_rhs[component]
                else:
                    for component in qd.static(range(6)):
                        linear_system.b_rhs[offset + component] = 0.0
        self.restrict_proxy_wrenches()
        self.restrict_body_wrenches(linear_system.b_rhs)

    @qd.func(requires_top_level=True)
    def prepare_physical_direction(self, reduced: qd.template()):
        for dof in range(self.total_dof[()]):
            self.physical_p[dof] = reduced[dof]
        self.expand_reduced_direction(reduced)
        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                proxy = self.contact_proxy.proxy_body[pair]
                value = qd.Vector.zero(qd.f64, 6)
                normal = qd.Matrix.zero(qd.f64, 6, 6)
                slack = qd.Vector.zero(qd.f64, 6)
                offset = self.proxy_dof_offset[()] + pair * 6
                for row in qd.static(range(6)):
                    value[row] = self.body_twist[proxy, row]
                    slack[row] = reduced[offset + row]
                    for column in qd.static(range(6)):
                        normal[row, column] = self.contact_proxy.normal_map[pair, row, column]
                if self.contact_proxy.restoration_active[()] != 0:
                    value = value + expand_slack_twist(normal, slack)
                for component in qd.static(range(6)):
                    self.physical_p[offset + component] = value[component]

    @qd.func(requires_top_level=True)
    def finish_reduced_spmv(
        self,
        reduced: qd.template(),
        result: qd.template(),
    ):
        for dof in range(self.proxy_dof_offset[()]):
            result[dof] = result[dof] + self.physical_Ap[dof]

        self.clear_body_wrench()
        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                proxy = self.contact_proxy.proxy_body[pair]
                offset = self.proxy_dof_offset[()] + pair * 6
                proxy_wrench = qd.Vector.zero(qd.f64, 6)
                normal = qd.Matrix.zero(qd.f64, 6, 6)
                metric = qd.Matrix.zero(qd.f64, 6, 6)
                slack = qd.Vector.zero(qd.f64, 6)
                for row in qd.static(range(6)):
                    proxy_wrench[row] = self.physical_Ap[offset + row]
                    self.body_wrench[proxy, row] = proxy_wrench[row]
                    slack[row] = reduced[offset + row]
                    for column in qd.static(range(6)):
                        normal[row, column] = self.contact_proxy.normal_map[pair, row, column]
                        metric[row, column] = self.contact_proxy.metric[pair, row, column]
                if self.contact_proxy.restoration_active[()] != 0:
                    slack_result = (
                        restrict_slack_wrench(normal, proxy_wrench) + metric @ slack
                    )
                    for component in qd.static(range(6)):
                        result[offset + component] = (
                            result[offset + component] + slack_result[component]
                        )
                else:
                    for component in qd.static(range(6)):
                        result[offset + component] = (
                            result[offset + component] + reduced[offset + component]
                        )
        self.restrict_proxy_wrenches()
        self.restrict_body_wrenches(result)

    @qd.func(requires_top_level=True)
    def expand_solution(self, solution: qd.template()):
        self.expand_reduced_direction(solution)
        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                proxy = self.contact_proxy.proxy_body[pair]
                offset = self.proxy_dof_offset[()] + pair * 6
                normal = qd.Matrix.zero(qd.f64, 6, 6)
                slack = qd.Vector.zero(qd.f64, 6)
                for row in qd.static(range(6)):
                    slack[row] = solution[offset + row]
                    for column in qd.static(range(6)):
                        normal[row, column] = self.contact_proxy.normal_map[pair, row, column]
                value = qd.Vector.zero(qd.f64, 6)
                for component in qd.static(range(6)):
                    value[component] = (
                        self.contact_proxy.particular[pair, component]
                        - self.body_twist[proxy, component]
                    )
                if self.contact_proxy.restoration_active[()] != 0:
                    value = value - expand_slack_twist(normal, slack)
                for component in qd.static(range(6)):
                    self.contact_proxy.dq[pair, component] = value[component]

    @qd.func(requires_top_level=True)
    def apply_proxy_preconditioner(
        self,
        residual: qd.template(),
        result: qd.template(),
    ):
        if qd.static(self.has_contact_proxy):
            for pair in range(self.contact_proxy.n_pairs[()]):
                offset = self.proxy_dof_offset[()] + pair * 6
                for component in qd.static(range(6)):
                    if self.contact_proxy.restoration_active[()] == 0:
                        result[offset + component] = residual[offset + component]
