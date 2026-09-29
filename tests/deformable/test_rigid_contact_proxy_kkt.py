import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.systems.global_linear_system import GlobalLinearSystem
from genesis.engine.systems.global_vertex_manager import GlobalVertexManager
from genesis.engine.systems.rigid_contact_assemble import RigidContactAssemble
from genesis.engine.systems.rigid_contact_proxy import (
    RigidContactProxyGeometry,
    RigidContactProxySystem,
)
from genesis.engine.systems.rigid_contact_proxy_kkt import (
    fk_defect_prefix_cap,
    rigid_contact_proxy_constraint,
    rigid_contact_proxy_prepare_maps,
)
from genesis.engine.systems.rigid_joint_forest import RigidJointForestSystem
from genesis.engine.systems.rigid_system import RigidSystem
from genesis.utils.misc import qd_to_numpy


@qd.data_oriented
class ProxyContactFixture:
    def __init__(self):
        self.n_unique_doublets = qd.ndarray(qd.i32, shape=())
        self.n_unique_triplets = qd.ndarray(qd.i32, shape=())
        self.unique_doublet_vertices = qd.ndarray(qd.i32, shape=(2,))
        self.unique_doublet_gradients = qd.ndarray(qd.f64, shape=(2, 3))
        self.unique_triplet_rows = qd.ndarray(qd.i32, shape=(3,))
        self.unique_triplet_cols = qd.ndarray(qd.i32, shape=(3,))
        self.unique_triplet_values = qd.ndarray(qd.f64, shape=(3, 3, 3))


@qd.data_oriented
class ProxyFEMFixture:
    def __init__(self):
        self.dof_offset = qd.ndarray(qd.i32, shape=())
        self.is_fixed = qd.ndarray(qd.i32, shape=(1,))


@qd.data_oriented
class ProxyForestFixture:
    def __init__(self):
        self.proxy_dof_offset = qd.ndarray(qd.i32, shape=())


def _skew(value):
    x, y, z = value
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _quat_exp(rotation):
    angle = np.linalg.norm(rotation)
    if angle < 1.0e-12:
        return np.array([1.0, *(0.5 * rotation)], dtype=np.float64)
    half = 0.5 * angle
    return np.array([np.cos(half), *(np.sin(half) * rotation / angle)], dtype=np.float64)


def _quat_mul(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _left_jacobian(rotation):
    theta_sq = float(rotation @ rotation)
    hat = _skew(rotation)
    if theta_sq < 1.0e-10:
        return np.eye(3) + 0.5 * hat + (1.0 / 6.0) * hat @ hat
    theta = np.sqrt(theta_sq)
    return np.eye(3) + (1.0 - np.cos(theta)) / theta_sq * hat + (
        theta - np.sin(theta)
    ) / (theta_sq * theta) * (hat @ hat)


def _right_jacobian_inverse(rotation):
    theta_sq = float(rotation @ rotation)
    hat = _skew(-rotation)
    if theta_sq < 1.0e-10:
        coefficient = 1.0 / 12.0 + theta_sq / 720.0
    else:
        theta = np.sqrt(theta_sq)
        half = 0.5 * theta
        coefficient = (1.0 - half / np.tan(half)) / theta_sq
    return np.eye(3) - 0.5 * hat + coefficient * hat @ hat


@qd.kernel(fastcache=True)
def evaluate_proxy_maps(
    mechanism_position: qd.types.ndarray(qd.f64, ndim=1),
    mechanism_quaternion: qd.types.ndarray(qd.f64, ndim=1),
    proxy_position: qd.types.ndarray(qd.f64, ndim=1),
    proxy_quaternion: qd.types.ndarray(qd.f64, ndim=1),
    output: qd.types.ndarray(qd.f64, ndim=1),
):
    for _ in range(1):
        mechanism_t = qd.Vector.zero(qd.f64, 3)
        mechanism_q = qd.Vector.zero(qd.f64, 4)
        proxy_t = qd.Vector.zero(qd.f64, 3)
        proxy_q = qd.Vector.zero(qd.f64, 4)
        for axis in qd.static(range(3)):
            mechanism_t[axis] = mechanism_position[axis]
            proxy_t[axis] = proxy_position[axis]
        for axis in qd.static(range(4)):
            mechanism_q[axis] = mechanism_quaternion[axis]
            proxy_q[axis] = proxy_quaternion[axis]

        translation = qd.Vector.zero(qd.f64, 3)
        rotation = qd.Vector.zero(qd.f64, 3)
        rigid_contact_proxy_constraint(
            mechanism_t,
            mechanism_q,
            proxy_t,
            proxy_q,
            translation,
            rotation,
        )
        constraint = qd.Vector.zero(qd.f64, 6)
        tangent_map = qd.Matrix.zero(qd.f64, 6, 6)
        normal_map = qd.Matrix.zero(qd.f64, 6, 6)
        particular = qd.Vector.zero(qd.f64, 6)
        rigid_contact_proxy_prepare_maps(
            translation,
            rotation,
            constraint,
            tangent_map,
            normal_map,
            particular,
        )
        for row in qd.static(range(6)):
            output[row] = constraint[row]
            output[6 + row] = particular[row]
            for column in qd.static(range(6)):
                output[12 + row * 6 + column] = tangent_map[row, column]
                output[48 + row * 6 + column] = normal_map[row, column]


@qd.kernel(fastcache=True)
def evaluate_fk_prefix_caps(
    inputs: qd.types.ndarray(qd.f64, ndim=2),
    outputs: qd.types.ndarray(qd.f64, ndim=1),
):
    for index in range(inputs.shape[0]):
        outputs[index] = fk_defect_prefix_cap(
            inputs[index, 0],
            inputs[index, 1],
            inputs[index, 2],
            inputs[index, 3],
        )


@qd.kernel(fastcache=True)
def initialize_and_prepare_proxy(proxy: qd.template(), vertex: qd.template()):
    proxy.initialize_proxy_state()
    proxy.prepare_metric()
    proxy.prepare_constraint()
    proxy.initialize_global_vertices(vertex)


@qd.kernel(fastcache=True)
def publish_proxy_trajectory(proxy: qd.template(), vertex: qd.template()):
    proxy.publish_trajectory_end_positions(vertex)


@qd.kernel(fastcache=True)
def prepare_proxy_constraint(proxy: qd.template()):
    proxy.prepare_constraint()


@qd.kernel(fastcache=True)
def evaluate_forest_virtual_work(
    forest: qd.template(),
    proxy: qd.template(),
    reduced_direction: qd.types.ndarray(qd.f64, ndim=1),
    proxy_wrench: qd.types.ndarray(qd.f64, ndim=1),
    reduced_wrench: qd.types.ndarray(qd.f64, ndim=1),
):
    forest.expand_reduced_direction(reduced_direction)
    forest.clear_body_wrench()
    for component in range(6):
        forest.body_wrench[proxy.proxy_body[0], component] = proxy_wrench[component]
    forest.restrict_proxy_wrenches()
    for dof in range(reduced_wrench.shape[0]):
        reduced_wrench[dof] = 0.0
    forest.restrict_body_wrenches(reduced_wrench)


@qd.kernel(fastcache=True)
def apply_reduced_contact_operator(
    forest: qd.template(),
    linear_system: qd.template(),
    direction: qd.types.ndarray(qd.f64, ndim=1),
    result: qd.types.ndarray(qd.f64, ndim=1),
):
    for dof in range(forest.total_dof[()]):
        result[dof] = 0.0
        forest.physical_Ap[dof] = 0.0
    forest.prepare_physical_direction(direction)
    linear_system.spmv(forest.physical_p, forest.physical_Ap)
    forest.finish_reduced_spmv(direction, result)


@qd.kernel(fastcache=True)
def project_particular_rhs(
    forest: qd.template(),
    linear_system: qd.template(),
):
    for dof in range(forest.total_dof[()]):
        linear_system.b_rhs[dof] = 0.0
    forest.prepare_particular()
    forest.particular_spmv(linear_system)
    forest.project_physical_rhs(linear_system)


@qd.kernel(fastcache=True)
def prepare_physical_direction(
    forest: qd.template(),
    direction: qd.types.ndarray(qd.f64, ndim=1),
):
    forest.prepare_physical_direction(direction)


@qd.kernel(fastcache=True)
def assemble_proxy_contact(
    route: qd.template(),
    linear_system: qd.template(),
):
    route.classify()
    linear_system.derive_extents()
    linear_system.zero_rhs()
    linear_system.zero_triplet()
    route.distribute()
    linear_system.body_sort_reduce()


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_cgq_proxy_constraint_maps():
    mechanism_t_host = np.array([0.2, -0.1, 0.3], dtype=np.float64)
    mechanism_q_host = _quat_exp(np.array([0.2, -0.1, 0.15], dtype=np.float64))
    rotation = np.array([-0.12, 0.08, 0.05], dtype=np.float64)
    proxy_t_host = mechanism_t_host + np.array([0.03, 0.02, -0.03], dtype=np.float64)
    proxy_q_host = _quat_mul(_quat_exp(rotation), mechanism_q_host)

    mechanism_t = qd.ndarray(qd.f64, shape=(3,))
    mechanism_q = qd.ndarray(qd.f64, shape=(4,))
    proxy_t = qd.ndarray(qd.f64, shape=(3,))
    proxy_q = qd.ndarray(qd.f64, shape=(4,))
    output = qd.ndarray(qd.f64, shape=(84,))
    mechanism_t.from_numpy(mechanism_t_host)
    mechanism_q.from_numpy(mechanism_q_host)
    proxy_t.from_numpy(proxy_t_host)
    proxy_q.from_numpy(proxy_q_host)

    evaluate_proxy_maps(mechanism_t, mechanism_q, proxy_t, proxy_q, output)
    actual = qd_to_numpy(output)

    translation = proxy_t_host - mechanism_t_host
    left = _left_jacobian(rotation)
    expected_constraint = np.concatenate((translation, rotation))
    expected_particular = np.concatenate((-translation, -(left @ rotation)))
    expected_tangent = np.eye(6)
    expected_tangent[3:, 3:] = left @ _right_jacobian_inverse(rotation)
    expected_normal = np.eye(6)
    expected_normal[3:, 3:] = left
    np.testing.assert_allclose(actual[:6], expected_constraint, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual[6:12], expected_particular, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual[12:48].reshape(6, 6), expected_tangent, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual[48:84].reshape(6, 6), expected_normal, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_cgq_fk_defect_prefix_cap():
    cases = np.array(
        [
            [0.2, 0.1, 0.0, 0.2],
            [0.2, 0.5, 0.0, 0.35],
            [0.2, 0.1, 3.0, 0.2],
            [0.2, 0.5, 3.0, 0.2],
            [0.0, 0.0, 3.0, 0.4],
        ],
        dtype=np.float64,
    )
    inputs = qd.ndarray(qd.f64, shape=cases.shape)
    outputs = qd.ndarray(qd.f64, shape=(len(cases),))
    inputs.from_numpy(cases)

    evaluate_fk_prefix_caps(inputs, outputs)
    caps = qd_to_numpy(outputs)
    for cap, (h0, h_slack, curvature, limit) in zip(caps, cases, strict=True):
        alpha = np.linspace(0.0, cap, 129)
        bound = h0 + (h_slack - h0) * alpha + 0.5 * curvature * alpha * alpha
        assert np.all(bound <= limit + 1.0e-12)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_proxy_system_initializes_on_genesis_inertial_pose():
    scene = gs.Scene()
    scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.1, -0.2, 0.4), radius=0.1),
        vis_mode="collision",
    )
    scene.build(compile_kernels=False)
    geometry = RigidContactProxyGeometry()
    assert geometry.init(scene, 0.01)
    rigid = RigidSystem(scene.rigid_solver)
    proxy = RigidContactProxySystem()
    proxy.rigid = rigid
    proxy.n_links_host = rigid.dyn_state.links.pos.shape[0]
    proxy.n_instances_host = rigid.n_instances_host
    proxy.configure()
    proxy.wire_data(
        2,
        np.array([0], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([0.1], dtype=np.float64),
    )
    proxy.wire_geometry(0, geometry)
    vertex = GlobalVertexManager()
    vertex.init(len(geometry.local_positions))

    initialize_and_prepare_proxy(proxy, vertex)

    np.testing.assert_allclose(qd_to_numpy(proxy.constraint)[0], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(qd_to_numpy(proxy.particular)[0], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(qd_to_numpy(proxy.tangent_map)[0], np.eye(6), atol=1.0e-14)
    np.testing.assert_allclose(qd_to_numpy(proxy.normal_map)[0], np.eye(6), atol=1.0e-14)
    assert np.linalg.eigvalsh(qd_to_numpy(proxy.metric)[0]).min() > 0.0
    assert np.isfinite(qd_to_numpy(vertex.positions)).all()

    direction = np.zeros((1, 6), dtype=np.float64)
    direction[0, 0] = 0.02
    direction[0, 5] = 0.1
    proxy.dq.from_numpy(direction)
    publish_proxy_trajectory(proxy, vertex)
    assert np.linalg.norm(
        qd_to_numpy(vertex.trajectory_end_positions) - qd_to_numpy(vertex.positions)
    ) > 0.0
    assert qd_to_numpy(vertex.path_inflation).max() > 0.0
    np.testing.assert_array_equal(qd_to_numpy(vertex.path_kind), 0)

    forest = RigidJointForestSystem(scene.rigid_solver)
    forest.rigid = rigid
    forest.contact_proxy = proxy
    forest.has_contact_proxy = True
    rigid.init(0)
    total_dof = rigid.storage_dof_count_host + 6
    forest.init(total_dof, 2, rigid.storage_dof_count_host)
    reduced_direction = qd.ndarray(qd.f64, shape=(total_dof,))
    proxy_wrench = qd.ndarray(qd.f64, shape=(6,))
    reduced_wrench = qd.ndarray(qd.f64, shape=(total_dof,))
    direction_host = np.linspace(-0.3, 0.4, total_dof, dtype=np.float64)
    wrench_host = np.array([0.7, -0.2, 0.5, -0.1, 0.4, 0.3], dtype=np.float64)
    reduced_direction.from_numpy(direction_host)
    proxy_wrench.from_numpy(wrench_host)
    evaluate_forest_virtual_work(
        forest,
        proxy,
        reduced_direction,
        proxy_wrench,
        reduced_wrench,
    )
    proxy_twist = qd_to_numpy(forest.body_twist)[1]
    np.testing.assert_allclose(
        direction_host @ qd_to_numpy(reduced_wrench),
        proxy_twist @ wrench_host,
        rtol=1.0e-11,
        atol=1.0e-12,
    )

    linear_system = GlobalLinearSystem()
    linear_system.do_build()
    linear_system.init(
        n_block_rows=4,
        n_elastic_triplets=3,
        max_contact_body_triplets=0,
        dof_block_base=2,
        pcg_tol_rate=1.0e-4,
    )
    linear_system.bcoo_nnz.from_numpy(np.array(3, dtype=np.int32))
    linear_system.bcoo_row.from_numpy(np.array([2, 2, 3], dtype=np.int32))
    linear_system.bcoo_col.from_numpy(np.array([2, 3, 3], dtype=np.int32))
    blocks = np.array(
        [
            [[4.0, 0.2, 0.0], [0.2, 5.0, 0.1], [0.0, 0.1, 6.0]],
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            [[3.0, 0.0, 0.0], [0.0, 4.0, 0.2], [0.0, 0.2, 5.0]],
        ],
        dtype=np.float64,
    )
    linear_system.bcoo_val.from_numpy(
        np.pad(blocks.reshape(-1), (0, linear_system.bcoo_val.shape[0] - blocks.size))
    )
    second = qd.ndarray(qd.f64, shape=(total_dof,))
    first_result = qd.ndarray(qd.f64, shape=(total_dof,))
    second_result = qd.ndarray(qd.f64, shape=(total_dof,))
    second_host = np.linspace(0.5, -0.2, total_dof, dtype=np.float64)
    second.from_numpy(second_host)
    apply_reduced_contact_operator(forest, linear_system, reduced_direction, first_result)
    apply_reduced_contact_operator(forest, linear_system, second, second_result)
    first_result_host = qd_to_numpy(first_result)
    second_result_host = qd_to_numpy(second_result)
    np.testing.assert_allclose(
        direction_host @ second_result_host,
        second_host @ first_result_host,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert direction_host @ first_result_host > 0.0

    perturbed = qd_to_numpy(proxy.t).copy()
    perturbed[0] += np.array([0.01, -0.005, 0.002], dtype=np.float64)
    proxy.t.from_numpy(perturbed)
    prepare_proxy_constraint(proxy)
    assert np.linalg.norm(qd_to_numpy(proxy.particular)[0]) > 0.0
    project_particular_rhs(forest, linear_system)
    reduced_rhs = qd_to_numpy(linear_system.b_rhs)
    h_particular = qd_to_numpy(forest.physical_Ap).copy()
    prepare_physical_direction(forest, reduced_direction)
    physical_direction = qd_to_numpy(forest.physical_p)
    np.testing.assert_allclose(
        direction_host @ reduced_rhs,
        physical_direction @ h_particular,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        reduced_rhs[rigid.storage_dof_count_host :],
        0.0,
        atol=1.0e-14,
    )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_proxy_contact_routes_emit_cgq_block_counts():
    contact = ProxyContactFixture()
    contact.n_unique_doublets.from_numpy(np.array(2, dtype=np.int32))
    contact.n_unique_triplets.from_numpy(np.array(3, dtype=np.int32))
    contact.unique_doublet_vertices.from_numpy(np.array([1, 2], dtype=np.int32))
    contact.unique_doublet_gradients.from_numpy(
        np.array([[0.3, -0.2, 0.4], [-0.1, 0.5, 0.2]], dtype=np.float64)
    )
    contact.unique_triplet_rows.from_numpy(np.array([0, 0, 1], dtype=np.int32))
    contact.unique_triplet_cols.from_numpy(np.array([0, 1, 2], dtype=np.int32))
    contact.unique_triplet_values.from_numpy(
        np.array(
            [
                np.diag([2.0, 3.0, 4.0]),
                [[0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.4]],
                [[1.0, 0.1, 0.0], [0.1, 1.2, 0.2], [0.0, 0.2, 1.4]],
            ],
            dtype=np.float64,
        )
    )

    fem = ProxyFEMFixture()
    fem.dof_offset.from_numpy(np.array(0, dtype=np.int32))
    fem.is_fixed.from_numpy(np.array([0], dtype=np.int32))
    forest = ProxyForestFixture()
    forest.proxy_dof_offset.from_numpy(np.array(3, dtype=np.int32))

    proxy = RigidContactProxySystem()
    proxy.n_links_host = 1
    proxy.n_instances_host = 1
    proxy.configure()
    proxy.wire_data(
        2,
        np.array([0], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([1.0], dtype=np.float64),
    )
    geometry = RigidContactProxyGeometry()
    geometry.local_positions = np.array(
        [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
        dtype=np.float64,
    )
    geometry.vertex_pair = np.array([0, 0], dtype=np.int32)
    proxy.wire_geometry(1, geometry)
    proxy.t.from_numpy(np.zeros((1, 3), dtype=np.float64))

    vertex = GlobalVertexManager()
    vertex.init(3)
    vertex.positions.from_numpy(
        np.array(
            [[-0.2, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            dtype=np.float64,
        )
    )

    linear_system = GlobalLinearSystem()
    extent_slot = linear_system.register_extent_slot()
    linear_system.init(
        n_block_rows=3,
        n_elastic_triplets=0,
        max_contact_body_triplets=9,
        dof_block_base=0,
        pcg_tol_rate=1.0e-4,
    )
    route = RigidContactAssemble()
    route.contact = contact
    route.fem = fem
    route.vertex = vertex
    route.linear_system = linear_system
    route.proxy = proxy
    route.forest = forest
    route.extent_slot = extent_slot
    route.init()

    assemble_proxy_contact(route, linear_system)

    np.testing.assert_array_equal(qd_to_numpy(linear_system.n_triplets), 9)
    assert int(qd_to_numpy(linear_system.bcoo_valid)) == 1
    assert int(qd_to_numpy(linear_system.bcoo_nnz)) <= 9
    rhs = qd_to_numpy(linear_system.b_rhs)
    gradient = np.array([0.2, 0.3, 0.6])
    angular = np.cross(np.array([0.5, 0.0, 0.0]), np.array([0.3, -0.2, 0.4]))
    angular += np.cross(np.array([0.0, 0.5, 0.0]), np.array([-0.1, 0.5, 0.2]))
    np.testing.assert_allclose(rhs[3:6], gradient, atol=1.0e-12)
    np.testing.assert_allclose(rhs[6:9], angular, atol=1.0e-12)
