import numpy as np
import pytest
import quadrants as qd
from quadrants.lang import impl

import genesis as gs
from genesis.engine.systems import (
    GlobalLayout,
    GlobalLinearSystem,
    LinearPCG,
    PhysicsSystem,
    RigidSystem,
    SimConfig,
    SimEngine,
)
from genesis.utils.misc import qd_to_numpy

from ..utils.assertions import assert_allclose


@qd.data_oriented
class QuadraticSystem(PhysicsSystem):
    def __init__(self, dof_range):
        super().__init__()
        self.scalar_offset = dof_range.scalar_offset
        self.n_dofs = dof_range.n_dofs
        self.n_storage_dofs = dof_range.n_storage_dofs
        self.state = qd.ndarray(qd.f64, shape=(self.n_dofs,))
        self.target = qd.ndarray(qd.f64, shape=(self.n_dofs,))
        self.diagonal = qd.ndarray(qd.f64, shape=(self.n_dofs,))
        self.direction = qd.ndarray(qd.f64, shape=(self.n_dofs,))

    def build(self):
        self.state.from_numpy(np.zeros(self.n_dofs))
        self.target.from_numpy(np.array([1.0, -2.0, 0.5]))
        self.diagonal.from_numpy(np.array([2.0, 3.0, 4.0]))

    @qd.func(requires_top_level=True)
    def predict(self):
        for _ in qd.static(range(0)):
            self.state[0] = qd.f64(0.0)

    @qd.func(requires_top_level=True)
    def assemble_candidate_rows(self):
        for _ in qd.static(range(0)):
            self.state[0] = qd.f64(0.0)

    @qd.func(requires_top_level=True)
    def initialize_newton(self):
        for _ in qd.static(range(0)):
            self.state[0] = qd.f64(0.0)

    @qd.func(requires_top_level=True)
    def add_current_energy(self, energy: qd.template()):
        for i_d in range(self.n_dofs):
            difference = self.state[i_d] - self.target[i_d]
            qd.atomic_add(energy[()], 0.5 * self.diagonal[i_d] * difference * difference)

    @qd.func(requires_top_level=True)
    def assemble_gradient(self, linear_system: qd.template(), gradient_squared: qd.template()):
        for i_d in range(self.n_dofs):
            gradient = self.diagonal[i_d] * (self.state[i_d] - self.target[i_d])
            linear_system.rhs[self.scalar_offset + i_d] = gradient
            qd.atomic_add(gradient_squared[()], gradient * gradient)
        for i_padding in range(self.n_dofs, self.n_storage_dofs):
            linear_system.rhs[self.scalar_offset + i_padding] = qd.f64(0.0)

    @qd.func(requires_top_level=True)
    def apply_hessian(self, x: qd.template(), y: qd.template()):
        for i_d in range(self.n_dofs):
            i_global = self.scalar_offset + i_d
            y[i_global] = y[i_global] + self.diagonal[i_d] * x[i_global]
        for i_padding in range(self.n_dofs, self.n_storage_dofs):
            i_global = self.scalar_offset + i_padding
            y[i_global] = y[i_global] + x[i_global]

    @qd.func(requires_top_level=True)
    def apply_preconditioner(self, residual: qd.template(), result: qd.template()):
        for i_d in range(self.n_dofs):
            i_global = self.scalar_offset + i_d
            result[i_global] = residual[i_global] / self.diagonal[i_d]
        for i_padding in range(self.n_dofs, self.n_storage_dofs):
            i_global = self.scalar_offset + i_padding
            result[i_global] = residual[i_global]

    @qd.func(requires_top_level=True)
    def prepare_direction(self, direction: qd.template()):
        for i_d in range(self.n_dofs):
            self.direction[i_d] = direction[self.scalar_offset + i_d]

    @qd.func(requires_top_level=True)
    def evaluate_energy_delta(self, alpha, energy_delta: qd.template()):
        for i_d in range(self.n_dofs):
            difference = self.state[i_d] - self.target[i_d]
            gradient = self.diagonal[i_d] * difference
            qd.atomic_add(
                energy_delta[()],
                alpha * gradient * self.direction[i_d]
                + 0.5 * alpha * alpha * self.diagonal[i_d] * self.direction[i_d] * self.direction[i_d],
            )

    @qd.func(requires_top_level=True)
    def accept(self, alpha):
        for i_d in range(self.n_dofs):
            self.state[i_d] = self.state[i_d] + alpha * self.direction[i_d]

    @qd.func(requires_top_level=True)
    def set_newton_active(self, is_active):
        for _ in qd.static(range(0)):
            self.state[0] = qd.f64(is_active)

    @qd.func(requires_top_level=True)
    def build_preconditioner(self, compute_envelope: qd.template()):
        for _ in qd.static(range(0)):
            self.state[0] = qd.f64(compute_envelope)

    @qd.func(requires_top_level=True)
    def finalize(self):
        for _ in qd.static(range(0)):
            self.state[0] = qd.f64(0.0)


@qd.kernel
def kernel_apply_bcoo(
    linear_system: qd.template(),
    x: qd.types.ndarray(),
    y: qd.types.ndarray(),
):
    linear_system.finalize_bcoo()
    for i_d in range(linear_system.n_dofs_device[0]):
        y[i_d] = qd.f64(0.0)
    linear_system.apply_bcoo(x, y)


@pytest.mark.required
@pytest.mark.precision("64")
def test_global_layout_scalar_tail():
    layout = GlobalLayout()
    dof_range = layout.allocate(7)

    assert dof_range.scalar_offset == 0
    assert dof_range.block_offset == 0
    assert dof_range.n_blocks == 3
    assert dof_range.n_storage_dofs == 9
    assert layout.n_dofs == 9


@pytest.mark.required
@pytest.mark.precision("64")
def test_global_bcoo_spmv():
    linear_system = GlobalLinearSystem(n_block_rows=2, n_triplets=3)
    linear_system.build()
    linear_system.triplet_row.from_numpy(np.array([0, 0, 1], dtype=np.int32))
    linear_system.triplet_col.from_numpy(np.array([0, 1, 1], dtype=np.int32))
    blocks = np.array(
        [
            [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
            [[0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.4]],
            [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
        ],
        dtype=np.float64,
    )
    linear_system.triplet_value.from_numpy(blocks.reshape(27))
    x = qd.ndarray(qd.f64, shape=(6,))
    y = qd.ndarray(qd.f64, shape=(6,))
    x_host = np.arange(1.0, 7.0)
    x.from_numpy(x_host)

    kernel_apply_bcoo(linear_system, x, y)

    dense = np.zeros((6, 6))
    dense[:3, :3] = blocks[0]
    dense[:3, 3:] = blocks[1]
    dense[3:, :3] = blocks[1].T
    dense[3:, 3:] = blocks[2]
    assert_allclose(qd_to_numpy(y), dense @ x_host, rtol=1e-12, atol=1e-12)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("n_envs", [0, 2])
def test_global_newton_native_contact(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -1.0, 0.7),
            camera_lookat=(0.0, 0.0, 0.1),
            camera_fov=30,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.11),
            radius=0.1,
        ),
        vis_mode="collision",
    )
    scene.build(n_envs=n_envs)
    sphere.set_dofs_velocity([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])

    rigid_solver = scene.rigid_solver
    layout = GlobalLayout()
    rigid_range = layout.allocate(rigid_solver.n_dofs * rigid_solver._B)
    quadratic_range = layout.allocate(3)
    linear_system = GlobalLinearSystem(layout.n_block_rows)
    pcg = LinearPCG(layout.n_dofs)
    config = SimConfig(
        h=rigid_solver._substep_dt,
        max_newton=rigid_solver._options.iterations,
        max_pcg=rigid_solver._options.iterations,
        max_line_search=rigid_solver._options.ls_iterations,
        newton_tolerance=rigid_solver._options.tolerance,
        pcg_tolerance=rigid_solver._options.tolerance,
    )
    rigid_system = RigidSystem(rigid_solver, rigid_range)
    quadratic_system = QuadraticSystem(quadratic_range)
    engine = SimEngine()
    for system in (layout, config, linear_system, pcg, rigid_system, quadratic_system):
        engine.add_system(system)
    engine.initialize()

    for _ in range(5):
        engine.step()

    if gs.backend == gs.gpu:
        assert impl.get_runtime().prog.get_graph_cache_used_on_last_call()
        assert impl.get_runtime().prog.get_graph_num_nodes_on_last_call() > 0
    assert (sphere.get_pos()[..., 2] > 0.095).all()
    assert (sphere.get_dofs_velocity()[..., 2] > -0.05).all()
    assert_allclose(qd_to_numpy(quadratic_system.state), qd_to_numpy(quadratic_system.target), atol=1e-12)
    assert engine.n_pcg_iterations == 1
