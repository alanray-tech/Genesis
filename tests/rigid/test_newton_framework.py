import numpy as np
import pytest
import quadrants as qd
from quadrants.lang import impl

import genesis as gs
from genesis.engine.systems import GlobalLinearSystem, build_rigid_engine
from genesis.utils.misc import qd_to_numpy

from ..utils.assertions import assert_allclose


@qd.kernel
def kernel_apply_bcoo(
    linear_system: qd.template(),
    x: qd.types.ndarray(),
    y: qd.types.ndarray(),
):
    linear_system.body_sort_reduce()
    for i_dof in range(linear_system.total_dof[()]):
        y[i_dof] = qd.f64(0.0)
    linear_system.spmv(x, y)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_global_bcoo_spmv():
    linear_system = GlobalLinearSystem()
    linear_system.do_build()
    linear_system.init(
        n_block_rows=2,
        n_elastic_triplets=3,
        max_contact_body_triplets=0,
        dof_block_base=0,
        pcg_tol_rate=1e-4,
    )
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
    linear_system.triplet_val.from_numpy(blocks.reshape(27))
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
@pytest.mark.parametrize("backend", [gs.gpu])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_global_newton_native_contact(n_envs, show_viewer):
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(morph=gs.morphs.Plane())
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.11),
            radius=0.1,
        ),
        vis_mode="collision",
    )
    scene.build(n_envs=n_envs, compile_kernels=False)
    sphere.set_dofs_velocity([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])

    engine = build_rigid_engine(scene.rigid_solver)
    for _ in range(5):
        engine.step()

    assert impl.get_runtime().prog.get_graph_cache_used_on_last_call()
    assert impl.get_runtime().prog.get_graph_num_nodes_on_last_call() > 0
    assert (sphere.get_pos()[..., 2] > 0.095).all()
    assert (sphere.get_dofs_velocity()[..., 2] > -0.05).all()
    assert engine.get_max_pcg_iters() == 1
