import numpy as np
import pytest
import quadrants as qd
from quadrants.lang import impl

import genesis as gs
from genesis.engine.systems import (
    GlobalLinearSystem,
    build_rigid_engine,
    validate_rigid_dynamics_backend,
)
from genesis.engine.systems.builders import _resolve_system_config
from genesis.utils.misc import qd_to_numpy

from ..utils.assertions import assert_allclose


def test_rigid_dynamics_backend_validation():
    assert validate_rigid_dynamics_backend("genesis") == "genesis"
    assert validate_rigid_dynamics_backend("cgq_mincoo") == "cgq_mincoo"
    with pytest.raises(ValueError, match="rigid/dynamics_backend"):
        validate_rigid_dynamics_backend("unknown")


def test_linear_system_solver_validation():
    assert _resolve_system_config(None)["linear_system/solver"] == "partition_pcg"
    assert _resolve_system_config({"linear_system/solver": "linear_pcg"})["linear_system/solver"] == "linear_pcg"
    with pytest.raises(ValueError, match="linear_system/solver"):
        _resolve_system_config({"linear_system/solver": "unknown"})
    with pytest.raises(
        ValueError,
        match="linear_system/partition_sv_max_iter",
    ):
        _resolve_system_config({"linear_system/partition_sv_max_iter": 0})


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
@pytest.mark.parametrize("linear_solver", [None, "linear_pcg"])
def test_global_newton_native_contact(
    n_envs,
    linear_solver,
    show_viewer,
):
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

    config = None if linear_solver is None else {"linear_system/solver": linear_solver}
    engine = build_rigid_engine(scene.rigid_solver, config=config)
    assert engine.linear_solver_name == ("partition_pcg" if linear_solver is None else linear_solver)
    assert engine.rigid.dynamics_backend == "genesis"
    assert engine.rigid.has_collision
    for _ in range(5):
        engine.step()

    assert impl.get_runtime().prog.get_graph_cache_used_on_last_call()
    assert impl.get_runtime().prog.get_graph_num_nodes_on_last_call() > 0
    assert (sphere.get_pos()[..., 2] > 0.095).all()
    assert (sphere.get_dofs_velocity()[..., 2] > -0.05).all()
    assert engine.get_max_pcg_iters() == 1


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_cgq_mincoo_rigid_freefall(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -9.81),
        ),
        show_viewer=show_viewer,
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 1.0),
            quat=(0.7071067811865476, 0.0, 0.0, 0.7071067811865475),
            radius=0.1,
        ),
    )
    scene.build(n_envs=n_envs, compile_kernels=False)
    angular_velocity = np.array([0.3, -0.2, 0.1])
    sphere.set_dofs_velocity(np.concatenate((np.zeros(3), angular_velocity)))
    initial_height = sphere.get_pos()[..., 2].clone()

    engine = build_rigid_engine(
        scene.rigid_solver,
        config={"rigid/dynamics_backend": "cgq_mincoo"},
    )
    assert engine.rigid.dynamics_backend == "cgq_mincoo"
    assert engine.rigid_forest is not None

    engine.step()

    assert impl.get_runtime().prog.get_graph_cache_used_on_last_call()
    assert (sphere.get_pos()[..., 2] < initial_height).all()
    assert_allclose(sphere.get_dofs_velocity()[..., 3:], angular_velocity, atol=1e-8)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
@pytest.mark.parametrize(
    ("asset", "target"),
    [
        ("urdf/simple/two_cube_revolute.urdf", 0.5),
        ("urdf/simple/two_cube_prismatic.urdf", 0.1),
    ],
)
def test_cgq_mincoo_scalar_joint_position_control(asset, target, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    arm = scene.add_entity(
        morph=gs.morphs.URDF(
            file=asset,
            fixed=True,
            merge_fixed_links=False,
        ),
    )
    scene.build(compile_kernels=False)
    arm.set_dofs_kp([100.0])
    arm.set_dofs_kv([10.0])
    arm.control_dofs_position([target])
    initial_position = float(arm.get_dofs_position()[0])

    engine = build_rigid_engine(
        scene.rigid_solver,
        config={"rigid/dynamics_backend": "cgq_mincoo"},
    )
    engine.step()

    assert impl.get_runtime().prog.get_graph_cache_used_on_last_call()
    assert float(arm.get_dofs_position()[0]) > initial_position


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_cgq_mincoo_joint_limits_use_absolute_qpos(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    arm = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_revolute.urdf",
            fixed=True,
            merge_fixed_links=False,
        ),
    )
    scene.build(compile_kernels=False)

    neutral_qpos = np.full_like(qd_to_numpy(scene.rigid_solver.rigid_info.qpos0), 1.0)
    scene.rigid_solver.rigid_info.qpos0.from_numpy(neutral_qpos)
    arm.set_qpos([1.0])
    arm.set_dofs_limit([0.9], [1.1])

    engine = build_rigid_engine(
        scene.rigid_solver,
        config={"rigid/dynamics_backend": "cgq_mincoo"},
    )
    engine.step()

    assert_allclose(arm.get_qpos(), np.array([1.0]), atol=1e-10)
