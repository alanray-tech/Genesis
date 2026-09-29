import argparse
import time

import quadrants as qd

import genesis as gs
from genesis.engine.systems import build_rigid_engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("native", "newton"), default="newton")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    args = parser.parse_args()
    if args.runtime == "newton" and not args.gpu:
        parser.error("the graph-native newton runtime is GPU-only; pass --gpu")

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(960, 640),
            camera_pos=(3.0, -1.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        vis_mode="collision",
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.65, 0.0, 0.02),
            size=(0.04, 0.04, 0.04),
        ),
        vis_mode="collision",
    )
    scene.build(compile_kernels=args.runtime == "native")

    motors_dof = [0, 1, 2, 3, 4, 5, 6]
    fingers_dof = [7, 8]
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)
    franka.set_qpos([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])

    engine = None
    if args.runtime == "newton":
        engine = build_rigid_engine(scene.rigid_solver)
        step = engine.step
    else:
        step = scene.step
    refresh_viewer = args.vis and engine is not None
    qd.sync()
    time_start = time.perf_counter()
    step()
    if refresh_viewer:
        scene.viewer.update(force=True)
    qd.sync()
    first_step_seconds = time.perf_counter() - time_start

    end_effector = franka.get_link("hand")
    target = franka.inverse_kinematics(
        link=end_effector,
        pos=[0.65, 0.0, 0.135],
        quat=[0.0, 1.0, 0.0, 0.0],
    )
    franka.control_dofs_position(target[:-2], motors_dof)

    qd.sync()
    time_start = time.perf_counter()
    for _ in range(100):
        step()
        if refresh_viewer:
            scene.viewer.update(force=True)

    for _ in range(100):
        franka.control_dofs_position(target[:-2], motors_dof)
        franka.control_dofs_position(0.0, fingers_dof)
        step()
        if refresh_viewer:
            scene.viewer.update(force=True)

    target = franka.inverse_kinematics(
        link=end_effector,
        pos=[0.65, 0.0, 0.3],
        quat=[0.0, 1.0, 0.0, 0.0],
    )
    for _ in range(200):
        franka.control_dofs_position(target[:-2], motors_dof)
        franka.control_dofs_position(0.0, fingers_dof)
        step()
        if refresh_viewer:
            scene.viewer.update(force=True)
    qd.sync()
    step_seconds = time.perf_counter() - time_start

    print("cube position:", cube.get_pos())
    print("first step seconds:", first_step_seconds)
    print("steady step milliseconds:", 1000.0 * step_seconds / 400)
    if engine is not None:
        print(
            "iterations:",
            engine.get_newton_iters(),
            engine.get_max_pcg_iters(),
            engine.get_max_ls_iters(),
        )


if __name__ == "__main__":
    main()
