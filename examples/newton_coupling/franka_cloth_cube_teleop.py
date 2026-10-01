"""Teleoperate Franka over a tabletop, cloth, and support cube.

Controls:
  Arrow keys    move the target in X/Y
  J/K           move down/up
  N/M           yaw left/right
  U/O           pitch up/down
  L/;           roll left/right
  Space         hold to close the gripper
  Backslash     reset the target pose
  Escape        quit
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.systems import ContactTabular, build_scene_engine
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin
from genesis.utils.misc import qd_to_numpy
from genesis.vis.keybindings import Key, KeyAction, Keybind

from cloth_grid_asset import cloth_grid_asset

DT = 0.01
TARGET_TRANSLATION_STEP = 0.003
TARGET_ROTATION_STEP = 0.02

TABLE_CENTER_X = 0.55
TABLE_TOP = 0.40
TABLE_SIZE = (0.75, 0.70, 0.06)

CUBE_SIZE = 0.08 * 0.6
CLEARANCE = 0.002
CUBE_CENTER = (
    0.55,
    0.0,
    TABLE_TOP + CLEARANCE + 0.5 * CUBE_SIZE,
)
CUBE_TOP = TABLE_TOP + CLEARANCE + CUBE_SIZE

CLOTH_SIZE = 0.40
CLOTH_RESOLUTION = 41
CLOTH_CENTER = (0.55, 0.0, CUBE_TOP + 0.015)
CLOTH_COLOR = (0.25, 0.45, 0.90, 1.0)

HOME_QPOS = np.array(
    [0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7854, 0.04, 0.04],
    dtype=np.float64,
)


def build_franka_cloth_cube_scene(*, show_viewer: bool):
    """Build the shared Franka/cloth scene without compiling coupling kernels."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_fem=False),
        viewer_options=gs.options.ViewerOptions(
            res=(1100, 720),
            camera_pos=(1.65, -1.35, 1.15),
            camera_lookat=(0.55, 0.0, 0.48),
            camera_fov=38,
            enable_gui=True,
        ),
        show_viewer=show_viewer,
    )

    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        vis_mode="collision",
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(TABLE_CENTER_X, 0.0, TABLE_TOP - 0.5 * TABLE_SIZE[2]),
            size=TABLE_SIZE,
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=(0.42, 0.32, 0.24, 1.0)),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=CUBE_CENTER,
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=(0.75, 0.72, 0.62, 1.0)),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(
                cloth_grid_asset(
                    resolution=CLOTH_RESOLUTION,
                    size=CLOTH_SIZE,
                )
            ),
            pos=CLOTH_CENTER,
        ),
        material=gs.materials.FEM.QCloth(
            E=2e4,
            shear_modulus=2e3,
            rho=200.0,
            thickness=1e-3,
            bending_youngs_modulus=3e3,
        ),
        surface=gs.surfaces.Default(color=CLOTH_COLOR),
    )
    scene.build(compile_kernels=False)

    arm_dofs = np.arange(7, dtype=np.int32)
    finger_dofs = np.arange(7, 9, dtype=np.int32)
    franka.set_qpos(HOME_QPOS)
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]),
        dofs_idx_local=arm_dofs,
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200]),
        dofs_idx_local=arm_dofs,
    )
    franka.set_dofs_force_range(
        -400.0,
        400.0,
        dofs_idx_local=arm_dofs,
    )
    franka.set_dofs_kp(100.0, dofs_idx_local=finger_dofs)
    franka.set_dofs_kv(10.0, dofs_idx_local=finger_dofs)
    return scene, franka, arm_dofs, finger_dofs


def build_franka_cloth_cube_engine(scene, *, ee_query: str):
    """Build the coupling engine used by teleoperation and compile benchmarks."""
    contact_tabular = ContactTabular()
    contact_tabular.default_model(friction_rate=1.0, resistance=1e4)
    return build_scene_engine(
        scene,
        contact_config={
            "contact/d_hat": 1e-3,
            "contact/init_collision_pair_capacity": 20_000,
            "contact/intersection_check": 1,
            "bvh/ee_query": ee_query,
        },
        contact_tabular=contact_tabular,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headless for --steps frames",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--press-depth",
        type=float,
        default=0.0,
        help="Headless check: command the hand below the tabletop",
    )
    parser.add_argument(
        "--ee-query",
        choices=("dual", "warp"),
        default="dual",
    )
    parser.add_argument(
        "--advanced-optimization",
        action="store_true",
        help="Enable the production compiler policy (advanced optimization and LLVM O3)",
    )
    parser.add_argument(
        "--external-optimization-level",
        type=int,
        choices=range(4),
        default=None,
        help="Override LLVM optimization level; development defaults to O1 and production to O3",
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="64", logging_level="info")
    qd.cfg.advanced_optimization = args.advanced_optimization
    qd.cfg.external_optimization_level = (
        args.external_optimization_level
        if args.external_optimization_level is not None
        else (3 if args.advanced_optimization else 1)
    )
    scene, franka, arm_dofs, finger_dofs = build_franka_cloth_cube_scene(
        show_viewer=not args.no_gui,
    )
    engine = build_franka_cloth_cube_engine(scene, ee_query=args.ee_query)

    end_effector = franka.get_link("hand")
    target_home_pos = end_effector.get_pos().cpu().numpy().reshape(3)
    target_home_quat = end_effector.get_quat().cpu().numpy().reshape(4)
    target_pos = target_home_pos.copy()
    target_quat = target_home_quat.copy()
    press_target = target_home_pos.copy()
    press_quaternion = target_home_quat.copy()
    if args.no_gui and args.press_depth > 0.0:
        press_target[0] = 0.55
        press_target[1] = 0.0
        press_quaternion[:] = (0.0, 1.0, 0.0, 0.0)
    press_target[2] = TABLE_TOP - args.press_depth
    gripper_closed = np.array(False, dtype=bool)
    is_running = True

    target_frame = None
    if not args.no_gui:
        target_frame = scene.draw_debug_frame(
            T=gu.trans_quat_to_T(target_pos, target_quat),
            axis_length=0.12,
            origin_size=0.008,
            axis_radius=0.005,
        )

        def move(delta: tuple[float, float, float]) -> None:
            target_pos[:] += np.asarray(delta, dtype=np.float64)

        def rotate(axis: int, angle: float) -> None:
            delta = np.zeros(3, dtype=np.float64)
            delta[axis] = angle
            delta_quaternion = gu.xyz_to_quat(delta)
            target_quat[:] = gu.transform_quat_by_quat(
                target_quat,
                delta_quaternion,
            )

        def set_gripper(closed: bool) -> None:
            gripper_closed[()] = closed

        def reset_target() -> None:
            target_pos[:] = target_home_pos
            target_quat[:] = target_home_quat

        def stop() -> None:
            nonlocal is_running
            is_running = False

        scene.viewer.register_keybinds(
            Keybind(
                "move_forward",
                Key.UP,
                KeyAction.HOLD,
                callback=move,
                args=((TARGET_TRANSLATION_STEP, 0.0, 0.0),),
            ),
            Keybind(
                "move_backward",
                Key.DOWN,
                KeyAction.HOLD,
                callback=move,
                args=((-TARGET_TRANSLATION_STEP, 0.0, 0.0),),
            ),
            Keybind(
                "move_left",
                Key.LEFT,
                KeyAction.HOLD,
                callback=move,
                args=((0.0, TARGET_TRANSLATION_STEP, 0.0),),
            ),
            Keybind(
                "move_right",
                Key.RIGHT,
                KeyAction.HOLD,
                callback=move,
                args=((0.0, -TARGET_TRANSLATION_STEP, 0.0),),
            ),
            Keybind(
                "move_down",
                Key.J,
                KeyAction.HOLD,
                callback=move,
                args=((0.0, 0.0, -TARGET_TRANSLATION_STEP),),
            ),
            Keybind(
                "move_up",
                Key.K,
                KeyAction.HOLD,
                callback=move,
                args=((0.0, 0.0, TARGET_TRANSLATION_STEP),),
            ),
            Keybind(
                "yaw_left",
                Key.N,
                KeyAction.HOLD,
                callback=rotate,
                args=(2, TARGET_ROTATION_STEP),
            ),
            Keybind(
                "yaw_right",
                Key.M,
                KeyAction.HOLD,
                callback=rotate,
                args=(2, -TARGET_ROTATION_STEP),
            ),
            Keybind(
                "pitch_up",
                Key.U,
                KeyAction.HOLD,
                callback=rotate,
                args=(1, TARGET_ROTATION_STEP),
            ),
            Keybind(
                "pitch_down",
                Key.O,
                KeyAction.HOLD,
                callback=rotate,
                args=(1, -TARGET_ROTATION_STEP),
            ),
            Keybind(
                "roll_left",
                Key.L,
                KeyAction.HOLD,
                callback=rotate,
                args=(0, TARGET_ROTATION_STEP),
            ),
            Keybind(
                "roll_right",
                Key.SEMICOLON,
                KeyAction.HOLD,
                callback=rotate,
                args=(0, -TARGET_ROTATION_STEP),
            ),
            Keybind(
                "reset_target",
                Key.BACKSLASH,
                KeyAction.RELEASE,
                callback=reset_target,
            ),
            Keybind(
                "close_gripper",
                Key.SPACE,
                KeyAction.PRESS,
                callback=set_gripper,
                args=(True,),
            ),
            Keybind(
                "open_gripper",
                Key.SPACE,
                KeyAction.RELEASE,
                callback=set_gripper,
                args=(False,),
            ),
            Keybind(
                "quit",
                Key.ESCAPE,
                KeyAction.RELEASE,
                callback=stop,
            ),
            overwrite=True,
        )

    print(__doc__)
    frame = 0
    max_newton = 0
    max_pcg = 0
    max_line_search = 0
    max_ee_pairs = 0
    min_ccd_alpha = 1.0
    last_target_qpos = HOME_QPOS[:7].copy()
    performance = {
        "step_ms": 0.0,
        "step_ms_ema": 0.0,
        "newton": 0,
        "pcg": 0,
        "line_search": 0,
        "ccd_alpha": 1.0,
    }
    if not args.no_gui:
        overlay = next(
            (plugin for plugin in scene.viewer.plugins if isinstance(plugin, ImGuiOverlayPlugin)),
            None,
        )
        if overlay is not None:

            def draw_newton_performance(imgui) -> None:
                imgui.separator()
                imgui.text("Newton Coupling")
                imgui.text(f"Step: {performance['step_ms']:.2f} ms  EMA: {performance['step_ms_ema']:.2f} ms")
                simulation_fps = 1000.0 / performance["step_ms_ema"] if performance["step_ms_ema"] > 0.0 else 0.0
                imgui.text(f"Simulation FPS: {simulation_fps:.1f}")
                imgui.text(
                    f"Newton: {performance['newton']}  PCG: {performance['pcg']}  LS: {performance['line_search']}"
                )
                imgui.text(f"CCD alpha: {performance['ccd_alpha']:.6f}")

            overlay.register_panel(draw_newton_performance, section="side")
    try:
        while is_running and ((not args.no_gui and scene.viewer.is_alive()) or (args.no_gui and frame < args.steps)):
            if args.no_gui and args.press_depth > 0.0:
                target_pos[:] += np.clip(
                    press_target - target_pos,
                    -TARGET_TRANSLATION_STEP,
                    TARGET_TRANSLATION_STEP,
                )
                target_quat[:] = press_quaternion
            if target_frame is not None:
                scene.update_debug_objects(
                    (target_frame,),
                    (gu.trans_quat_to_T(target_pos, target_quat),),
                )

            target_qpos = franka.inverse_kinematics(
                link=end_effector,
                pos=target_pos,
                quat=target_quat,
                init_qpos=franka.get_qpos(),
                max_samples=8 if args.press_depth > 0.0 else 1,
                max_solver_iters=8,
                damping=0.05,
                max_step_size=0.1,
                dofs_idx_local=arm_dofs,
            )
            last_target_qpos = target_qpos[arm_dofs].cpu().numpy()
            franka.control_dofs_position(
                target_qpos[arm_dofs],
                dofs_idx_local=arm_dofs,
            )
            franka.control_dofs_position(
                0.0 if gripper_closed[()] else 0.04,
                dofs_idx_local=finger_dofs,
            )

            step_begin = time.perf_counter()
            engine.step()
            step_ms = (time.perf_counter() - step_begin) * 1000.0
            performance["step_ms"] = step_ms
            performance["step_ms_ema"] = (
                step_ms if performance["step_ms_ema"] == 0.0 else 0.9 * performance["step_ms_ema"] + 0.1 * step_ms
            )
            performance["newton"] = engine.get_newton_iters()
            performance["pcg"] = engine.get_max_pcg_iters()
            performance["line_search"] = engine.get_max_ls_iters()
            performance["ccd_alpha"] = float(qd_to_numpy(engine.contact.frame_ccd_alpha))
            max_newton = max(max_newton, engine.get_newton_iters())
            max_pcg = max(max_pcg, engine.get_max_pcg_iters())
            max_line_search = max(
                max_line_search,
                engine.get_max_ls_iters(),
            )
            max_ee_pairs = max(
                max_ee_pairs,
                int(qd_to_numpy(engine.contact.n_pairs_ee)),
            )
            min_ccd_alpha = min(
                min_ccd_alpha,
                float(qd_to_numpy(engine.contact.frame_ccd_alpha)),
            )
            if not args.no_gui:
                scene.viewer.update(force=True)
            frame += 1
            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        pass
    finally:
        qd.sync()
        print(
            "max iterations:",
            max_newton,
            max_pcg,
            max_line_search,
            f"max_ee_pairs={max_ee_pairs}",
            f"min_ccd_alpha={min_ccd_alpha:.9g}",
        )
        hand_position = end_effector.get_pos().cpu().numpy().reshape(3)
        print(
            f"target_pos={target_pos.tolist()}, "
            f"hand_pos={hand_position.tolist()}, "
            f"target_qpos={last_target_qpos.tolist()}"
        )
        if engine.contact.broad_phase.use_dual_ee:
            dual = engine.contact.broad_phase.ee_dual_state
            print(
                "dual state:",
                f"selected={int(qd_to_numpy(dual.selected_count))}",
                f"parity={int(qd_to_numpy(dual.selected_parity))}",
                f"level={int(qd_to_numpy(dual.current_level))}",
                f"next_task={int(qd_to_numpy(dual.next_task))}",
                f"overflow={int(qd_to_numpy(dual.overflow_bits))}",
            )


if __name__ == "__main__":
    main()
