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
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.systems import ContactTabular, build_scene_engine
from genesis.vis.keybindings import Key, KeyAction, Keybind

DT = 0.01
TARGET_TRANSLATION_STEP = 0.003
TARGET_ROTATION_STEP = 0.02

TABLE_CENTER_X = 0.72
TABLE_TOP = 0.40
TABLE_SIZE = (0.75, 0.70, 0.06)

CUBE_SIZE = 0.08
CLEARANCE = 0.002
CUBE_CENTER = (
    0.85,
    0.0,
    TABLE_TOP + CLEARANCE + 0.5 * CUBE_SIZE,
)
CUBE_TOP = TABLE_TOP + CLEARANCE + CUBE_SIZE

CLOTH_CENTER = (0.85, 0.0, CUBE_TOP + 0.015)
CLOTH_COLOR = (0.25, 0.45, 0.90, 1.0)

HOME_QPOS = np.array(
    [0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7854, 0.04, 0.04],
    dtype=np.float64,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headless for --steps frames",
    )
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="64", logging_level="info")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_fem=False),
        viewer_options=gs.options.ViewerOptions(
            res=(1100, 720),
            camera_pos=(1.65, -1.35, 1.15),
            camera_lookat=(0.50, 0.0, 0.48),
            camera_fov=38,
        ),
        show_viewer=not args.no_gui,
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
            file=str(Path(__file__).parent / "assets" / "qcloth_grid.obj"),
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

    contact_tabular = ContactTabular()
    contact_tabular.default_model(friction_rate=1.0, resistance=1e4)
    engine = build_scene_engine(
        scene,
        contact_config={
            "contact/d_hat": 1e-3,
            "contact/init_collision_pair_capacity": 20_000,
            "contact/intersection_check": 1,
        },
        contact_tabular=contact_tabular,
    )

    end_effector = franka.get_link("hand")
    target_home_pos = end_effector.get_pos().cpu().numpy().reshape(3)
    target_home_quat = end_effector.get_quat().cpu().numpy().reshape(4)
    target_pos = target_home_pos.copy()
    target_quat = target_home_quat.copy()
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
    try:
        while (
            is_running
            and (
                (not args.no_gui and scene.viewer.is_alive())
                or (args.no_gui and frame < args.steps)
            )
        ):
            if target_frame is not None:
                scene.update_debug_objects(
                    (target_frame,),
                    (gu.trans_quat_to_T(target_pos, target_quat),),
                )

            target_qpos = franka.inverse_kinematics(
                link=end_effector,
                pos=target_pos,
                quat=target_quat,
                dofs_idx_local=arm_dofs,
            )
            franka.control_dofs_position(
                target_qpos[arm_dofs],
                dofs_idx_local=arm_dofs,
            )
            franka.control_dofs_position(
                0.0 if gripper_closed[()] else 0.04,
                dofs_idx_local=finger_dofs,
            )

            engine.step()
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
            "iterations:",
            engine.get_newton_iters(),
            engine.get_max_pcg_iters(),
            engine.get_max_ls_iters(),
        )


if __name__ == "__main__":
    main()
