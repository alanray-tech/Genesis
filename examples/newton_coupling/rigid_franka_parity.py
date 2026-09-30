"""Pure-rigid Genesis Franka trajectory for CGQ conformance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import genesis as gs

DT = 0.01
HOME_QPOS = np.array(
    [0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7854, 0.04, 0.04],
    dtype=np.float64,
)
CGQ_KP = np.array(
    [4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0],
    dtype=np.float64,
)
CGQ_KV = np.array(
    [450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0],
    dtype=np.float64,
)
CGQ_FRANKA_MJCF = (
    Path(__file__).resolve().parents[3]
    / "cuda-graph-qipc"
    / "assets"
    / "download"
    / "franka_panda_mjcf_v2"
    / "panda.xml"
)


def as_list(value) -> list[float]:
    return np.asarray(value.cpu(), dtype=np.float64).reshape(-1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, default=CGQ_FRANKA_MJCF)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(enable_collision=False),
        show_viewer=False,
    )
    robot = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=str(args.mjcf.resolve()),
            pos=(0.0, 0.0, 0.41),
            merge_fixed_links=True,
            convexify=False,
            decimate=False,
            watertighten=None,
            default_armature=None,
        ),
        vis_mode="collision",
    )
    scene.build(compile_kernels=False)
    robot.set_qpos(HOME_QPOS)
    robot.set_dofs_kp(CGQ_KP)
    robot.set_dofs_kv(CGQ_KV)
    robot.control_dofs_position(HOME_QPOS)

    q = [as_list(robot.get_qpos())]
    control_force = [as_list(robot.get_dofs_control_force())]
    internal_force = [as_list(robot.get_dofs_force())]
    for _ in range(args.frames):
        scene.step()
        q.append(as_list(robot.get_qpos()))
        control_force.append(as_list(robot.get_dofs_control_force()))
        internal_force.append(as_list(robot.get_dofs_force()))

    q_array = np.asarray(q, dtype=np.float64)
    velocity = np.diff(q_array, axis=0) / DT
    act_bias = np.stack(
        [np.asarray(component.cpu(), dtype=np.float64).reshape(-1) for component in robot.get_dofs_act_bias()],
        axis=1,
    )
    result = {
        "implementation": "GenesisRigidSolver",
        "dt": DT,
        "frames": args.frames,
        "q": q,
        "velocity": velocity.tolist(),
        "control_force": control_force,
        "internal_force": internal_force,
        "act_gain": as_list(robot.get_dofs_act_gain()),
        "act_bias": act_bias.tolist(),
        "n_links": robot.n_links,
        "n_dofs": robot.n_dofs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
