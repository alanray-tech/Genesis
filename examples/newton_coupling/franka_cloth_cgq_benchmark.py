"""Matched headless benchmark for CGQ rigid_franka_cloth_grasp."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.systems import ContactTabular, build_scene_engine
from genesis.utils.misc import qd_to_numpy

from cloth_grid_asset import cloth_grid_asset

DT = 0.01
GROUND_HEIGHT = 0.40
CLEARANCE = 0.002
TABLE_TOP = 0.64
TABLE_SIZE = (
    0.36,
    0.66,
    TABLE_TOP - GROUND_HEIGHT - CLEARANCE,
)
TABLE_CENTER = (
    0.45,
    0.0,
    GROUND_HEIGHT + CLEARANCE + 0.5 * TABLE_SIZE[2],
)
CUBE_SIZE = 0.05
CUBE_CENTER = (
    0.52,
    0.0,
    TABLE_TOP + CLEARANCE + 0.5 * CUBE_SIZE,
)
CUBE_TOP = TABLE_TOP + CLEARANCE + CUBE_SIZE
CLOTH_SIZE = 0.24
CLOTH_RESOLUTION = 25
CLOTH_CENTER = (0.45, 0.0, CUBE_TOP + 0.005)
HOME_QPOS = np.array(
    [0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7854, 0.04, 0.04],
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


def profile_window_mark(frame: int) -> None:
    spec = os.environ.get("QIPC_PROFILE_WINDOW")
    if not spec:
        return
    lower, upper = (int(value) for value in spec.split(":"))
    if frame in (lower, upper):
        import torch

        torch.cuda.synchronize()
        runtime = torch.cuda.cudart()
        if frame == lower:
            runtime.cudaProfilerStart()
        else:
            runtime.cudaProfilerStop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, default=CGQ_FRANKA_MJCF)
    parser.add_argument(
        "--forest-path",
        choices=("genesis_legacy", "cgq_level", "cgq_tree"),
        default="cgq_tree",
    )
    args = parser.parse_args()
    mjcf_path = args.mjcf.resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"CGQ Franka MJCF not found: {mjcf_path}")

    process_start = time.perf_counter()
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_fem=False),
        show_viewer=False,
    )
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=str(mjcf_path),
            pos=(0.0, 0.0, GROUND_HEIGHT + 0.01),
            merge_fixed_links=True,
            convexify=False,
            decimate=False,
            watertighten=None,
            default_armature=None,
        ),
        vis_mode="collision",
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=TABLE_CENTER,
            size=TABLE_SIZE,
            fixed=True,
        ),
        vis_mode="collision",
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=CUBE_CENTER,
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            fixed=True,
        ),
        vis_mode="collision",
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
    )
    scene.build(compile_kernels=False)
    franka.set_qpos(HOME_QPOS)
    franka.set_dofs_force_range(
        -400.0,
        400.0,
        dofs_idx_local=np.arange(7, dtype=np.int32),
    )
    franka.control_dofs_position(HOME_QPOS)

    contact_tabular = ContactTabular()
    contact_tabular.default_model(friction_rate=1.0, resistance=1e4)
    build_start = time.perf_counter()
    engine = build_scene_engine(
        scene,
        contact_config={
            "contact/d_hat": 1e-3,
            "contact/init_collision_pair_capacity": 20_000,
            "contact/intersection_check": 0,
            "linear_system/tol_rate": 1e-5,
            "rigid_forest/fused": int(args.forest_path == "cgq_tree"),
            "extras/rigid_forest/genesis_legacy": int(args.forest_path == "genesis_legacy"),
        },
        contact_tabular=contact_tabular,
        halfplanes=(
            np.array([[0.0, 0.0, GROUND_HEIGHT]], dtype=np.float64),
            np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        ),
    )
    build_seconds = time.perf_counter() - build_start

    warmup_ms = []
    for frame in range(args.warmup):
        profile_window_mark(frame)
        start = time.perf_counter()
        engine.step()
        qd.sync()
        warmup_ms.append((time.perf_counter() - start) * 1000.0)

    samples_ms = []
    newton = []
    pcg = []
    total_pcg = []
    line_search = []
    ccd_alpha = []
    accepted_alpha = []
    max_disp = []
    proxy_residual = []
    contact_info = []
    for frame in range(args.frames):
        profile_window_mark(args.warmup + frame)
        start = time.perf_counter()
        engine.step()
        qd.sync()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
        newton.append(engine.get_newton_iters())
        pcg.append(engine.get_max_pcg_iters())
        total_pcg.append(engine.get_total_pcg_iters())
        line_search.append(engine.get_max_ls_iters())
        ccd_alpha.append(float(qd_to_numpy(engine.contact.ccd_alpha)))
        accepted_alpha.append(float(qd_to_numpy(engine.alpha)))
        max_disp.append(float(qd_to_numpy(engine.max_disp)))
        proxy_residual.append(
            float(
                qd_to_numpy(
                    engine.rigid_contact_proxy.max_surface_residual
                )
            )
        )
        contact_info.append(
            [
                int(qd_to_numpy(engine.contact.n_pairs_pt)),
                int(qd_to_numpy(engine.contact.n_pairs_ee)),
                int(qd_to_numpy(engine.contact.n_active_pairs)),
            ]
        )

    result = {
        "implementation": "GenesisWorld",
        "warmup_frames": args.warmup,
        "measured_frames": args.frames,
        "warmup_ms": warmup_ms,
        "samples_ms": samples_ms,
        "newton": newton,
        "pcg": pcg,
        "total_pcg": total_pcg,
        "line_search": line_search,
        "ccd_alpha": ccd_alpha,
        "accepted_alpha": accepted_alpha,
        "max_disp": max_disp,
        "proxy_residual": proxy_residual,
        "contact_info": contact_info,
        "build_seconds": build_seconds,
        "process_seconds": time.perf_counter() - process_start,
        "dt": DT,
        "cloth_resolution": CLOTH_RESOLUTION,
        "cloth_size": CLOTH_SIZE,
        "contact_d_hat": 1e-3,
        "linear_tolerance_rate": 1e-5,
        "requested_forest_path": args.forest_path,
        "selected_forest_path": engine.rigid_forest.selected_path,
        "franka_mjcf": str(mjcf_path),
        "franka_n_links": franka.n_links,
        "franka_n_dofs": franka.n_dofs,
        "franka_n_geoms": franka.n_geoms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[benchmark] wrote {args.output}")
    print(
        "[benchmark] measured "
        f"median={np.median(samples_ms):.3f}ms "
        f"mean={np.mean(samples_ms):.3f}ms "
        f"p95={np.percentile(samples_ms, 95):.3f}ms"
    )


if __name__ == "__main__":
    main()
