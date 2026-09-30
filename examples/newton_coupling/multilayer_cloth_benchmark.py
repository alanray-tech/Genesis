"""Matched Genesis/CGQ multi-layer cloth performance benchmark.

Run this file with the virtual environment belonging to the selected backend.
Both backends receive the same vertices, triangles, material values, solver
limits, halfplane, and contact parameters.  The scene deliberately contains
no rigid body so rigid integration cannot affect the comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

DT = 0.01
GRAVITY = (0.0, -9.8, 0.0)
D_HAT = 1.0e-3
CONTACT_RESISTANCE = 1.0e5
PAIR_CAPACITY = 500_000
YOUNGS_MODULUS = 60_000.0
SHEAR_MODULUS = 30_000.0
DENSITY = 200.0
THICKNESS = 1.0e-3
BENDING_YOUNGS_MODULUS = 1.0e6

LAYER_SPECS = (
    ("cloth_large", 80, 0.5, 0.10),
    ("cloth_mid", 40, 0.3, 0.14),
    ("cloth_small", 20, 0.2, 0.16),
    ("cloth_tiny", 10, 0.1, 0.18),
)


def make_cloth_grid(
    subdivisions: int,
    size: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.empty(
        ((subdivisions + 1) * (subdivisions + 1), 3),
        dtype=np.float64,
    )
    cursor = 0
    for row in range(subdivisions + 1):
        for column in range(subdivisions + 1):
            vertices[cursor] = (
                (column / subdivisions - 0.5) * size,
                height,
                (row / subdivisions - 0.5) * size,
            )
            cursor += 1

    triangles = np.empty((2 * subdivisions * subdivisions, 3), dtype=np.int32)
    cursor = 0
    row_width = subdivisions + 1
    for row in range(subdivisions):
        for column in range(subdivisions):
            lower_left = row * row_width + column
            lower_right = lower_left + 1
            upper_left = lower_left + row_width
            upper_right = upper_left + 1
            triangles[cursor] = (lower_left, lower_right, upper_left)
            triangles[cursor + 1] = (lower_right, upper_right, upper_left)
            cursor += 2
    return vertices, triangles


def write_obj(
    name: str,
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> Path:
    path = Path(tempfile.gettempdir()) / f"genesis_qipc_{name}.obj"
    lines = [f"v {x:.17g} {y:.17g} {z:.17g}" for x, y, z in vertices]
    lines.extend(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}" for a, b, c in triangles)
    contents = "\n".join(lines) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != contents:
        path.write_text(contents, encoding="utf-8")
    return path


def profile_window_mark(frame: int) -> None:
    spec = os.environ.get("QIPC_PROFILE_WINDOW")
    if not spec:
        return
    lower, upper = (int(value) for value in spec.split(":"))
    if frame not in (lower, upper):
        return
    import torch

    torch.cuda.synchronize()
    runtime = torch.cuda.cudart()
    if frame == lower:
        runtime.cudaProfilerStart()
        print(f"[profile] cudaProfilerStart at frame {frame}", flush=True)
    else:
        runtime.cudaProfilerStop()
        print(f"[profile] cudaProfilerStop at frame {frame}", flush=True)


def build_cgq(disable_contact: bool):
    import torch
    from qipc import Cloth, Scene, trimesh
    from qipc.geometry import ground

    scene = Scene(
        dt=DT,
        gravity=GRAVITY,
        **{
            "contact/enable": int(not disable_contact),
            "contact/d_hat": D_HAT,
            "contact/init_collision_pair_capacity": PAIR_CAPACITY,
            "contact/ccd_partition": 0,
            "contact/intersection_check": 0,
            "newton/velocity_tol": 5.0e-2,
            "newton/max_iter": 1024,
            "linear_system/solver": "linear_pcg",
            "linear_system/preconditioner": "diag",
            "linear_system/tol_rate": 1.0e-4,
            "line_search/max_iter": 12,
            "friction/eps_v": 1.0e-2,
        },
    )
    scene.contact_tabular.default_model(
        friction_rate=0.0,
        resistance=CONTACT_RESISTANCE,
    )
    scene.geometries.create("ground", ground(height=0.0, N=(0.0, 1.0, 0.0)))
    for name, subdivisions, size, height in LAYER_SPECS:
        vertices, triangles = make_cloth_grid(subdivisions, size, height)
        geometry = trimesh(vertices, triangles)
        Cloth().apply_to(
            geometry,
            youngs_modulus=YOUNGS_MODULUS,
            shear_modulus=SHEAR_MODULUS,
            mass_density=DENSITY,
            thickness=THICKNESS,
            bending="quadratic",
            bending_youngs_modulus=BENDING_YOUNGS_MODULUS,
        )
        scene.geometries.create(name, geometry)
    scene.init()

    def sample() -> dict[str, object]:
        solver = scene.solver
        contact = [0, 0, 0] if disable_contact else [int(value) for value in solver.get_contact_info()]
        return {
            "newton": int(solver.newton_iters),
            "max_pcg": int(solver.max_pcg_iters),
            "total_pcg": int(solver.total_pcg_iters),
            "line_search": int(solver.max_ls_iters),
            "ccd_alpha": float(solver.min_ccd_alpha),
            "contact": contact,
        }

    def positions() -> np.ndarray:
        return scene.finite_element.x.detach().cpu().numpy().reshape(-1, 3).copy()

    return scene.step, torch.cuda.synchronize, sample, positions


def build_genesis(disable_contact: bool, genesis_serial_pipeline: bool):
    import quadrants as qd

    import genesis as gs
    from genesis.engine.systems import ContactTabular, build_scene_engine
    from genesis.utils.misc import qd_to_numpy

    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=GRAVITY),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_fem=False),
        show_viewer=False,
    )
    for name, subdivisions, size, height in LAYER_SPECS:
        vertices, triangles = make_cloth_grid(subdivisions, size, height)
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(write_obj(name, vertices, triangles)),
            ),
            material=gs.materials.FEM.QCloth(
                E=YOUNGS_MODULUS,
                shear_modulus=SHEAR_MODULUS,
                rho=DENSITY,
                thickness=THICKNESS,
                bending_youngs_modulus=BENDING_YOUNGS_MODULUS,
            ),
        )
    scene.build(compile_kernels=False)

    contact_tabular = ContactTabular()
    contact_tabular.default_model(
        friction_rate=0.0,
        resistance=CONTACT_RESISTANCE,
    )
    engine = build_scene_engine(
        scene,
        contact_config={
            "contact/enable": int(not disable_contact),
            "contact/d_hat": D_HAT,
            "contact/init_collision_pair_capacity": PAIR_CAPACITY,
            "contact/ccd_partition": 0,
            "contact/intersection_check": 0,
            "linear_system/tol_rate": 1.0e-4,
            "friction/eps_v": 1.0e-2,
            "extras/pipeline/genesis_serial": int(genesis_serial_pipeline),
        },
        contact_tabular=contact_tabular,
        halfplanes=(
            np.zeros((1, 3), dtype=np.float64),
            np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
        ),
    )

    def sample() -> dict[str, object]:
        contact = (
            [0, 0, 0]
            if disable_contact
            else [
                int(qd_to_numpy(engine.contact.n_pairs_pt)),
                int(qd_to_numpy(engine.contact.n_pairs_ee)),
                int(qd_to_numpy(engine.contact.n_active_pairs)),
            ]
        )
        return {
            "newton": int(engine.get_newton_iters()),
            "max_pcg": int(engine.get_max_pcg_iters()),
            "total_pcg": int(engine.get_total_pcg_iters()),
            "line_search": int(engine.get_max_ls_iters()),
            "ccd_alpha": (1.0 if disable_contact else float(qd_to_numpy(engine.contact.ccd_alpha))),
            "contact": contact,
        }

    def positions() -> np.ndarray:
        return qd_to_numpy(engine.fem.x).copy()

    return engine.step, qd.sync, sample, positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("genesis", "cgq"), required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--frames",
        type=int,
        default=200,
        help="Measured frames after warmup; use at least 200 for performance comparisons",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-output", type=Path, default=None)
    parser.add_argument("--disable-contact", action="store_true")
    parser.add_argument("--genesis-serial-pipeline", action="store_true")
    args = parser.parse_args()

    if args.backend == "genesis":
        step, synchronize, sample, positions = build_genesis(
            args.disable_contact,
            args.genesis_serial_pipeline,
        )
    else:
        if args.genesis_serial_pipeline:
            parser.error("--genesis-serial-pipeline is only valid with --backend genesis")
        step, synchronize, sample, positions = build_cgq(args.disable_contact)

    samples_ms: list[float] = []
    newton: list[int] = []
    max_pcg: list[int] = []
    total_pcg: list[int] = []
    line_search: list[int] = []
    ccd_alpha: list[float] = []
    contact_info: list[list[int]] = []
    total_frames = args.warmup + args.frames
    for frame in range(total_frames):
        profile_window_mark(frame)
        start = time.perf_counter()
        step()
        synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        frame_sample = sample()
        if frame >= args.warmup:
            samples_ms.append(elapsed_ms)
            newton.append(int(frame_sample["newton"]))
            max_pcg.append(int(frame_sample["max_pcg"]))
            total_pcg.append(int(frame_sample["total_pcg"]))
            line_search.append(int(frame_sample["line_search"]))
            ccd_alpha.append(float(frame_sample["ccd_alpha"]))
            contact_info.append([int(value) for value in frame_sample["contact"]])
        print(
            f"[{args.backend}] frame={frame} step={elapsed_ms:.3f}ms "
            f"newton={frame_sample['newton']} "
            f"total_pcg={frame_sample['total_pcg']} "
            f"contact={frame_sample['contact']}",
            flush=True,
        )

    final_positions = positions()
    result = {
        "implementation": args.backend,
        "scene": "multilayer_cloth_halfplane",
        "contact_enabled": not args.disable_contact,
        "genesis_serial_pipeline": args.genesis_serial_pipeline,
        "warmup_frames": args.warmup,
        "measured_frames": args.frames,
        "n_vertices": int(final_positions.shape[0]),
        "n_triangles": int(sum(2 * spec[1] * spec[1] for spec in LAYER_SPECS)),
        "samples_ms": samples_ms,
        "newton": newton,
        "max_pcg": max_pcg,
        "total_pcg": total_pcg,
        "line_search": line_search,
        "ccd_alpha": ccd_alpha,
        "contact_info": contact_info,
        "final_position_sum": final_positions.sum(axis=0).tolist(),
        "final_position_min": final_positions.min(axis=0).tolist(),
        "final_position_max": final_positions.max(axis=0).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.state_output is not None:
        args.state_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.state_output, positions=final_positions)

    print(
        f"[{args.backend}] median={np.median(samples_ms):.3f}ms "
        f"mean={np.mean(samples_ms):.3f}ms "
        f"p95={np.percentile(samples_ms, 95):.3f}ms "
        f"newton={sum(newton)} total_pcg={sum(total_pcg)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
