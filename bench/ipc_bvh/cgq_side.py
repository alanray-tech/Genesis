"""CGQ side of the IPC broad-phase benchmark: capture, ground truth and timing loop.

Runs with the Python of a cuda-graph-qipc (CGQ) checkout, from anywhere:

    <cgq>/.venv/Scripts/python.exe bench/ipc_bvh/cgq_side.py --cgq <cgq> --scene kimono capture
    nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node -t cuda,nvtx -o <out> \
        <cgq>/.venv/Scripts/python.exe bench/ipc_bvh/cgq_side.py --cgq <cgq> --scene kimono time

`capture` simulates the scene with the CGQ defaults (bvh/type = ef_dop14 and the other bvh/* defaults untouched),
dumps the contact closure of the first Newton iteration of every captured frame, and writes under
data/<scene>/:

- frames.npz: the inputs both sides consume (topology, per-vertex d_hat and thickness, contact tables, ignore lists,
  and per frame the start positions, trajectory end positions and path inflation), packed by CGQ's own
  benchmarks/bvh_ee_dual_dataset.pack_capture, plus the half-plane data;
- gt.npz: per frame and channel (PT, EE, PE, PP, PH) every candidate of CGQ's production broad phase as
  `scene.contact_system.recompute()` classifies it (global stencil, distance flag, squared distance, active flag),
  and the candidates the production Newton iteration itself dumped.

`time` rebuilds the scene without stepping and replays every captured frame through `recompute()` (warmup, then
repeats inside cudaProfilerStart/Stop and one NVTX range per repeat). The recompute pipeline borrows the production
build and query subgraphs of the configured broad phase (ContactSystem::bvh_*_build, trajectory_query), so the
kernel timeline nsys records holds the production broad phase of each frame (see cgq_nsys.py).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
SCENES = ("two_bunnies", "cloth_stack", "cloth_stack_x3", "wrecking_ball", "kimono")
CHANNELS = {"pt": 4, "ee": 4, "pe": 3, "pp": 2, "ph": 2}
BVH_KEYS = (
    "bvh/type",
    "bvh/ee_query",
    "bvh/dual/frontier_levels",
    "bvh/dual/target_waves",
    "bvh/dual/max_levels",
    "bvh/deferred_predicates",
    "bvh/ignore_cull",
    "bvh/body_cull",
    "bvh/body_cull_directional",
    "bvh/warp_codim_queries",
)
# Optional iteration-state inputs of recompute(), all or none per group (docs/contact_closure.md)
PATH_FIELDS = ("path_rot", "path_pivot", "path_pivot_disp", "path_inflation", "path_kind", "path_speed")
FOREST_FIELDS = ("forest_q_start", "forest_edge_dq", "rigid_body_t_start", "rigid_body_quat_start", "rigid_body_dq")
KAPPA_FIELDS = {
    "adaptive_kappa_global": "kappa_scale_global",
    "adaptive_kappa_vert": "kappa_scale_vert",
    "adaptive_kappa_body": "kappa_scale_body",
}


def closure_overrides(frame_begin: int, frame_count: int) -> dict:
    return {
        "extras/contact_closure/enable": 1,
        "extras/contact_closure/frame_begin": frame_begin,
        "extras/contact_closure/frame_end": frame_begin + frame_count - 1,
        "extras/contact_closure/max_dumps": 4096,
    }


def build_two_bunnies(workspace: Path, frame_begin: int, frame_count: int, steps: bool):
    from benchmarks.capture_two_bunnies_bvh_frames import _build_scene

    args = SimpleNamespace(
        workspace=workspace,
        frames=frame_begin + frame_count,
        center_separation=1.12,
        closing_speed=1.0,
        dt=0.01,
        d_hat=0.005,
        youngs_modulus=1.0e5,
        max_dumps=4096,
    )
    scene, _ = _build_scene(args)
    if steps:
        for _ in range(frame_begin + frame_count):
            scene.step()
    return scene


def build_wrecking_ball(workspace: Path, frame_begin: int, frame_count: int, steps: bool):
    from benchmarks.capture_wrecking_ball_bvh_frames import _build_scene

    args = SimpleNamespace(workspace=workspace, frame_begin=frame_begin, frame_count=frame_count, max_dumps=4096)
    scene = _build_scene(args)
    if steps:
        for _ in range(frame_begin + frame_count):
            scene.step()
    return scene


def build_kimono(workspace: Path, frame_begin: int, frame_count: int, steps: bool):
    from examples.bspline_franka_cloth_fold import KIMONO_OUTER, KIMONO_SCALE
    from examples.bspline_franka_cloth_fold import main as run_franka_bspline

    # The example steps its frames inside main(); 0 frames builds the scene alone.
    n_frames = frame_begin + frame_count if steps else 0
    argv = ["--no-gui", "--frames", str(n_frames), "--kimono-obj", str(KIMONO_OUTER)]
    argv += ["--kimono-scale", str(KIMONO_SCALE)]
    return run_franka_bspline(
        cloth_kind="kimono",
        workspace=str(workspace),
        scene_config_overrides=closure_overrides(frame_begin, frame_count),
        argv=argv,
    )


def build_cloth_stack(workspace: Path, frame_begin: int, frame_count: int, steps: bool, resolution: int = 1):
    """examples/cloth_stack.py (four stacked cloths onto 16 fixed cubes), headless, hinge bending, the cloth grids
    refined `resolution` times along each side.

    A refined grid scales its contact thickness with its spacing: the non-adjacent primitives of one cloth start
    closer than twice the 1 mm thickness otherwise, inside their own thickness shell, which CGQ refuses.
    """
    from qipc import AffineBodyConstitution, Cloth, Scene, tetmesh, trimesh
    from qipc.unit import kPa

    def cloth_grid(n: int, size: float, height: float):
        axis = (np.arange(n + 1) / n - 0.5) * size
        grid_z, grid_x = np.meshgrid(axis, axis, indexing="ij")
        verts = np.column_stack([grid_x.ravel(), np.full((n + 1) ** 2, height), grid_z.ravel()])
        v00 = (np.arange(n)[:, None] * (n + 1) + np.arange(n)[None, :]).ravel()
        tris = np.concatenate(
            [np.column_stack([v00, v00 + 1, v00 + n + 1]), np.column_stack([v00 + 1, v00 + n + 2, v00 + n + 1])]
        )
        order = np.stack([np.arange(len(v00)), np.arange(len(v00)) + len(v00)], axis=1).ravel()
        return verts, np.ascontiguousarray(tris[order], dtype=np.int32)

    def box(hx: float, hy: float, hz: float):
        verts = np.array(
            [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz], [-hx, -hy, hz], [hx, -hy, hz],
             [hx, hy, hz], [-hx, hy, hz]],
            dtype=np.float64,
        )
        tets = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 3, 4, 6], [1, 4, 5, 6], [3, 4, 6, 7]], dtype=np.int32)
        return verts, tets

    scene = Scene(
        dt=0.01,
        gravity=(0, -9.8, 0),
        workspace=str(workspace),
        **{
            "contact/d_hat": 0.001,
            "contact/init_collision_pair_capacity": 500_000,
            "newton/velocity_tol": 1e-1,
            "line_search/max_iter": 8,
            **closure_overrides(frame_begin, frame_count),
        },
    )
    scene.contact_tabular.default_model(friction_rate=0.0, resistance=1e5)
    abd = AffineBodyConstitution()
    ground = tetmesh(*box(0.4, 0.02, 0.4))
    abd.apply_to(ground, kappa=1e8, mass_density=1e3, is_fixed=np.array([1], dtype=np.int32))
    scene.geometries.create("ground", ground)
    cube_half = 0.025
    cubes = tetmesh(*box(cube_half, cube_half, cube_half))
    cubes.instances.resize(16)
    transforms = np.tile(np.eye(4), (16, 1, 1))
    for idx in range(16):
        transforms[idx, 0, 3] = (idx // 4 - 1.5) * 0.15
        transforms[idx, 1, 3] = 0.02 + cube_half + 0.001
        transforms[idx, 2, 3] = (idx % 4 - 1.5) * 0.15
    cubes.transforms = transforms
    abd.apply_to(cubes, kappa=1e8, mass_density=1e3, is_fixed=np.ones(16, dtype=np.int32))
    scene.geometries.create("cubes", cubes)
    for name, n, size, height in (
        ("cloth_large", 80, 0.5, 0.10),
        ("cloth_mid", 40, 0.3, 0.14),
        ("cloth_small", 20, 0.2, 0.16),
        ("cloth_tiny", 10, 0.1, 0.18),
    ):
        cloth = trimesh(*cloth_grid(n * resolution, size, height))
        Cloth().apply_to(
            cloth,
            youngs_modulus=60 * kPa,
            mass_density=200.0,
            thickness=0.001,
            contact_thickness=None if resolution == 1 else 0.001 / resolution,
            bending="hinge",
            bending_youngs_modulus=1e6,
        )
        scene.geometries.create(name, cloth)
    scene.init()
    if steps:
        for _ in range(frame_begin + frame_count):
            scene.step()
    return scene


# Builder, first captured native frame and frame count of each scene. The frame ranges are those of CGQ's own
# captures (benchmarks/capture_*_bvh_frames.py); cloth_stack starts once the large cloth rests on the cubes.
SCENE_SPECS = {
    "two_bunnies": (build_two_bunnies, 0, 30),
    "cloth_stack": (build_cloth_stack, 20, 30),
    "cloth_stack_x3": (functools.partial(build_cloth_stack, resolution=3), 20, 30),
    "wrecking_ball": (build_wrecking_ball, 80, 30),
    "kimono": (build_kimono, 0, 30),
}


def recompute_inputs(dyn: dict) -> dict:
    """Keyword inputs of recompute() for a dumped iteration (see ContactClosure.audit)."""

    def present(key: str) -> bool:
        return key in dyn and np.asarray(dyn[key]).size > 0

    kwargs = {}
    if present("path_rot"):
        kwargs.update({key: dyn[key] for key in PATH_FIELDS})
    if present("forest_q_start"):
        kwargs.update({key: dyn[key] for key in FOREST_FIELDS})
    for key, name in KAPPA_FIELDS.items():
        if present(key):
            kwargs[name] = dyn[key]
    return kwargs


def git_revision(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def capture(args: argparse.Namespace) -> None:
    import torch

    from benchmarks.bvh_ee_dual_dataset import pack_capture, prepare_capture_output, write_selected_newton_only
    from qipc.offline import ContactClosure

    builder, frame_begin, frame_count = SCENE_SPECS[args.scene]
    out_dir = args.data / args.scene
    workspace = out_dir / "capture"
    closure_dir = workspace / "contact_closure"
    frames_path = out_dir / "frames.npz"
    prepare_capture_output(closure_dir, frames_path, overwrite=True)

    t_start = time.perf_counter()
    with write_selected_newton_only(closure_dir):
        scene = builder(workspace, frame_begin, frame_count, steps=True)
    print(f"[{args.scene}] simulated {frame_begin + frame_count} frames in {time.perf_counter() - t_start:.1f} s")
    for key in BVH_KEYS:
        print(f"[{args.scene}] {key} = {scene.config[key]}")

    native_frames = list(range(frame_begin, frame_begin + frame_count))
    pack_capture(scene, closure_dir, frames_path, dataset_id=args.scene, native_frames=native_frames)
    closure = ContactClosure(closure_dir)
    static = closure.static
    with np.load(frames_path) as packed:
        archive = {name: packed[name] for name in packed.files}
    n_hp = int(static["n_halfplanes"])
    archive["halfplane_normals"] = np.asarray(static["halfplane_normals"], dtype=np.float64).reshape(n_hp, 3)
    archive["halfplane_contact_element_ids"] = np.asarray(static["halfplane_contact_element_ids"], dtype=np.int32)
    archive["contact_kappa_table"] = np.asarray(static["contact_kappa_table"], dtype=np.float64)
    archive["cgq_revision"] = np.asarray(git_revision(args.cgq))
    archive["bvh_config"] = np.asarray(json.dumps({key: str(scene.config[key]) for key in BVH_KEYS}))

    tables = {f"{ch}_{name}": [] for ch in CHANNELS for name in ("stencil", "flag", "active", "d2")}
    prod = {f"prod_{ch}": [] for ch in CHANNELS}
    hp_off = None
    n_verts = archive["positions"].shape[1]
    # The path state the distance rule of the candidates reads (contact_function/candidate_reach.h), zero for
    # straight paths
    path_speed, path_kind = [], []
    for frame in native_frames:
        dyn = closure.load_dynamic(frame, 0)
        speed, kind = np.asarray(dyn.get("path_speed", ())), np.asarray(dyn.get("path_kind", ()))
        path_speed.append(speed.reshape(n_verts) if speed.size else np.zeros(n_verts))
        path_kind.append(kind.reshape(n_verts).astype(np.int32) if kind.size else np.zeros(n_verts, np.int32))
        result = scene.contact_system.recompute(dyn["positions"], dyn["trajectory_end_positions"], **recompute_inputs(dyn))
        hp_off = result.hp_off
        for ch, width in CHANNELS.items():
            table = getattr(result, ch)
            n = table.n_candidates
            tables[f"{ch}_stencil"].append(table.stencil.cpu().numpy().reshape(n, width))
            flag = table.flag.cpu().numpy() if table.flag is not None else np.zeros(n, dtype=np.uint32)
            tables[f"{ch}_flag"].append(flag.astype(np.uint32))
            tables[f"{ch}_active"].append(table.active.cpu().numpy().astype(np.int8))
            tables[f"{ch}_d2"].append(table.d2.cpu().numpy())
            pairs = np.asarray(dyn.get(f"candidate_pairs_{ch}", np.zeros((0, 2))), dtype=np.int32).reshape(-1, 2)
            prod[f"prod_{ch}"].append(pairs)
        counts = {ch: int(getattr(result, ch).active.sum().item()) for ch in CHANNELS}
        print(
            f"[{args.scene}] frame {frame}: candidates "
            + " ".join(f"{ch}={getattr(result, ch).n_candidates}" for ch in CHANNELS)
            + " | active "
            + " ".join(f"{ch}={n}" for ch, n in counts.items()),
            flush=True,
        )
    archive["hp_off"] = np.asarray(hp_off, dtype=np.int64)
    archive["path_speed"] = np.stack(path_speed)
    archive["path_kind"] = np.stack(path_kind)
    np.savez_compressed(frames_path, **archive)

    gt = {}
    for key, chunks in {**tables, **prod}.items():
        gt[key] = np.concatenate(chunks)
        gt[f"{key}_offsets"] = np.concatenate([[0], np.cumsum([len(c) for c in chunks])]).astype(np.int64)
    gt["native_frames"] = np.asarray(native_frames, dtype=np.int32)
    np.savez_compressed(out_dir / "gt.npz", **gt)
    torch.cuda.synchronize()
    print(f"[{args.scene}] wrote {frames_path} and {out_dir / 'gt.npz'}")


def time_frames(args: argparse.Namespace) -> None:
    import torch

    from qipc.offline import ContactClosure

    builder, frame_begin, frame_count = SCENE_SPECS[args.scene]
    out_dir = args.data / args.scene
    closure = ContactClosure(out_dir / "capture" / "contact_closure")
    scene = builder(out_dir / "time_ws", frame_begin, frame_count, steps=False)
    native_frames = list(range(frame_begin, frame_begin + frame_count))
    dyns = []
    for frame in native_frames:
        dyn = closure.load_dynamic(frame, 0)
        # Staged on the device once, so a repeat copies device to device alone
        staged = {key: torch.as_tensor(np.asarray(value)).cuda() for key, value in recompute_inputs(dyn).items()}
        dyns.append(
            (torch.as_tensor(dyn["positions"]).cuda(), torch.as_tensor(dyn["trajectory_end_positions"]).cuda(), staged)
        )
    # Every frame once first, so that each buffer has grown to the largest frame before the measured repeats
    for pos, end, staged in dyns:
        scene.contact_system.recompute(pos, end, **staged)
    torch.cuda.synchronize()
    memory = scene.solver.memory_stats
    # Wall time of a whole recompute() call (inputs copy, pipeline, export kernels, count download), which also
    # shows how much the kernel tracing of an nsys run slows the pipeline down
    walls = {}
    torch.cuda.profiler.start()
    for i_f, (pos, end, staged) in enumerate(dyns):
        for _ in range(args.warmup):
            scene.contact_system.recompute(pos, end, **staged)
        torch.cuda.synchronize()
        samples = []
        for i_r in range(args.repeats):
            torch.cuda.nvtx.range_push(f"cgq frame={native_frames[i_f]} rep={i_r}")
            start = time.perf_counter()
            scene.contact_system.recompute(pos, end, **staged)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
            torch.cuda.nvtx.range_pop()
        walls[native_frames[i_f]] = float(np.median(samples) * 1e3)
    torch.cuda.profiler.stop()
    (out_dir / "cgq_memory.json").write_text(json.dumps(memory, indent=2), encoding="utf-8")
    label = "traced" if args.traced else "untraced"
    (out_dir / f"cgq_recompute_wall_{label}.json").write_text(json.dumps(walls, indent=1), encoding="utf-8")
    print(f"[{args.scene}] timed {len(dyns)} frames x {args.repeats} repeats; solver memory {memory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cgq", type=Path, required=True, help="cuda-graph-qipc checkout whose build runs")
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--traced", action="store_true", help="the time mode runs under nsys (labels the wall times)")
    parser.add_argument("mode", choices=("capture", "time"))
    args = parser.parse_args()
    args.cgq = args.cgq.resolve()
    args.data = args.data.resolve()
    # CGQ's benchmarks and examples import as packages from its root, and resolve their assets from it
    sys.path.insert(0, str(args.cgq))
    os.chdir(args.cgq)
    if args.mode == "capture":
        capture(args)
    else:
        time_frames(args)


if __name__ == "__main__":
    main()
