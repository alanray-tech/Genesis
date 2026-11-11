"""CGQ's classification-only narrow phase (cgq_filter/cgq_filter.cu) on both candidate sets, checked and timed.

    <cgq>/.venv/Scripts/python.exe bench/ipc_bvh/cgq_narrow.py --cgq <cgq> --scene kimono [--schedule deferred]

Per frame:
- on the candidates the production Newton iteration dumped (gt.npz `prod_*`): the active pairs and their distance
  flags must equal those of `recompute()` (gt.npz), which validates the extension against CGQ's export kernels;
- on the candidates of the bvh.py side (data/<scene>/upstream_<schedule>_candidates.npz, written by upstream_side.py):
  the active pairs must equal CGQ's, with the same distance flags, which is the active-set check of this benchmark run
  through CGQ's own classifiers;
- the warm time of the count-only kernels (no per-pair output, as cipc_count_active_*) on each candidate set, CUDA
  events, median and p90 over the repeats.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def load_extension(cgq: Path):
    from torch.utils.cpp_extension import load

    sys.path.insert(0, str(cgq))
    from build import _msvc_env

    # nvcc needs the MSVC host compiler on PATH, set up as CGQ's own build does
    os.environ.update(_msvc_env())
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "{}.{}".format(*torch.cuda.get_device_capability()))
    native = cgq / "qipc" / "_src" / "native"
    return load(
        name="cgq_filter",
        sources=[str(HERE / "cgq_filter" / "cgq_filter.cu")],
        extra_include_paths=[str(native), str(cgq / "external" / "eigen")],
        extra_cuda_cflags=["-std=c++20", "--expt-relaxed-constexpr", "--extended-lambda", "-O3", "-DNOMINMAX"],
        extra_cflags=["/std:c++20", "/DNOMINMAX"] if os.name == "nt" else ["-std=c++20"],
        build_directory=str(_build_dir()),
        verbose=False,
    )


def _build_dir() -> Path:
    path = HERE / "data" / "cgq_filter_build"
    path.mkdir(parents=True, exist_ok=True)
    return path


def frame_slice(arrays: dict, key: str, i_f: int) -> np.ndarray:
    offsets = arrays[f"{key}_offsets"]
    return arrays[key][offsets[i_f] : offsets[i_f + 1]]


def rows_view(rows: np.ndarray) -> np.ndarray:
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    return rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()


def set_diff(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    view_a, view_b = np.unique(rows_view(a)), np.unique(rows_view(b))
    common = np.intersect1d(view_a, view_b, assume_unique=True)
    return len(view_a) - len(common), len(view_b) - len(common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cgq", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--schedule", default="deferred")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    ext = load_extension(args.cgq.resolve())

    data = HERE / "data" / args.scene
    with np.load(data / "frames.npz") as f, np.load(data / "gt.npz") as g:
        frames = {k: f[k] for k in f.files}
        gt = {k: g[k] for k in g.files}
    upstream_path = data / f"upstream_{args.schedule}_candidates.npz"
    upstream = None
    if upstream_path.is_file():
        with np.load(upstream_path) as u:
            upstream = {k: u[k] for k in u.files}

    def cuda(array, dtype):
        return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device="cuda")

    surf_verts = cuda(frames["surf_verts"], torch.int32)
    faces = cuda(frames["surf_faces"], torch.int32)
    edges = cuda(frames["surf_edges"], torch.int32)
    d_hats = cuda(frames["d_hats"], torch.float64)
    thicknesses = cuda(frames["thicknesses"], torch.float64)
    faces_np, edges_np, surf_verts_np = frames["surf_faces"], frames["surf_edges"], frames["surf_verts"]

    def run(pt_pairs, ee_pairs, positions, write_pairs):
        return ext.classify(pt_pairs, ee_pairs, surf_verts, faces, edges, positions, d_hats, thicknesses, write_pairs)

    def active_rows(pt_pairs_np, ee_pairs_np, outputs):
        pt_flag, pt_act, ee_flag, ee_act, _ = (t.cpu().numpy() for t in outputs)
        pt_stencil = np.column_stack([surf_verts_np[pt_pairs_np[:, 0]], faces_np[pt_pairs_np[:, 1]], pt_flag])
        ee_stencil = np.column_stack([edges_np[ee_pairs_np[:, 0]], edges_np[ee_pairs_np[:, 1]], ee_flag])
        return {"pt": pt_stencil[pt_act.astype(bool)], "ee": ee_stencil[ee_act.astype(bool)]}

    def timed(pt_pairs, ee_pairs, positions) -> dict:
        for _ in range(args.warmup):
            run(pt_pairs, ee_pairs, positions, False)
        samples = []
        for _ in range(args.repeats):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            run(pt_pairs, ee_pairs, positions, False)
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop))
        return {"median_ms": float(np.median(samples)), "p90_ms": float(np.percentile(samples, 90))}

    rows = []
    for i_f, native_frame in enumerate(gt["native_frames"].tolist()):
        positions = cuda(frames["positions"][i_f], torch.float64)
        cgq_active = {}
        for ch in ("pt", "ee"):
            stencil = frame_slice(gt, f"{ch}_stencil", i_f)
            active = frame_slice(gt, f"{ch}_active", i_f).astype(bool)
            flag = frame_slice(gt, f"{ch}_flag", i_f).astype(np.int64)
            cgq_active[ch] = np.column_stack([stencil, flag])[active]
        prod_pt = frame_slice(gt, "prod_pt", i_f).reshape(-1, 2)
        prod_ee = frame_slice(gt, "prod_ee", i_f).reshape(-1, 2)
        prod_pt_t, prod_ee_t = cuda(prod_pt, torch.int32), cuda(prod_ee, torch.int32)
        prod_rows = active_rows(prod_pt, prod_ee, run(prod_pt_t, prod_ee_t, positions, True))
        row = {"frame": native_frame, "cgq_candidates": {"pt": len(prod_pt), "ee": len(prod_ee)}}
        row["extension_vs_recompute"] = {ch: set_diff(prod_rows[ch], cgq_active[ch]) for ch in ("pt", "ee")}
        row["narrow_on_cgq_candidates"] = timed(prod_pt_t, prod_ee_t, positions)
        if upstream is not None:
            up_pt = frame_slice(upstream, "pt_pairs", i_f)
            up_ee = frame_slice(upstream, "ee_pairs", i_f)
            up_pt_t, up_ee_t = cuda(up_pt, torch.int32), cuda(up_ee, torch.int32)
            up_rows = active_rows(up_pt, up_ee, run(up_pt_t, up_ee_t, positions, True))
            row["upstream_candidates"] = {"pt": len(up_pt), "ee": len(up_ee)}
            row["upstream_vs_cgq_active"] = {ch: set_diff(up_rows[ch], cgq_active[ch]) for ch in ("pt", "ee")}
            row["narrow_on_upstream_candidates"] = timed(up_pt_t, up_ee_t, positions)
        rows.append(row)
        print(f"[{args.scene}] " + json.dumps(row), flush=True)

    is_extension_exact = all(d == (0, 0) for r in rows for d in r["extension_vs_recompute"].values())
    is_upstream_exact = upstream is not None and all(
        d == (0, 0) for r in rows for d in r["upstream_vs_cgq_active"].values()
    )
    result = {
        "scene": args.scene,
        "schedule": args.schedule,
        "extension_reproduces_recompute": is_extension_exact,
        "upstream_active_equals_cgq": is_upstream_exact,
        "frames": rows,
    }
    out = HERE / "results" / f"{args.scene}_cgq_narrow_{args.schedule}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[{args.scene}] extension == recompute: {is_extension_exact}; upstream active == CGQ: {is_upstream_exact}")
    print(f"[{args.scene}] wrote {out}")


if __name__ == "__main__":
    main()
