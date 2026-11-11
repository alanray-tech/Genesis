"""Upstream side of the IPC broad-phase benchmark: CGQ's PT / EE / PH queries expressed on genesis/engine/bvh.py.

    python bench/ipc_bvh/upstream_side.py --scene kimono [--repeats 20] [--nsys-frame -1]

Run with the Genesis environment (Quadrants on PYTHONPATH, see run_upstream.ps1). For each captured frame of
data/<scene>/frames.npz (written by cgq_side.py), the pipeline a Newton iteration would run is:

- bounds: one kernel writing the leaf boxes of the triangles and edges, and the query boxes of the surface points and
  edges, exactly as CGQ inflates and sweeps them for the same query (start and trajectory end positions, expanded by
  the thickness, the path inflation and the d_hat of the primitive, the point query by its gap d_hat + thickness +
  inflation, the edge query by the global d_hat), in fp64 and rounded outward to the fp32 boxes of bvh.py;
- build: build_bvh of a triangle tree and of an edge tree (bvh.py holds one leaf type per tree);
- PT query: every surface point box against the triangle tree, BoxQuery.is_kept applying CGQ's PT leaf predicates
  (same body without self collision, ignored body pairs, the point being a triangle vertex, the PT contact table);
- EE query: every edge query box against the edge tree, each unordered pair once in CGQ's orientation (lower edge
  index first), with CGQ's EE predicates (shared vertex, EE contact table);
- PH query: every surface vertex against every half-plane, CGQ's halfplane_query rule (no BVH on either side);
- narrow phase: a Quadrants kernel classifying every candidate (distance flag, squared distance, activation) as CGQ's
  narrow phase does, over the device-side candidate counts.

The candidates and the classification are checked against CGQ (results/<scene>_upstream.json): the official active
set of this side is the reference port (narrow_ref.py, verified by verify_port.py) applied to the bvh.py candidates,
and the Quadrants classification is compared to it.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import quadrants as qd

import genesis as gs

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--schedule",
        choices=("inline", "deferred"),
        default="deferred",
        help="CGQ's distance rule inside BoxQuery.is_kept, or in a filter kernel over the raw hits",
    )
    parser.add_argument(
        "--nsys-frame", type=int, default=None, help="time only this frame index (-1: the most candidates), for nsys"
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", performance_mode=True, logging_level="warning")
    # Imported after gs.init: the module reads the precision of the simulation
    import pipeline

    with np.load(args.data / args.scene / "frames.npz") as frames, np.load(args.data / args.scene / "gt.npz") as gt:
        frames = {key: frames[key] for key in frames.files}
        gt = {key: gt[key] for key in gt.files}
    bench = pipeline.Pipeline(frames, args.schedule)
    result = pipeline.run_benchmark(bench, frames, gt, args)
    result["genesis_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    result["quadrants_version"] = ".".join(map(str, qd.__version__))
    result["gpu"] = torch.cuda.get_device_name()
    suffix = "_nsys" if args.nsys_frame is not None else ""
    out = HERE / "results" / f"{args.scene}_upstream_{args.schedule}{suffix}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[{args.scene}] wrote {out}")


if __name__ == "__main__":
    main()
