"""How the raw hits of the bvh.py queries spread over their queries, on one frame (load balance of the traversal).

    powershell -File bench\ipc_bvh\run_upstream.ps1 ... is the environment; run as
    <genesis python> bench/ipc_bvh/hit_distribution.py --scene kimono --frame-index 20

bvh.py traverses one query per thread, so a query with many hits keeps its thread, and the warp it belongs to, busy
long after the others finished. For the PT and EE queries of the deferred schedule this prints the hit count
quantiles over the queries, the share of all hits held by the heaviest queries, and the bodies those queries belong
to (results/<scene>_hit_distribution.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import genesis as gs

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frame-index", type=int, default=20)
    args = parser.parse_args()
    gs.init(backend=gs.gpu, precision="32", performance_mode=True, logging_level="warning")
    import pipeline
    from genesis.utils.misc import qd_to_numpy

    with np.load(HERE / "data" / args.scene / "frames.npz") as f:
        frames = {k: f[k] for k in f.files}
    bench = pipeline.Pipeline(frames, "deferred")
    bench.load_frame(frames, args.frame_index)
    bench.resize(bench.run())
    counts = bench.run()
    body = frames["body_id"]
    result = {"scene": args.scene, "frame_index": args.frame_index}
    for ch, raw, n_queries, query_verts in (
        ("pt", bench.pt_raw, len(frames["surf_verts"]), frames["surf_verts"][:, None]),
        ("ee", bench.ee_raw, len(frames["surf_edges"]), frames["surf_edges"]),
    ):
        triplets = qd_to_numpy(raw.triplets)[: counts[f"{ch}_raw"]]
        hits = np.bincount(triplets[:, 2], minlength=n_queries)
        order = np.argsort(hits)[::-1]
        top = order[:32]
        top_bodies, top_counts = np.unique(body[query_verts[top, 0]], return_counts=True)
        result[ch] = {
            "raw_hits": int(hits.sum()),
            "queries": n_queries,
            "hits_quantiles": {q: float(np.percentile(hits, q)) for q in (50, 90, 99, 99.9, 100)},
            "share_of_hits_in_top_0.1pct_queries": float(hits[order[: max(1, n_queries // 1000)]].sum() / hits.sum()),
            "bodies_of_top_32_queries": dict(zip(map(int, top_bodies), map(int, top_counts))),
        }
        print(f"[{args.scene}] {ch}: " + json.dumps(result[ch]))
    (HERE / "results" / f"{args.scene}_hit_distribution.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
