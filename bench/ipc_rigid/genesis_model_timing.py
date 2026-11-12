"""Step timings of Genesis and CGQ's Genesis rigid model on the coupled parity scenes, at equal Newton work (run after the
timed runs of genesis_model_timing.ps1, under an exclusive GPU lease; any Python with numpy).

CGQ's step times are those of its own run of a scene (genesis_model_case2_cgq.py, genesis_model_case3_cgq.py), Genesis's
those of its one-step replays of CGQ's frames (genesis_side.py, genesis_model_case3_genesis.py): both sides step from
the same state at every frame. The medians are over every frame but the first (graph capture, compilation), and over the
frames where both took the same number of Newton iterations. The rigid preconditioners differ: CGQ's articulated
factorization of the forest against Genesis's island Cholesky of its rigid block, so the PCG counts are laid beside the
times.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import genesis_model_scenes as gms

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="_timing")
args = parser.parse_args()


def load_case2():
    cgq = np.load(gms.DATA / "cgq_box_on_cloth.npz")
    genesis = json.loads((gms.RESULTS / f"genesis_gpu{args.tag}.json").read_text())["scenes"]["box_on_cloth"]
    counts = genesis["newton_counts"]["per_frame"]
    return {
        "cgq_ms": np.asarray(cgq["step_ms"]),
        "genesis_ms": np.asarray(genesis["genesis_step_ms"]),
        **{f"{side}_{name}": np.asarray(counts[f"{side}_{name}"]) for side in ("cgq", "genesis") for name in ("newton", "pcg")},
    }


def load_case3(tag):
    cgq = np.load(gms.DATA / f"cgq_case3{tag}.npz")
    genesis = json.loads((gms.RESULTS / f"genesis_model_case3_gpu{tag}.json").read_text())
    counts = genesis["newton_counts"]["per_frame"]
    n = len(genesis["step_ms"])
    return {
        "cgq_ms": np.asarray(cgq["step_ms"])[:n],
        "genesis_ms": np.asarray(genesis["step_ms"]),
        **{f"{side}_{name}": np.asarray(counts[f"{side}_{name}"]) for side in ("cgq", "genesis") for name in ("newton", "pcg")},
    }


def row(name, data):
    frames = np.arange(1, len(data["genesis_ms"]))
    equal = frames[data["cgq_newton"][frames] == data["genesis_newton"][frames]]
    out = {"scene": name, "frames": int(len(frames)), "equal_newton_frames": int(len(equal))}
    for side in ("genesis", "cgq"):
        out[f"{side}_median_ms"] = float(np.median(data[f"{side}_ms"][frames]))
        out[f"{side}_median_ms_equal_newton"] = float(np.median(data[f"{side}_ms"][equal]))
        out[f"{side}_newton"] = int(data[f"{side}_newton"][frames].sum())
        out[f"{side}_pcg"] = int(data[f"{side}_pcg"][frames].sum())
        out[f"{side}_ms_per_newton"] = float(data[f"{side}_ms"][frames].sum() / data[f"{side}_newton"][frames].sum())
    out["ratio_median"] = out["genesis_median_ms"] / out["cgq_median_ms"]
    out["ratio_median_equal_newton"] = out["genesis_median_ms_equal_newton"] / out["cgq_median_ms_equal_newton"]
    out["ratio_per_newton"] = out["genesis_ms_per_newton"] / out["cgq_ms_per_newton"]
    return out


rows = [row("case 2 box_on_cloth (tol 1e-6)", load_case2())]
for tag, label in (("_timing", "case 3 Franka press (tol 5e-2)"), ("_timing_tol1e-6", "case 3 Franka press (tol 1e-6)")):
    if (gms.DATA / f"cgq_case3{tag}.npz").exists():
        rows.append(row(label, load_case3(tag)))
lines = [
    "| scene | frames (equal Newton) | Genesis median ms | CGQ median ms | ratio | equal-Newton median ms G / C | ratio | "
    "ms per Newton G / C | ratio | Newton G / C | PCG G / C |",
    "|---|---|---|---|---|---|---|---|---|---|---|",
]
for r in rows:
    lines.append(
        f"| {r['scene']} | {r['frames']} ({r['equal_newton_frames']}) | {r['genesis_median_ms']:.2f} | "
        f"{r['cgq_median_ms']:.2f} | {r['ratio_median']:.2f} | {r['genesis_median_ms_equal_newton']:.2f} / "
        f"{r['cgq_median_ms_equal_newton']:.2f} | {r['ratio_median_equal_newton']:.2f} | {r['genesis_ms_per_newton']:.2f} / "
        f"{r['cgq_ms_per_newton']:.2f} | {r['ratio_per_newton']:.2f} | {r['genesis_newton']} / {r['cgq_newton']} | "
        f"{r['genesis_pcg']} / {r['cgq_pcg']} |"
    )
lines.append(
    "\nRigid preconditioners differ: CGQ's articulated factorization of the forest (a 6x6 block per body, edge pivots) "
    "against Genesis's island Cholesky of its rigid block (with the proxy stiffness); the PCG counts are beside the times."
)
text = "\n".join(lines)
print(text)
(gms.RESULTS / "genesis_model_timing.md").write_text(text + "\n")
(gms.RESULTS / "genesis_model_timing.json").write_text(json.dumps(rows, indent=1))
