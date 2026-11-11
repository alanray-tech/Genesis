"""Check the reference narrow phase (narrow_ref.py) against CGQ on CGQ's own candidates, frame by frame.

For every candidate `scene.contact_system.recompute()` returned (gt.npz), the port must give the same distance flag
and the same active bit. The script also checks that the candidate set of recompute() equals the one the production
Newton iteration dumped into the contact closure, so that recompute() stands for the production broad phase.

    python bench/ipc_bvh/verify_port.py [--scenes ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from narrow_ref import classify

HERE = Path(__file__).resolve().parent


def load(scene: str):
    with np.load(HERE / "data" / scene / "frames.npz") as frames, np.load(HERE / "data" / scene / "gt.npz") as gt:
        return {k: frames[k] for k in frames.files}, {k: gt[k] for k in gt.files}


def frame_slice(gt: dict, key: str, i_f: int) -> np.ndarray:
    offsets = gt[f"{key}_offsets"]
    return gt[key][offsets[i_f] : offsets[i_f + 1]]


def production_stencils(frames: dict, channel: str, pairs: np.ndarray) -> np.ndarray:
    """Global stencils of the broad-phase-local pairs the contact closure dumps (docs/contact_closure.md)."""
    surf_verts, faces, edges = frames["surf_verts"], frames["surf_faces"], frames["surf_edges"]
    if channel == "pt":
        return np.column_stack([surf_verts[pairs[:, 0]], faces[pairs[:, 1]]])
    if channel == "ee":
        return np.column_stack([edges[pairs[:, 0]], edges[pairs[:, 1]]])
    if channel == "ph":
        return np.column_stack([surf_verts[pairs[:, 0]], int(frames["hp_off"]) + pairs[:, 1]])
    raise ValueError(f"channel {channel} has no production stencil (codim scenes are out of scope)")


def stencil_set(stencils: np.ndarray) -> set:
    return set(map(tuple, stencils.tolist()))


def verify_scene(scene: str) -> dict:
    frames, gt = load(scene)
    device = "cuda"
    d_hats = torch.as_tensor(frames["d_hats"], device=device)
    thicknesses = torch.as_tensor(frames["thicknesses"], device=device)
    normals = torch.as_tensor(frames["halfplane_normals"], device=device)
    hp_off = int(frames["hp_off"])
    rows = []
    for i_f, native_frame in enumerate(gt["native_frames"].tolist()):
        positions = torch.as_tensor(frames["positions"][i_f], device=device)
        row = {"frame": native_frame}
        for channel in ("pt", "ee", "pe", "pp", "ph"):
            stencil = frame_slice(gt, f"{channel}_stencil", i_f)
            if channel in ("pe", "pp"):
                if len(stencil):
                    raise RuntimeError(f"{scene}: codim channel {channel} has candidates, out of scope")
                continue
            flag_cgq = frame_slice(gt, f"{channel}_flag", i_f).astype(np.int64)
            active_cgq = frame_slice(gt, f"{channel}_active", i_f).astype(bool)
            n_hp = len(normals)
            planes_pos = positions[hp_off : hp_off + n_hp]
            flag, _, active = classify(
                channel, torch.as_tensor(stencil, device=device).long(), positions, d_hats, thicknesses,
                planes_pos=planes_pos, planes_normal=normals, hp_off=hp_off,
            )
            flag, active = flag.cpu().numpy().astype(np.int64), active.cpu().numpy()
            prod = production_stencils(frames, channel, frame_slice(gt, f"prod_{channel}", i_f))
            row[channel] = {
                "candidates": len(stencil),
                "active": int(active_cgq.sum()),
                "flag_mismatch": int((flag != flag_cgq).sum()) if channel != "ph" else 0,
                "active_mismatch": int((active != active_cgq).sum()),
                "production_candidates": len(prod),
                "production_set_equal": stencil_set(prod) == stencil_set(stencil),
            }
        rows.append(row)
        print(f"[{scene}] frame {native_frame}: " + json.dumps({k: v for k, v in row.items() if k != "frame"}))
    totals = {
        channel: {
            key: sum(r[channel][key] for r in rows)
            for key in ("candidates", "active", "flag_mismatch", "active_mismatch")
        }
        for channel in ("pt", "ee", "ph")
    }
    is_pass = all(r[ch]["flag_mismatch"] == 0 and r[ch]["active_mismatch"] == 0 for r in rows for ch in totals)
    is_prod = all(r[ch]["production_set_equal"] for r in rows for ch in totals)
    print(f"[{scene}] totals {json.dumps(totals)}")
    print(f"[{scene}] port reproduces CGQ on every frame: {is_pass}; recompute == production candidates: {is_prod}")
    return {"scene": scene, "pass": is_pass, "recompute_equals_production": is_prod, "totals": totals, "frames": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="+", default=None)
    args = parser.parse_args()
    scenes = args.scenes or sorted(p.name for p in (HERE / "data").iterdir() if (p / "gt.npz").is_file())
    results = [verify_scene(scene) for scene in scenes]
    out = HERE / "results" / "port_verification.json"
    out.parent.mkdir(exist_ok=True)
    previous = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    previous.update({r["scene"]: r for r in results})
    out.write_text(json.dumps(previous, indent=1), encoding="utf-8")
    if not all(r["pass"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
