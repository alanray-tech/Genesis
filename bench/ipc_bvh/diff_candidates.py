"""Describe the candidates one side lists and the other does not, on one frame.

    python bench/ipc_bvh/diff_candidates.py --scene kimono --frame-index 0 [--schedule deferred]

For every PT / EE candidate of the bvh.py side missing from CGQ's production candidates (and the reverse), print the
bodies of both primitives, their path kinds, and the terms of CGQ's distance rule (squared distance, band, travel),
so that the rule or the box test responsible for the difference shows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from narrow_ref import classify

HERE = Path(__file__).resolve().parent


def frame_slice(arrays: dict, key: str, i_f: int) -> np.ndarray:
    offsets = arrays[f"{key}_offsets"]
    return arrays[key][offsets[i_f] : offsets[i_f + 1]]


def rows_view(rows: np.ndarray) -> np.ndarray:
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    return rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--schedule", default="deferred")
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()
    data = HERE / "data" / args.scene
    with np.load(data / "frames.npz") as f, np.load(data / "gt.npz") as g, np.load(
        data / f"upstream_{args.schedule}_candidates.npz"
    ) as u:
        frames, gt, up = ({k: z[k] for k in z.files} for z in (f, g, u))
    i_f = args.frame_index
    pos, end = frames["positions"][i_f], frames["trajectory_end_positions"][i_f]
    infl, speed, kind = frames["path_inflation"][i_f], frames["path_speed"][i_f], frames["path_kind"][i_f]
    body, d_hats, thick = frames["body_id"], frames["d_hats"], frames["thicknesses"]
    travel = np.where(speed > 0.0, speed, np.linalg.norm(end - pos, axis=1) + infl)
    tables = {
        "pt": (frames["surf_verts"], frames["surf_faces"]),
        "ee": (frames["surf_edges"], frames["surf_edges"]),
    }
    for ch, (side_a, side_b) in tables.items():
        up_pairs = frame_slice(up, f"{ch}_pairs", i_f)
        cgq_pairs = frame_slice(gt, f"prod_{ch}", i_f).reshape(-1, 2)
        view_up, view_cgq = rows_view(up_pairs), rows_view(cgq_pairs)
        for label, pairs, mask in (
            ("only bvh.py", up_pairs, ~np.isin(view_up, view_cgq)),
            ("only CGQ", cgq_pairs, ~np.isin(view_cgq, view_up)),
        ):
            extra = pairs[mask]
            print(f"[{args.scene}] frame index {i_f} {ch}: {label}: {len(extra)} of {len(pairs)}")
            if not len(extra):
                continue
            verts_a = side_a[extra[:, 0]].reshape(len(extra), -1)
            verts_b = side_b[extra[:, 1]].reshape(len(extra), -1)
            stencil = np.concatenate([verts_a, verts_b], axis=1)
            _, d2, active = classify(
                ch, torch.as_tensor(stencil).long(), torch.as_tensor(pos), torch.as_tensor(d_hats),
                torch.as_tensor(thick),
            )
            d2 = d2.numpy()
            n_a = verts_a.shape[1]
            if ch == "pt":
                xi = thick[stencil[:, 0]] + thick[stencil[:, 1:]].max(axis=1)
                d_hat = (d_hats[stencil[:, 0]] + d_hats[stencil[:, 1]]) / 2.0
            else:
                xi = thick[stencil[:, :2]].max(axis=1) + thick[stencil[:, 2:]].max(axis=1)
                d_hat = (d_hats[stencil[:, 0]] + d_hats[stencil[:, 2]]) / 2.0
            band = d_hat + xi
            # candidate_vertex_travel summed over the farthest-moving vertex of each side
            reach = band + travel[stencil[:, :n_a]].max(axis=1) + travel[stencil[:, n_a:]].max(axis=1)
            bodies = np.column_stack([body[stencil[:, 0]], body[stencil[:, n_a]]])
            unique_bodies, counts = np.unique(bodies, axis=0, return_counts=True)
            print(f"  body pairs (first, count): {list(zip(map(tuple, unique_bodies[:10].tolist()), counts[:10]))}")
            print(f"  path kinds present: {np.unique(kind[stencil])}, active at start: {int(active.sum())}")
            order = np.argsort(np.sqrt(d2) - reach)[::-1][: args.show]
            for i in order:
                print(
                    f"  {stencil[i].tolist()} d={np.sqrt(d2[i]):.6g} band={band[i]:.6g} reach={reach[i]:.6g} "
                    f"kind={kind[stencil[i]].tolist()} speed={np.round(speed[stencil[i]], 6).tolist()}"
                )


if __name__ == "__main__":
    main()
