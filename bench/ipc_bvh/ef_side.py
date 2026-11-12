"""Acceptance of the EF broad phase of Genesis (genesis/engine/ipc/broad_phase.py) against CGQ on T0's captured frames.

    python bench/ipc_bvh/ef_side.py --scene kimono [--backend gpu|cpu]

Run with the Genesis environment of the tree under test first on PYTHONPATH (see run_ef.ps1). For each captured frame
of data/<scene>/frames.npz, the surface (surf_verts, surf_edges, surf_faces) runs the EF broad phase over the sweep from
the positions to the trajectory end positions, the path inflation of every vertex standing as the bulge of its path,
every body a group and no vertex rigid (CGQ pairs every body with every body). The sub-pairs then go through CGQ's
predicates and distance rule (pipeline.Mesh, the same rules the bvh.py side applies), and the reference narrow phase
(narrow_ref.py) classifies them. The candidate set and the active set (stencil and distance flag) are compared to
CGQ's on every frame, for the PT and EE channels. The PH channel (vertex against half-plane) is outside the EF broad
phase.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--capacity", type=int, default=1 << 12, help="initial pair capacity, small to test growth")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    # Imported after gs.init: the modules read the precision of the simulation
    from genesis.engine.ipc.broad_phase import EFBroadPhase
    from genesis.utils.misc import qd_to_numpy

    import pipeline
    from narrow_ref import classify

    @qd.kernel
    def kernel_ef_predicates(
        mesh: qd.template(),
        surf_verts: qd.types.ndarray(),
        sub_pt: qd.types.ndarray(),
        sub_ee: qd.types.ndarray(),
        n_pt: int,
        n_ee: int,
        pt_kept: qd.types.ndarray(),
        ee_kept: qd.types.ndarray(),
    ):
        """CGQ's PT and EE predicates and distance rule over the sub-pairs (see pipeline.PTQuery / EEQuery)."""
        for i_r in range(n_pt):
            i_v = surf_verts[sub_pt[i_r][1]]
            tri = mesh.faces[sub_pt[i_r][2]]
            is_kept = mesh.is_body_pair_kept(mesh.verts_body[i_v], mesh.body_of_face(tri))
            if i_v == tri[0] or i_v == tri[1] or i_v == tri[2]:
                is_kept = False
            if qd.static(mesh.n_ce > 0):
                for k in qd.static(range(3)):
                    if not mesh.is_enabled(mesh.mask_pt, i_v, tri[k]):
                        is_kept = False
            if is_kept:
                is_kept = mesh.is_pt_within_reach(i_v, tri)
            pt_kept[i_r] = qd.select(is_kept, 1, 0)
        for i_r in range(n_ee):
            ea = mesh.edges[sub_ee[i_r][1]]
            eb = mesh.edges[sub_ee[i_r][2]]
            is_kept = mesh.is_body_pair_kept(mesh.body_of_edge(ea), mesh.body_of_edge(eb))
            if ea[0] == eb[0] or ea[0] == eb[1] or ea[1] == eb[0] or ea[1] == eb[1]:
                is_kept = False
            if qd.static(mesh.n_ce > 0):
                for i, j in qd.static(qd.ndrange(2, 2)):
                    if not mesh.is_enabled(mesh.mask_ee, ea[i], eb[j]):
                        is_kept = False
            if is_kept:
                is_kept = mesh.is_ee_within_reach(ea, eb)
            ee_kept[i_r] = qd.select(is_kept, 1, 0)

    with np.load(args.data / args.scene / "frames.npz") as frames, np.load(args.data / args.scene / "gt.npz") as gt:
        frames = {key: frames[key] for key in frames.files}
        gt = {key: gt[key] for key in gt.files}
    mesh = pipeline.Mesh(frames)
    surf_verts = frames["surf_verts"]
    faces, edges = frames["surf_faces"], frames["surf_edges"]
    n_surf = len(surf_verts)
    slots = np.full(int(frames["n_verts"]), -1, dtype=np.int64)
    slots[surf_verts] = np.arange(n_surf)
    body_id = frames["body_id"][surf_verts]
    n_bodies = len(frames["self_collision"])

    def ndarray(values: np.ndarray, dtype, n: int = 0):
        array = qd.ndarray(qd.types.vector(n, dtype) if n else dtype, shape=values.shape[: values.ndim - (n > 0)])
        array.from_numpy(values)
        return array

    edges_local = ndarray(slots[edges].astype(np.int32), qd.i32, 2)
    faces_local = ndarray(slots[faces].astype(np.int32), qd.i32, 3)
    thickness = ndarray(frames["thicknesses"][surf_verts], qd.f64)
    d_hat = ndarray(frames["d_hats"][surf_verts], qd.f64)
    is_rigid = ndarray(np.zeros(n_surf, dtype=np.int32), qd.i32)
    surf_verts_qd = ndarray(surf_verts.astype(np.int32), qd.i32)
    replaced = []
    broad_phase = EFBroadPhase(
        1,
        edges_local,
        faces_local,
        thickness,
        d_hat,
        np.where(body_id >= 0, body_id, n_bodies),
        is_rigid,
        args.capacity,
        replaced.append,
    )
    positions = qd.ndarray(qd.types.vector(3, qd.f64), shape=(n_surf, 1))
    displacements = qd.ndarray(qd.types.vector(3, qd.f64), shape=(n_surf, 1))
    pivots = qd.ndarray(qd.types.vector(3, qd.f64), shape=(n_surf, 1))
    rotations = qd.ndarray(qd.types.vector(3, qd.f64), shape=(n_surf, 1))
    rotations.from_numpy(np.broadcast_to(np.array([1.0, 0.0, 0.0]), (n_surf, 1, 3)).copy())

    device = "cuda" if args.backend == "gpu" else "cpu"
    d_hats = torch.as_tensor(frames["d_hats"], device=device)
    thicknesses = torch.as_tensor(frames["thicknesses"], device=device)
    rows = []
    for i_f, native_frame in enumerate(gt["native_frames"].tolist()):
        mesh.pos.from_numpy(frames["positions"][i_f])
        mesh.end.from_numpy(frames["trajectory_end_positions"][i_f])
        mesh.inflation.from_numpy(frames["path_inflation"][i_f])
        mesh.path_speed.from_numpy(frames["path_speed"][i_f])
        mesh.path_kind.from_numpy(frames["path_kind"][i_f])
        pos = frames["positions"][i_f][surf_verts]
        end = frames["trajectory_end_positions"][i_f][surf_verts]
        inflation = frames["path_inflation"][i_f][surf_verts]
        positions.from_numpy(pos[:, None])
        displacements.from_numpy((end - pos)[:, None])
        # A bulge r |w|^2 / 8 of lever r = 8 * inflation and rotation |w| = 1 is the path inflation exactly
        pivots.from_numpy((pos - np.column_stack([8.0 * inflation, np.zeros((n_surf, 2))]))[:, None])
        n_runs = 0
        is_complete = False
        while not is_complete:
            broad_phase.run(positions, displacements, pivots, rotations, is_screw=True, is_swept=True, is_expanded=True)
            counts = [int(qd_to_numpy(broad_phase.groups_hits.count)), int(qd_to_numpy(broad_phase.hits.count))]
            counts += [int(count) for count in qd_to_numpy(broad_phase.sub_pairs.counts)]
            is_complete = not broad_phase.grow(*counts)
            n_runs += 1
        n_groups_hits, n_hits, n_pt, n_ee = counts
        sub_pt = qd_to_numpy(broad_phase.sub_pairs.pt)[:n_pt]
        sub_ee = qd_to_numpy(broad_phase.sub_pairs.ee)[:n_ee]
        pt_kept = qd.ndarray(qd.i32, shape=max(n_pt, 1))
        ee_kept = qd.ndarray(qd.i32, shape=max(n_ee, 1))
        kernel_ef_predicates(
            mesh, surf_verts_qd, broad_phase.sub_pairs.pt, broad_phase.sub_pairs.ee, n_pt, n_ee, pt_kept, ee_kept
        )
        pt = sub_pt[qd_to_numpy(pt_kept)[:n_pt] == 1]
        ee = sub_ee[qd_to_numpy(ee_kept)[:n_ee] == 1]
        stencils = {
            "pt": np.column_stack([surf_verts[pt[:, 1]], faces[pt[:, 2]]]),
            "ee": np.column_stack([edges[ee[:, 1]], edges[ee[:, 2]]]),
        }
        positions_t = torch.as_tensor(frames["positions"][i_f], device=device)
        row = {
            "frame": native_frame,
            "runs": n_runs,
            "groups_hits": n_groups_hits,
            "hits": n_hits,
            "sub_pt": n_pt,
            "sub_ee": n_ee,
        }
        for ch, stencil in stencils.items():
            flag, _, active = classify(ch, torch.as_tensor(stencil, device=device).long(), positions_t, d_hats, thicknesses)
            flag, active = flag.cpu().numpy(), active.cpu().numpy()
            gt_stencil = pipeline.frame_slice(gt, f"{ch}_stencil", i_f)
            gt_active = pipeline.frame_slice(gt, f"{ch}_active", i_f).astype(bool)
            gt_flag = pipeline.frame_slice(gt, f"{ch}_flag", i_f).astype(np.int64)
            only_ef, only_cgq = pipeline.set_diff_count(
                np.column_stack([stencil[active], flag[active]]), np.column_stack([gt_stencil[gt_active], gt_flag[gt_active]])
            )
            cand_only_ef, cand_only_cgq = pipeline.set_diff_count(stencil, gt_stencil)
            row[ch] = {
                "candidates_ef": len(stencil),
                "candidates_cgq": len(gt_stencil),
                "candidates_only_ef": cand_only_ef,
                "candidates_only_cgq": cand_only_cgq,
                "duplicates_ef": len(stencil) - len(np.unique(pipeline.rows_view(stencil))),
                "active_ef": int(active.sum()),
                "active_cgq": int(gt_active.sum()),
                "active_only_ef": only_ef,
                "active_only_cgq": only_cgq,
            }
        rows.append(row)
        print(
            f"[{args.scene}] frame {native_frame}: runs {n_runs} hits {n_hits} sub {n_pt}/{n_ee} "
            + " ".join(
                f"{ch}: cand {row[ch]['candidates_ef']}/{row[ch]['candidates_cgq']} "
                f"(+{row[ch]['candidates_only_ef']}/-{row[ch]['candidates_only_cgq']}, dup {row[ch]['duplicates_ef']}) "
                f"active {row[ch]['active_ef']}/{row[ch]['active_cgq']} "
                f"(+{row[ch]['active_only_ef']}/-{row[ch]['active_only_cgq']})"
                for ch in stencils
            ),
            flush=True,
        )
    is_pass = all(
        row[ch][key] == 0
        for row in rows
        for ch in ("pt", "ee")
        for key in ("active_only_ef", "active_only_cgq", "candidates_only_ef", "candidates_only_cgq", "duplicates_ef")
    )
    print(f"[{args.scene}] PASS (candidates and active pairs identical to CGQ on every frame): {is_pass}")
    out = HERE / "results" / f"{args.scene}_ef_{args.backend}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"scene": args.scene, "backend": args.backend, "pass": is_pass, "frames": rows}, indent=1))
    if not is_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
