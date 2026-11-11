"""Measure the fewest steps after which each assertion of the cloth contact tests of test_fem.py holds.

The scenes are those of the tests, with their default options. Each step logs the quantities the tests assert on, so that
the step budget of a test is the last step at which one of its assertions fails, padded as the testing guidelines say.
"""

import argparse
import json
from pathlib import Path

import igl
import numpy as np
import torch

import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="cpu")
parser.add_argument("--data", default=str(Path(__file__).parent / "data"))
parser.add_argument("--results", default=str(Path(__file__).parent / "results"))
args = parser.parse_args()

gs.init(backend=getattr(gs, args.backend), precision="64", logging_level="warning")

n = 8
grid_verts = np.array([[i / n - 0.5, j / n - 0.5, 0.0] for j in range(n + 1) for i in range(n + 1)])
grid_faces = []
for j in range(n):
    for i in range(n):
        v00 = j * (n + 1) + i
        grid_faces += [[v00, v00 + 1, v00 + n + 1], [v00 + 1, v00 + n + 2, v00 + n + 1]]
grid_path = str(Path(args.data) / "horizons_grid.obj")
igl.write_triangle_mesh(grid_path, grid_verts, np.array(grid_faces))

n_length, n_width, flat_length, radius = 24, 6, 0.3, 0.03
fold_path_2d = []
for s in np.linspace(0.0, 2.0 * flat_length + np.pi * radius, n_length + 1):
    if s <= flat_length:
        fold_path_2d.append((s - flat_length, 0.1))
    elif s <= flat_length + np.pi * radius:
        angle = (s - flat_length) / radius - 0.5 * np.pi
        fold_path_2d.append((radius * np.cos(angle), 0.1 + radius + radius * np.sin(angle)))
    else:
        fold_path_2d.append((flat_length + np.pi * radius - s, 0.1 + 2.0 * radius))
fold_verts = np.array([[x, (j / n_width - 0.5) * 0.3, z] for j in range(n_width + 1) for x, z in fold_path_2d])
fold_faces = []
for j in range(n_width):
    for i in range(n_length):
        v00 = j * (n_length + 1) + i
        fold_faces += [[v00, v00 + 1, v00 + n_length + 1], [v00 + 1, v00 + n_length + 2, v00 + n_length + 1]]
fold_path = str(Path(args.data) / "horizons_fold.obj")
igl.write_triangle_mesh(fold_path, fold_verts, np.array(fold_faces))


def add_crossed_sheets(scene):
    return [
        scene.add_entity(
            morph=gs.morphs.Mesh(file=grid_path, scale=0.4, pos=(0.0, 0.0, z), euler=euler),
            material=gs.materials.FEM.Cloth(),
        )
        for euler, z in (((90.0, 0.0, 0.0), 0.23), ((90.0, 0.0, 90.0), 0.68))
    ]


summary = {"backend": args.backend}

# test_cloth_contact_between_cloths
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), ipc_options=gs.options.IPCOptions(friction_eps_v=1e-2))
scene.add_entity(morph=gs.morphs.Plane())
sheets = add_crossed_sheets(scene)
scene.build(n_envs=2)
crossed_failed = []
crossed_speed = []
for i_step in range(160):
    scene.step()
    speed = max(float(sheet.get_verts_vel().norm(dim=-1).max()) for sheet in sheets)
    is_above_ground = all(bool((sheet.get_verts_pos()[..., 2] > 1e-3).all()) for sheet in sheets)
    is_stacked = bool(
        (sheets[1].get_verts_pos()[..., 2].max(dim=-1).values > sheets[0].get_verts_pos()[..., 2].max(dim=-1).values).all()
    )
    crossed_speed.append(speed)
    if not (speed < 1e-2 and is_above_ground and is_stacked):
        crossed_failed.append(i_step + 1)
summary["crossed"] = {"last_failing_step": max(crossed_failed, default=0), "speed": crossed_speed}
print(f"crossed: last failing step {summary['crossed']['last_failing_step']}", flush=True)
print("crossed speed every 10 steps:", [f"{v:.2e}" for v in crossed_speed[9::10]], flush=True)
scene.destroy()

# test_cloth_self_contact
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01))
scene.add_entity(morph=gs.morphs.Plane())
cloth = scene.add_entity(morph=gs.morphs.Mesh(file=fold_path), material=gs.materials.FEM.Cloth())
scene.build()
rows_start = np.arange(n_width + 1)[:, None] * (n_length + 1)
lower_idx = (rows_start + np.arange(10)).reshape(-1)
upper_idx = (rows_start + n_length - np.arange(10)).reshape(-1)
gaps_min = []
for _ in range(100):
    scene.step()
    verts_z = cloth.get_verts_pos()[..., 2]
    gaps_min.append(float((verts_z[upper_idx] - verts_z[lower_idx]).min()))
summary["self"] = {
    "first_step_gap_below_0.02": next((i + 1 for i, gap in enumerate(gaps_min) if gap < 0.02), None),
    "gaps_min": gaps_min,
}
print(f"self: first step with gap < 0.02: {summary['self']['first_step_gap_below_0.02']}", flush=True)
print("self gap every 5 steps:", [f"{v:.3e}" for v in gaps_min[4::5]], flush=True)
scene.destroy()

# test_cloth_contact_capacity_growth
sheets_pos = []
pairs_by_capacity = []
for capacity in (1, 1000):
    scene = gs.Scene(ipc_options=gs.options.IPCOptions(init_collision_pair_capacity=capacity))
    scene.add_entity(morph=gs.morphs.Plane())
    sheets = add_crossed_sheets(scene)
    scene.build()
    n_pairs, positions = [], []
    for _ in range(50):
        scene.step()
        n_pairs.append(scene.sim.newton_solver.ipc_contact.n_pairs)
        positions.append(torch.cat([sheet.get_verts_pos() for sheet in sheets]))
    pairs_by_capacity.append(n_pairs)
    sheets_pos.append(torch.stack(positions))
    scene.destroy()
diff = (sheets_pos[0] - sheets_pos[1]).abs().reshape(50, -1).max(dim=-1).values
summary["capacity"] = {
    "first_step_more_than_one_pair": next((i + 1 for i, (pt, ee) in enumerate(pairs_by_capacity[0]) if max(pt, ee) > 1), None),
    "n_pairs": pairs_by_capacity[0],
    "diff": diff.tolist(),
}
print(f"capacity: first step with more than one pair: {summary['capacity']['first_step_more_than_one_pair']}", flush=True)
print("capacity pairs every 5 steps:", pairs_by_capacity[0][4::5], flush=True)
print("capacity diff every 5 steps:", [f"{v:.2e}" for v in diff[4::5].tolist()], flush=True)
Path(args.results, f"test_horizons_{args.backend}.json").write_text(json.dumps(summary, indent=1))
