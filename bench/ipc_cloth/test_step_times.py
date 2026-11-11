"""Time the build and every step of the crossed-sheets test scene of test_fem.py, for two initial pair capacities."""

import argparse
import cProfile
import pstats
import time
from pathlib import Path

import igl
import numpy as np

import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="gpu")
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--data", default=str(Path(__file__).parent / "data"))
parser.add_argument("--capacities", type=int, nargs="+", default=[1, 1000, 1])
parser.add_argument("--profile-build", action="store_true")
args = parser.parse_args()

gs.init(backend=getattr(gs, args.backend), precision="64", logging_level="warning")

n = 8
grid_verts = np.array([[i / n - 0.5, j / n - 0.5, 0.0] for j in range(n + 1) for i in range(n + 1)])
grid_faces = []
for j in range(n):
    for i in range(n):
        v00 = j * (n + 1) + i
        grid_faces += [[v00, v00 + 1, v00 + n + 1], [v00 + 1, v00 + n + 2, v00 + n + 1]]
grid_path = str(Path(args.data) / "step_times_grid.obj")
igl.write_triangle_mesh(grid_path, grid_verts, np.array(grid_faces))

for capacity in args.capacities:
    t_start = time.perf_counter()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        ipc_options=gs.options.IPCOptions(init_collision_pair_capacity=capacity),
    )
    scene.add_entity(morph=gs.morphs.Plane())
    for euler, z in (((90.0, 0.0, 0.0), 0.23), ((90.0, 0.0, 90.0), 0.68)):
        scene.add_entity(
            morph=gs.morphs.Mesh(file=grid_path, scale=0.4, pos=(0.0, 0.0, z), euler=euler),
            material=gs.materials.FEM.Cloth(),
        )
    profiler = cProfile.Profile()
    if args.profile_build:
        profiler.enable()
    scene.build(n_envs=2)
    t_build = time.perf_counter() - t_start
    if args.profile_build:
        profiler.disable()
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(45)
    steps_time, capacities = [], []
    newton = scene.sim.newton_solver
    for _ in range(args.steps):
        t_step = time.perf_counter()
        scene.step()
        steps_time.append(time.perf_counter() - t_step)
        capacities.append(newton.ipc_contact.pairs.pt.shape[0])
    steps_time = np.array(steps_time)
    n_growths = len(set(capacities))
    slow_steps = np.flatnonzero(steps_time > 5.0 * np.median(steps_time)) + 1
    print(
        f"capacity {capacity}: build {t_build:.1f} s, steps total {steps_time.sum():.1f} s, median {np.median(steps_time) * 1e3:.1f} ms,"
        f" max {steps_time.max():.2f} s at step {int(steps_time.argmax()) + 1}, distinct capacities {n_growths}"
        f" (final {capacities[-1]}), slow steps {slow_steps.tolist()[:20]} totalling {steps_time[slow_steps - 1].sum():.1f} s",
        flush=True,
    )
    scene.destroy()
