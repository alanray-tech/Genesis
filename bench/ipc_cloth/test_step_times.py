"""Time the build and every step of the crossed-sheets test scene of test_fem.py, for two initial pair capacities."""

import argparse
import cProfile
import pstats
import time
from collections import defaultdict
from pathlib import Path

import igl
import numpy as np

from quadrants.lang.kernel import Kernel

import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="gpu")
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--data", default=str(Path(__file__).parent / "data"))
parser.add_argument("--capacities", type=int, nargs="+", default=[1, 1000, 1])
parser.add_argument("--profile-build", action="store_true")
parser.add_argument("--kernel-times", action="store_true", help="time the materialization of every kernel")
args = parser.parse_args()

gs.init(backend=getattr(gs, args.backend), precision="64", logging_level="warning")

materialize_times: dict[str, float] = defaultdict(float)
materialize = Kernel.materialize


def timed_materialize(self, key, py_args, arg_features=None):
    t_start = time.perf_counter()
    materialize(self, key, py_args, arg_features)
    materialize_times[f"{self.func.__module__}.{self.func.__name__}"] += time.perf_counter() - t_start


if args.kernel_times:
    Kernel.materialize = timed_materialize

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
    if args.kernel_times:
        modules_time: dict[str, float] = defaultdict(float)
        for name, duration in materialize_times.items():
            modules_time[name.rsplit(".", 1)[0]] += duration
        print(f"materialization {sum(materialize_times.values()):.1f} s over {len(materialize_times)} kernels")
        for module, duration in sorted(modules_time.items(), key=lambda item: -item[1]):
            print(f"  {duration:7.2f} s  {module}")
        for name, duration in sorted(materialize_times.items(), key=lambda item: -item[1])[:40]:
            print(f"    {duration:7.2f} s  {name}")
        materialize_times.clear()
    scene.destroy()
