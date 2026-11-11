"""Count the Newton iterations, line-search exhaustions and failures of the cloth scenes, at the default tolerance.

Every scene runs its whole trajectory with the default NewtonOptions, in the precision given. A line search is exhausted
when it reaches 'line_search_max_iterations' and keeps its last trial, a substep fails when it reaches 'max_iterations'
or produces values that are not finite.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("--precision", default="32")
parser.add_argument("--backend", default="gpu")
parser.add_argument("--results", default=str(Path(__file__).parent / "results"))
parser.add_argument("--data", default=str(Path(__file__).parent / "data"))
args = parser.parse_args()

# The solver modules read the types of the precision, set once Genesis is initialized
gs.init(backend=getattr(gs, args.backend), precision=args.precision, logging_level="warning")
import genesis.engine.solvers.newton_solver as newton_solver  # noqa: E402
from genesis.utils.misc import qd_to_numpy  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import scenes  # noqa: E402

line_search = {"exhausted": 0, "searches": 0}
kernel_end_iteration = newton_solver.kernel_end_iteration
max_line_search = gs.options.NewtonOptions().line_search_max_iterations


def counted_end_iteration(newton_envs, newton_loops, newton_info):
    """The end of a Newton iteration, counting the line searches of the active environments that reached their cap."""
    is_active = qd_to_numpy(newton_envs.is_active)
    n_trials = qd_to_numpy(newton_envs.n_line_search_iterations)
    line_search["searches"] += int(is_active.sum())
    line_search["exhausted"] += int((is_active & (n_trials >= max_line_search)).sum())
    kernel_end_iteration(newton_envs, newton_loops, newton_info)


newton_solver.kernel_end_iteration = counted_end_iteration

summary = {"precision": args.precision, "backend": args.backend, "scenes": {}}
selected = ("ground_flat", "ground_tilted", "crossed_drop", "multilayer", "self_fold")
for cloth_scene in scenes.make_scenes(10) + scenes.make_contact_scenes(10):
    if cloth_scene.name not in selected:
        continue
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=scenes.DT, gravity=scenes.GRAVITY),
        ipc_options=gs.options.IPCOptions(d_hat=scenes.D_HAT),
        show_viewer=False,
    )
    scene.add_entity(morph=gs.morphs.Plane(pos=scenes.GROUND_POS, normal=scenes.GROUND_NORMAL))
    for i_cloth, cloth in enumerate(cloth_scene.cloths):
        path = Path(args.data) / f"stats_{cloth_scene.name}_{i_cloth}.obj"
        path.write_text(trimesh.exchange.obj.export_obj(trimesh.Trimesh(cloth.verts, cloth.tris, process=False), digits=17))
        scene.add_entity(
            morph=gs.morphs.Mesh(file=str(path)),
            material=gs.materials.FEM.Cloth(
                E=scenes.YOUNGS_MODULUS,
                rho=scenes.MASS_DENSITY,
                thickness=scenes.THICKNESS,
                shear_modulus=scenes.SHEAR_MODULUS,
                bending=cloth_scene.bending,
                bending_E=scenes.BENDING_YOUNGS_MODULUS,
            ),
        )
    scene.build()
    newton = scene.sim.newton_solver
    line_search.update(exhausted=0, searches=0)
    iterations, n_failed_iterations, n_failed_nan, error = [], 0, 0, None
    for _ in range(cloth_scene.n_frames):
        try:
            scene.step()
        except gs.GenesisException as exception:
            error = str(exception).splitlines()[0]
            break
        iterations.append(int(qd_to_numpy(newton.newton_envs.n_iterations)[0]))
        failure = int(qd_to_numpy(newton.newton_envs.failure)[0])
        n_failed_iterations += failure == int(newton_solver.NewtonFailure.ITERATIONS)
        n_failed_nan += failure == int(newton_solver.NewtonFailure.NON_FINITE)
    result = {
        "substeps": len(iterations),
        "newton_iterations_median": float(np.median(iterations)) if iterations else None,
        "newton_iterations_max": int(max(iterations)) if iterations else None,
        "substeps_at_max_iterations": n_failed_iterations,
        "substeps_non_finite": n_failed_nan,
        "line_searches": line_search["searches"],
        "line_searches_exhausted": line_search["exhausted"],
        "error": error,
    }
    summary["scenes"][cloth_scene.name] = result
    print(f"{cloth_scene.name} fp{args.precision}: {json.dumps(result)}", flush=True)
    scene.destroy()
Path(args.results, f"newton_stats_{args.backend}_fp{args.precision}.json").write_text(json.dumps(summary, indent=1))
