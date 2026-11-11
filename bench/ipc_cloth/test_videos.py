"""Record a video of the scenes of the cloth tests of tests/deformable/test_fem.py, offscreen with a Genesis camera.

Every scene takes the options, the step count and the viewpoint of its test. Batched scenes render their first
environment. The videos play at a quarter of the simulated time rate. The precision applies to every scene selected: run
the contact scenes in double precision, as their tests do, and the others in the default single precision.
"""

import argparse
from pathlib import Path

import igl
import numpy as np

import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("--precision", default="32")
parser.add_argument("--scenes", nargs="+", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--data", default=str(Path(__file__).parent / "data"))
args = parser.parse_args()

gs.init(backend=gs.gpu, precision=args.precision, logging_level="warning")
Path(args.out).mkdir(parents=True, exist_ok=True)

n = 8
grid_verts = np.array([[i / n - 0.5, j / n - 0.5, 0.0] for j in range(n + 1) for i in range(n + 1)])
grid_faces = []
for j in range(n):
    for i in range(n):
        v00 = j * (n + 1) + i
        grid_faces += [[v00, v00 + 1, v00 + n + 1], [v00 + 1, v00 + n + 2, v00 + n + 1]]
grid_path = str(Path(args.data) / "videos_grid.obj")
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
fold_path = str(Path(args.data) / "videos_fold.obj")
igl.write_triangle_mesh(fold_path, fold_verts, np.array(fold_faces))

for name in args.scenes:
    # The recording paces its frames by the realtime factor: at 25 frames per second, one frame per step of 0.01 s
    viewer_options = gs.options.ViewerOptions(realtime_factor=0.25)
    vis_options = gs.options.VisOptions(rendered_envs_idx=[0])
    if name == "pinned_vertices":
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.81)),
            viewer_options=viewer_options,
            vis_options=vis_options,
            show_viewer=False,
        )
        cloth = scene.add_entity(
            morph=gs.morphs.Mesh(file=grid_path, pos=(0.0, 0.0, 1.0)), material=gs.materials.FEM.Cloth()
        )
        camera = scene.add_camera(res=(1280, 720), pos=(3.5, 0.0, 0.6), lookat=(0.0, -0.5, 0.5), fov=40, GUI=False)
        scene.build(n_envs=2)
        cloth.set_vertex_constraints(verts_idx_local=[0, n])
        n_steps = 100
    elif name == "ground_contact":
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            ipc_options=gs.options.IPCOptions(d_hat=1e-2),
            viewer_options=viewer_options,
            vis_options=vis_options,
            show_viewer=False,
        )
        scene.add_entity(morph=gs.morphs.Plane())
        scene.add_entity(
            morph=gs.morphs.Mesh(file=grid_path, pos=(0.0, 0.0, 0.05)),
            material=gs.materials.FEM.Cloth(contact_thickness=1e-3),
        )
        camera = scene.add_camera(res=(1280, 720), pos=(1.5, -1.5, 0.5), lookat=(0.0, 0.0, 0.0), fov=40, GUI=False)
        scene.build(n_envs=2)
        n_steps = 50
    elif name == "contact_between_cloths":
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            newton_options=gs.options.NewtonOptions(velocity_tol=5e-2),
            ipc_options=gs.options.IPCOptions(init_collision_pair_capacity=1),
            viewer_options=viewer_options,
            vis_options=vis_options,
            show_viewer=False,
        )
        scene.add_entity(morph=gs.morphs.Plane())
        for euler, z in (((90.0, 0.0, 0.0), 0.23), ((90.0, 0.0, 90.0), 0.68)):
            scene.add_entity(
                morph=gs.morphs.Mesh(file=grid_path, scale=0.4, pos=(0.0, 0.0, z), euler=euler),
                material=gs.materials.FEM.Cloth(),
            )
        camera = scene.add_camera(res=(1280, 720), pos=(1.2, -1.2, 0.8), lookat=(0.0, 0.0, 0.4), fov=40, GUI=False)
        scene.build(n_envs=2)
        n_steps = 150
    elif name == "self_contact":
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            viewer_options=viewer_options,
            vis_options=vis_options,
            show_viewer=False,
        )
        scene.add_entity(morph=gs.morphs.Plane())
        scene.add_entity(morph=gs.morphs.Mesh(file=fold_path), material=gs.materials.FEM.Cloth())
        camera = scene.add_camera(res=(1280, 720), pos=(0.3, -1.0, 0.4), lookat=(-0.15, 0.0, 0.05), fov=40, GUI=False)
        scene.build()
        n_steps = 50
    else:
        raise ValueError(f"Unknown scene {name}")
    path = Path(args.out) / f"test_cloth_{name}.mp4"
    camera.start_recording(save_to_filename=str(path), fps=25)
    for _ in range(n_steps):
        scene.step()
    camera.stop_recording()
    # A batched scene records its rendered environment to a file of its own, named after it
    path_env = path.with_name(f"{path.stem}_0.mp4")
    if path_env.exists():
        path_env.replace(path)
    print(f"wrote {path} ({n_steps} steps)", flush=True)
    scene.destroy()
