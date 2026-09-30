import argparse
import time
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.systems import build_scene_engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-n", "--steps", type=int, default=400)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="64", logging_level="debug")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_fem=False),
        viewer_options=gs.options.ViewerOptions(
            res=(960, 640),
            camera_pos=(1.4, -1.4, 0.9),
            camera_lookat=(0.0, 0.0, 0.25),
            camera_fov=35,
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Default(color=(0.35, 0.35, 0.35, 1.0)),
    )

    cloth_asset = str(Path(__file__).parent / "assets" / "qcloth_grid.obj")
    material = gs.materials.FEM.QCloth(
        E=2e4,
        shear_modulus=2e3,
        rho=200.0,
        thickness=1e-3,
        bending_youngs_modulus=1e6,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=cloth_asset,
            pos=(0.0, 0.0, 0.23),
            euler=(90.0, 0.0, 0.0),
        ),
        material=material,
        surface=gs.surfaces.Default(color=(0.2, 0.45, 0.9, 1.0)),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=cloth_asset,
            pos=(0.0, 0.0, 0.68),
            euler=(90.0, 0.0, 90.0),
        ),
        material=material,
        surface=gs.surfaces.Default(color=(0.9, 0.35, 0.2, 1.0)),
    )
    scene.build(compile_kernels=False)

    engine = build_scene_engine(
        scene,
        contact_config={},
        halfplanes=(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        ),
    )

    frame = 0
    while (args.vis and scene.viewer.is_alive()) or (not args.vis and frame < args.steps):
        try:
            engine.step()
        except RuntimeError as error:
            raise RuntimeError(f"crossed_cloth_drop failed at frame {frame}") from error
        if args.vis:
            scene.viewer.update(force=True)
            time.sleep(scene.sim.substep_dt)
        frame += 1

    qd.sync()
    print(
        "iterations:",
        engine.get_newton_iters(),
        engine.get_max_pcg_iters(),
        engine.get_max_ls_iters(),
    )


if __name__ == "__main__":
    main()
