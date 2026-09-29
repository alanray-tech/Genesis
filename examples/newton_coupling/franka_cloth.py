import argparse
import math
import time
from pathlib import Path

import quadrants as qd

import genesis as gs
from genesis.engine.systems import ContactTabular, build_scene_engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-n", "--steps", type=int, default=200)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="64", logging_level="debug")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_fem=False),
        viewer_options=gs.options.ViewerOptions(
            res=(960, 640),
            camera_pos=(2.0, -2.0, 1.4),
            camera_lookat=(0.4, 0.0, 0.5),
            camera_fov=35,
        ),
        show_viewer=args.vis,
    )

    franka = scene.add_entity(
        morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        vis_mode="collision",
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(Path(__file__).parent / "assets" / "qcloth_grid.obj"),
            pos=(0.9, 0.0, 0.7),
        ),
        material=gs.materials.FEM.QCloth(
            E=2e4,
            shear_modulus=2e3,
            rho=200.0,
            thickness=1e-3,
            bending_youngs_modulus=1e6,
        ),
        surface=gs.surfaces.Default(color=(0.25, 0.45, 0.9, 1.0)),
    )
    scene.build(compile_kernels=False)
    cloth.set_vertex_constraints([20, 24])

    contact_tabular = ContactTabular()
    contact_tabular.default_model(friction_rate=1.0, resistance=1e4)
    engine = build_scene_engine(
        scene,
        contact_config={
            "contact/d_hat": 1e-3,
            "contact/init_collision_pair_capacity": 20_000,
        },
        contact_tabular=contact_tabular,
    )
    reference_qpos = franka.get_qpos()
    motor_dofs = list(range(7))

    frame = 0
    while (args.vis and scene.viewer.is_alive()) or (not args.vis and frame < args.steps):
        target = reference_qpos[:7].clone()
        target[0] += 0.15 * math.sin(frame * 0.03)
        franka.control_dofs_position(target, motor_dofs)
        engine.step()
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
