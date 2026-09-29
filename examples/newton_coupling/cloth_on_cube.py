"""QCloth draping onto one fixed rigid-proxy cube."""

from __future__ import annotations

import argparse

import quadrants as qd

import genesis as gs
from genesis.engine.systems import ContactTabular, build_scene_engine
from genesis.utils.misc import qd_to_numpy

from cloth_grid_asset import cloth_grid_asset

CUBE_TOP = 0.08
CLOTH_RESOLUTION = 25
CLOTH_CENTER_VERTEX = (CLOTH_RESOLUTION * CLOTH_RESOLUTION) // 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument(
        "--pin-corners",
        action="store_true",
        help="Pin all four cloth corners",
    )
    parser.add_argument(
        "--ee-query",
        choices=("dual", "warp"),
        default="dual",
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="64", logging_level="info")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_fem=False),
        viewer_options=gs.options.ViewerOptions(
            res=(900, 700),
            camera_pos=(0.75, -0.75, 0.55),
            camera_lookat=(0.0, 0.0, 0.08),
            camera_fov=35,
        ),
        show_viewer=not args.no_gui,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.04),
            size=(0.08, 0.08, 0.08),
            fixed=True,
        ),
        vis_mode="collision",
        surface=gs.surfaces.Default(color=(0.75, 0.72, 0.62, 1.0)),
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(cloth_grid_asset(resolution=CLOTH_RESOLUTION)),
            pos=(0.0, 0.0, 0.14),
        ),
        material=gs.materials.FEM.QCloth(
            E=2e4,
            shear_modulus=2e3,
            rho=200.0,
            thickness=1e-3,
            bending_youngs_modulus=3e3,
        ),
        surface=gs.surfaces.Default(color=(0.25, 0.45, 0.90, 1.0)),
    )
    scene.build(compile_kernels=False)
    if args.pin_corners:
        cloth.set_vertex_constraints(
            [
                0,
                CLOTH_RESOLUTION - 1,
                CLOTH_RESOLUTION * (CLOTH_RESOLUTION - 1),
                CLOTH_RESOLUTION * CLOTH_RESOLUTION - 1,
            ]
        )

    contact_tabular = ContactTabular()
    contact_tabular.default_model(friction_rate=1.0, resistance=1e4)
    engine = build_scene_engine(
        scene,
        contact_config={
            "contact/d_hat": 1e-3,
            "contact/intersection_check": 1,
            "bvh/ee_query": args.ee_query,
        },
        contact_tabular=contact_tabular,
    )

    maximum_proxy_doublets = 0
    frame = 0
    while (
        (not args.no_gui and scene.viewer.is_alive())
        or (args.no_gui and frame < args.steps)
    ):
        engine.step()
        maximum_proxy_doublets = max(
            maximum_proxy_doublets,
            int(
                qd_to_numpy(
                    engine.rigid_contact_assemble.rigid_doublet_total
                )
            ),
        )
        if not args.no_gui:
            scene.viewer.update(force=True)
        frame += 1

    qd.sync()
    center_height = float(
        qd_to_numpy(engine.fem.x)[CLOTH_CENTER_VERTEX, 2]
    )
    print(
        f"center_height={center_height:.9f}, "
        f"cube_top={CUBE_TOP:.9f}, "
        f"max_proxy_doublets={maximum_proxy_doublets}, "
        "et_intersections=0"
    )
    if args.no_gui:
        if maximum_proxy_doublets == 0:
            raise RuntimeError("Cloth-cube proxy contact route was never active")
        if center_height <= CUBE_TOP:
            raise RuntimeError(
                "Cloth center penetrated the fixed cube "
                f"(z={center_height:.9f}, cube_top={CUBE_TOP:.9f})"
            )


if __name__ == "__main__":
    main()
