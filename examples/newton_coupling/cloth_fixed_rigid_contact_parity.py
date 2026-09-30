"""QCloth contact against one fixed minimal-coordinate rigid tetrahedron."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import quadrants as qd
from cloth_halfplane_contact_parity import (
    D_HAT,
    DT,
    KAPPA,
    capture_contact,
)
from cloth_static_mesh_contact_parity import (
    aggregate_contact,
    candidate_stencils,
    triangle_asset,
)

import genesis as gs
from genesis.engine.systems import ContactTabular, build_scene_engine
from genesis.utils.misc import qd_to_numpy

THICKNESS = 1.0e-3
HEIGHT = 1.5e-3
CLOTH_VERTICES = np.array(
    [
        [-0.02, HEIGHT, -0.01],
        [0.02, HEIGHT, -0.01],
        [0.00, HEIGHT, 0.02],
    ],
    dtype=np.float64,
)
RIGID_VERTICES = np.array(
    [
        [-0.05, 0.0, -0.04],
        [0.05, 0.0, -0.04],
        [0.00, 0.0, 0.06],
        [0.00, -0.10, 0.0],
    ],
    dtype=np.float64,
)
RIGID_FACES = np.array(
    [
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ],
    dtype=np.int32,
)


def tetrahedron_asset() -> Path:
    path = Path(tempfile.gettempdir()) / "genesis_contact_fixed_tetra.obj"
    lines = [f"v {x:.17g} {y:.17g} {z:.17g}" for x, y, z in RIGID_VERTICES]
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in RIGID_FACES)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, 0.0),
        ),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_fem=False),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(tetrahedron_asset()),
            fixed=True,
            convexify=False,
            decimate=False,
            watertighten=None,
            align=False,
        ),
        vis_mode="collision",
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(
                triangle_asset(
                    "genesis_rigid_contact_cloth.obj",
                    CLOTH_VERTICES,
                )
            )
        ),
        material=gs.materials.FEM.QCloth(
            E=2.0e4,
            shear_modulus=2.0e3,
            rho=200.0,
            thickness=THICKNESS,
        ),
    )
    scene.build(compile_kernels=False)

    contact_tabular = ContactTabular()
    contact_tabular.default_model(
        friction_rate=0.0,
        resistance=KAPPA,
    )
    engine = build_scene_engine(
        scene,
        contact_config={
            "contact/d_hat": D_HAT,
            "contact/adaptive_kappa_mode": "off",
            "contact/intersection_check": 0,
            "linear_system/tol_rate": 1.0e-4,
        },
        contact_tabular=contact_tabular,
    )
    capture_contact(engine)
    qd.sync()

    pt, ee = candidate_stencils(engine)
    gradient, hessian = aggregate_contact(engine)
    contact = engine.contact
    payload = {
        "positions": qd_to_numpy(engine.global_vertex_manager.positions),
        "pt_stencil": pt,
        "ee_stencil": ee,
        "candidate_counts": np.array(
            [
                int(qd_to_numpy(contact.n_pairs_pt)),
                int(qd_to_numpy(contact.n_pairs_ee)),
                int(qd_to_numpy(contact.n_pairs_pe)),
                int(qd_to_numpy(contact.n_pairs_pp)),
                int(qd_to_numpy(contact.n_pairs_ph)),
            ],
            dtype=np.int32,
        ),
        "active_count": qd_to_numpy(contact.n_active_pairs),
        "aggregate_grad": gradient,
        "aggregate_hess": hessian,
        "energy": qd_to_numpy(contact.contact_energy_value),
        "ccd_alpha": qd_to_numpy(contact.ccd_alpha),
    }

    engine.step()
    payload.update(
        {
            "final_fem_positions": qd_to_numpy(engine.fem.x),
            "newton": np.array(
                engine.get_newton_iters(),
                dtype=np.int32,
            ),
            "total_pcg": np.array(
                engine.get_total_pcg_iters(),
                dtype=np.int32,
            ),
            "max_pcg": np.array(
                engine.get_max_pcg_iters(),
                dtype=np.int32,
            ),
            "line_search": np.array(
                engine.get_max_ls_iters(),
                dtype=np.int32,
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)


if __name__ == "__main__":
    main()
