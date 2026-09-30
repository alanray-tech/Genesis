"""Frozen QCloth-halfplane contact state for CGQ conformance."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.systems import ContactTabular, build_scene_engine
from genesis.utils.misc import qd_to_numpy

DT = 0.01
N = 24
SIZE = 0.24
HEIGHT = 1.5e-3
D_HAT = 1.0e-3
KAPPA = 1.0e4


def cloth_asset() -> Path:
    path = Path(tempfile.gettempdir()) / "genesis_qcloth_halfplane_parity.obj"
    lines = []
    for row in range(N + 1):
        for column in range(N + 1):
            x = (column / N - 0.5) * SIZE
            z = (row / N - 0.5) * SIZE
            lines.append(f"v {x:.17g} {HEIGHT:.17g} {z:.17g}")
    for row in range(N):
        for column in range(N):
            lower_left = row * (N + 1) + column + 1
            lower_right = lower_left + 1
            upper_left = lower_left + N + 1
            upper_right = upper_left + 1
            lines.append(f"f {lower_left} {lower_right} {upper_left}")
            lines.append(f"f {lower_right} {upper_right} {upper_left}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@qd.kernel(graph=True, fastcache=True)
def capture_contact(engine: qd.template()):
    engine.fem.forward_global_vertices(engine.global_vertex_manager)
    engine.contact.reset_collision_counts()
    engine.contact.bvh_triangle_build()
    engine.contact.bvh_edge_build()
    engine.contact.trajectory_query()
    engine.contact.count_active()
    engine.contact.filter_assemble()
    engine.contact.sort_reduce()
    engine.contact.contact_energy()


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
        morph=gs.morphs.Mesh(file=str(cloth_asset())),
        material=gs.materials.FEM.QCloth(
            E=2.0e4,
            shear_modulus=2.0e3,
            rho=200.0,
            thickness=1.0e-3,
            bending_youngs_modulus=3.0e3,
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
        },
        contact_tabular=contact_tabular,
        halfplanes=(
            np.zeros((1, 3), dtype=np.float64),
            np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
        ),
    )
    capture_contact(engine)
    qd.sync()

    contact = engine.contact
    n_ph = int(qd_to_numpy(contact.n_pairs_ph))
    n_doublets = int(qd_to_numpy(contact.n_contact_doublets))
    n_triplets = int(qd_to_numpy(contact.n_contact_triplets))
    n_unique_doublets = int(qd_to_numpy(contact.n_unique_doublets))
    n_unique_triplets = int(qd_to_numpy(contact.n_unique_triplets))
    ph_pairs = qd_to_numpy(contact.pairs_ph)[:n_ph]
    surface_vertices = qd_to_numpy(engine.global_surface_manager.surf_verts)
    hp_off = int(qd_to_numpy(engine.fem.n_fem_verts))
    stencil = np.column_stack(
        (
            surface_vertices[ph_pairs[:, 0]],
            hp_off + ph_pairs[:, 1],
        )
    ).astype(np.int32, copy=False)
    order = np.lexsort((stencil[:, 1], stencil[:, 0]))
    stencil = stencil[order]

    raw_vertices = qd_to_numpy(contact.contact_doublet_vertices)[:n_doublets]
    raw_order = np.argsort(raw_vertices)
    raw_triplet_rows = qd_to_numpy(contact.contact_triplet_rows)[:n_triplets]
    raw_triplet_order = np.argsort(raw_triplet_rows)
    positions = np.concatenate(
        (
            qd_to_numpy(engine.global_vertex_manager.positions),
            np.zeros((1, 3), dtype=np.float64),
        ),
        axis=0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        positions=positions,
        hp_off=np.array(hp_off, dtype=np.int32),
        ccd_alpha=qd_to_numpy(contact.ccd_alpha),
        stencil=stencil,
        active=np.ones(n_ph, dtype=np.int32),
        d2=np.full(n_ph, HEIGHT * HEIGHT, dtype=np.float64),
        grad=qd_to_numpy(contact.contact_doublet_gradients)[:n_doublets][raw_order],
        hess=qd_to_numpy(contact.contact_triplet_values)[:n_triplets][raw_triplet_order],
        energy=qd_to_numpy(contact.contact_energy_value),
        candidate_counts=np.array(
            [
                int(qd_to_numpy(contact.n_pairs_pt)),
                int(qd_to_numpy(contact.n_pairs_ee)),
                int(qd_to_numpy(contact.n_pairs_pe)),
                int(qd_to_numpy(contact.n_pairs_pp)),
                n_ph,
            ],
            dtype=np.int32,
        ),
        active_count=qd_to_numpy(contact.n_active_pairs),
        assembly_counts=np.array(
            [
                n_doublets,
                n_triplets,
                n_unique_doublets,
                n_unique_triplets,
            ],
            dtype=np.int32,
        ),
    )


if __name__ == "__main__":
    main()
