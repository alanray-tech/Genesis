"""Frozen QCloth contact against a fixed triangular obstacle."""

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

import genesis as gs
from genesis.engine.systems import ContactTabular, build_scene_engine
from genesis.utils.misc import qd_to_numpy

THICKNESS = 1.0e-3
HEIGHT = 2.5e-3
MOVING_VERTICES = np.array(
    [
        [-0.02, HEIGHT, -0.01],
        [0.02, HEIGHT, -0.01],
        [0.00, HEIGHT, 0.02],
    ],
    dtype=np.float64,
)
FIXED_VERTICES = np.array(
    [
        [-0.05, 0.0, -0.04],
        [0.05, 0.0, -0.04],
        [0.00, 0.0, 0.06],
    ],
    dtype=np.float64,
)


def triangle_asset(name: str, vertices: np.ndarray) -> Path:
    path = Path(tempfile.gettempdir()) / name
    lines = [f"v {x:.17g} {y:.17g} {z:.17g}" for x, y, z in vertices]
    lines.append("f 1 2 3")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def candidate_stencils(engine) -> tuple[np.ndarray, np.ndarray]:
    contact = engine.contact
    surface = engine.global_surface_manager
    surf_verts = qd_to_numpy(surface.surf_verts)
    surf_triangles = qd_to_numpy(surface.surf_triangles)
    surf_edges = qd_to_numpy(surface.surf_edges)

    n_pt = int(qd_to_numpy(contact.n_pairs_pt))
    pairs_pt = qd_to_numpy(contact.pairs_pt)[:n_pt]
    pt = np.column_stack(
        (
            surf_verts[pairs_pt[:, 0]],
            surf_triangles[pairs_pt[:, 1]],
        )
    ).astype(np.int32, copy=False)
    if len(pt):
        pt = pt[np.lexsort(tuple(pt[:, i] for i in reversed(range(pt.shape[1]))))]

    n_ee = int(qd_to_numpy(contact.n_pairs_ee))
    pairs_ee = qd_to_numpy(contact.pairs_ee)[:n_ee]
    ee = np.column_stack(
        (
            surf_edges[pairs_ee[:, 0]],
            surf_edges[pairs_ee[:, 1]],
        )
    ).astype(np.int32, copy=False)
    if len(ee):
        ee = ee[np.lexsort(tuple(ee[:, i] for i in reversed(range(ee.shape[1]))))]
    return pt, ee


def aggregate_contact(engine) -> tuple[np.ndarray, np.ndarray]:
    contact = engine.contact
    n_vertices = int(qd_to_numpy(engine.global_vertex_manager.n_verts))
    gradient = np.zeros((n_vertices, 3), dtype=np.float64)
    n_gradient = int(qd_to_numpy(contact.n_unique_doublets))
    gradient_ids = qd_to_numpy(contact.unique_doublet_vertices)[:n_gradient]
    gradient[gradient_ids] = qd_to_numpy(contact.unique_doublet_gradients)[:n_gradient]

    hessian = np.zeros((n_vertices * 3, n_vertices * 3), dtype=np.float64)
    n_hessian = int(qd_to_numpy(contact.n_unique_triplets))
    rows = qd_to_numpy(contact.unique_triplet_rows)[:n_hessian]
    columns = qd_to_numpy(contact.unique_triplet_cols)[:n_hessian]
    values = qd_to_numpy(contact.unique_triplet_values)[:n_hessian]
    for row, column, block in zip(rows, columns, values, strict=True):
        row_slice = slice(int(row) * 3, int(row) * 3 + 3)
        column_slice = slice(int(column) * 3, int(column) * 3 + 3)
        hessian[row_slice, column_slice] += block
        if row != column:
            hessian[column_slice, row_slice] += block.T
    return gradient, hessian


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-output", type=Path, default=None)
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
    moving = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(
                triangle_asset(
                    "genesis_qcloth_contact_moving.obj",
                    MOVING_VERTICES,
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
    fixed = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(
                triangle_asset(
                    "genesis_qcloth_contact_fixed.obj",
                    FIXED_VERTICES,
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
    fixed.set_vertex_constraints([0, 1, 2])

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
    )
    capture_contact(engine)
    qd.sync()

    pt, ee = candidate_stencils(engine)
    gradient, hessian = aggregate_contact(engine)
    contact = engine.contact
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        positions=qd_to_numpy(engine.global_vertex_manager.positions),
        pt_stencil=pt,
        ee_stencil=ee,
        candidate_counts=np.array(
            [
                int(qd_to_numpy(contact.n_pairs_pt)),
                int(qd_to_numpy(contact.n_pairs_ee)),
                int(qd_to_numpy(contact.n_pairs_pe)),
                int(qd_to_numpy(contact.n_pairs_pp)),
                int(qd_to_numpy(contact.n_pairs_ph)),
            ],
            dtype=np.int32,
        ),
        active_count=qd_to_numpy(contact.n_active_pairs),
        aggregate_grad=gradient,
        aggregate_hess=hessian,
        energy=qd_to_numpy(contact.contact_energy_value),
        ccd_alpha=qd_to_numpy(contact.ccd_alpha),
    )

    if args.system_output is not None:
        initial_positions = qd_to_numpy(engine.fem.x)
        engine.sim_config.max_newton_iter.from_numpy(np.array(1, dtype=np.int64))
        try:
            engine.step()
        except RuntimeError:
            pass
        qd.sync()
        n_blocks = int(qd_to_numpy(engine.global_linear_system.bcoo_nnz))
        args.system_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.system_output,
            row=qd_to_numpy(engine.global_linear_system.bcoo_row)[:n_blocks],
            col=qd_to_numpy(engine.global_linear_system.bcoo_col)[:n_blocks],
            val=qd_to_numpy(engine.global_linear_system.bcoo_val)[: n_blocks * 9].reshape(n_blocks, 3, 3),
            rhs=qd_to_numpy(engine.global_linear_system.b_rhs),
            solution=qd_to_numpy(engine.global_linear_system.x_sol),
            residual=qd_to_numpy(engine.pcg_solver.linear_pcg.residual),
            dx=qd_to_numpy(engine.fem.dx),
            initial_positions=initial_positions,
            final_positions=qd_to_numpy(engine.fem.x),
        )


if __name__ == "__main__":
    main()
