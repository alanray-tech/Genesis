"""Contact-free Genesis QCloth trajectory for CGQ conformance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import quadrants as qd
from cloth_grid_asset import cloth_grid_asset

import genesis as gs
from genesis.engine.systems import build_scene_engine
from genesis.utils.misc import qd_to_numpy

DT = 0.01


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pcg-tol", type=float, default=1e-5)
    parser.add_argument("--system-output", type=Path, default=None)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -9.8),
        ),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_fem=False),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(cloth_grid_asset(resolution=25, size=0.24)),
            pos=(0.0, 0.0, 0.7),
        ),
        material=gs.materials.FEM.QCloth(
            E=2e4,
            shear_modulus=2e3,
            rho=200.0,
            thickness=1e-3,
            bending_youngs_modulus=3e3,
        ),
    )
    scene.build(compile_kernels=False)
    engine = build_scene_engine(
        scene,
        contact_config={
            "contact/enable": 0,
            "linear_system/tol_rate": args.pcg_tol,
        },
    )
    if args.system_output is not None:
        engine.sim_config.max_newton_iter.from_numpy(np.array(1, dtype=np.int64))

    positions = [qd_to_numpy(engine.fem.x).tolist()]
    newton = []
    total_pcg = []
    for _ in range(args.frames):
        try:
            engine.step()
        except RuntimeError:
            if args.system_output is None:
                raise
        qd.sync()
        positions.append(qd_to_numpy(engine.fem.x).tolist())
        newton.append(engine.get_newton_iters())
        total_pcg.append(engine.get_total_pcg_iters())
        if args.system_output is not None:
            break

    if args.system_output is not None:
        n_blocks = int(qd_to_numpy(engine.global_linear_system.bcoo_nnz))
        args.system_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.system_output,
            row=qd_to_numpy(engine.global_linear_system.bcoo_row)[:n_blocks],
            col=qd_to_numpy(engine.global_linear_system.bcoo_col)[:n_blocks],
            val=qd_to_numpy(engine.global_linear_system.bcoo_val)[: n_blocks * 9].reshape(n_blocks, 3, 3),
            rhs=qd_to_numpy(engine.global_linear_system.b_rhs),
            x_tilde=qd_to_numpy(engine.fem.x_tilde),
            masses=qd_to_numpy(engine.fem.masses),
            residual=qd_to_numpy(engine.pcg_solver.linear_pcg.residual),
            preconditioned_residual=qd_to_numpy(engine.pcg_solver.linear_pcg.preconditioned_residual),
            solution=qd_to_numpy(engine.global_linear_system.x_sol),
            dx=qd_to_numpy(engine.fem.dx),
            pcg_iterations=np.array(
                engine.get_total_pcg_iters(),
                dtype=np.int32,
            ),
        )

    result = {
        "implementation": "GenesisQCloth",
        "dt": DT,
        "frames": args.frames,
        "pcg_tol": args.pcg_tol,
        "positions": positions,
        "newton": newton,
        "total_pcg": total_pcg,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
