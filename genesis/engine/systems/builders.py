from __future__ import annotations

from genesis.engine.solvers.rigid.rigid_solver import RigidSolver

from .global_layout import GlobalLayout
from .global_linear_system import GlobalLinearSystem
from .linear_pcg import LinearPCG
from .rigid_system import RigidSystem
from .sim_config import SimConfig
from .sim_engine import SimEngine


def build_rigid_engine(rigid_solver: RigidSolver) -> SimEngine:
    """Build a global Newton runtime containing one Rigid physics system."""
    layout = GlobalLayout()
    dof_range = layout.allocate(rigid_solver.n_dofs * rigid_solver._B)
    linear_system = GlobalLinearSystem(layout.n_block_rows)
    pcg = LinearPCG(layout.n_dofs)
    config = SimConfig(
        h=rigid_solver._substep_dt,
        max_newton=rigid_solver._options.iterations,
        max_pcg=rigid_solver._options.iterations,
        max_line_search=rigid_solver._options.ls_iterations,
        newton_tolerance=rigid_solver._options.tolerance,
        pcg_tolerance=rigid_solver._options.tolerance,
    )
    rigid_system = RigidSystem(rigid_solver, dof_range)

    engine = SimEngine()
    engine.add_system(layout)
    engine.add_system(config)
    engine.add_system(linear_system)
    engine.add_system(pcg)
    engine.add_system(rigid_system)
    engine.initialize()
    return engine
