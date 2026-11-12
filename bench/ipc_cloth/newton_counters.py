"""Per-step counters of the Newton solve of Genesis for the parity benches: Newton iterations, line-search trials and
preconditioned conjugate gradient (PCG) iterations of environment 0, the counts CGQ's solver reports per frame
(newton_iters, total_ls_iters, total_pcg_iters).

The counters wrap the kernels of genesis/engine/solvers/newton_solver.py that solve the Newton direction (the PCG) and
run the line search, which the solver looks up in its module at every call. Import after gs.init.
"""

import numpy as np

import genesis.engine.solvers.newton_solver as newton_solver
from genesis.utils.misc import qd_to_numpy

# A frame is flagged once Genesis needs more than this many times CGQ's Newton iterations
FLAG_RATIO = 1.5


def compare_counts(counts: list, reference) -> dict:
    """Lay the per-frame counts of Genesis (see NewtonCounters.take) beside CGQ's, with their sums and the frames where
    Genesis needs more than FLAG_RATIO times CGQ's Newton iterations."""
    genesis = np.array(counts).reshape(-1, 3)
    cgq = np.stack([reference[key][: len(genesis)] for key in ("newton_iters", "ls_iters", "pcg_iters")], axis=1)
    flagged = np.nonzero(genesis[:, 0] > FLAG_RATIO * cgq[:, 0])[0]
    return {
        "per_frame": {
            f"{side}_{name}": values[:, i].tolist()
            for side, values in (("genesis", genesis), ("cgq", cgq))
            for i, name in enumerate(("newton", "line_search", "pcg"))
        },
        "totals": {
            side: dict(zip(("newton", "line_search", "pcg"), values.sum(axis=0).tolist()))
            for side, values in (("genesis", genesis), ("cgq", cgq))
        },
        "flagged_frames": flagged.tolist(),
    }


class NewtonCounters:
    def __init__(self, pcg_max_iterations: int) -> None:
        self.pcg_max_iterations = pcg_max_iterations
        self.n_line_search_trials = 0
        self.n_pcg_iterations = 0
        newton_direction = newton_solver.kernel_newton_direction
        line_search = newton_solver.kernel_line_search

        def solver_structs(args):
            """The environments and the loop flags of the Newton solve among the arguments of a kernel."""
            newton_envs = next(arg for arg in args if isinstance(arg, newton_solver.NewtonEnvs))
            newton_loops = next(arg for arg in args if isinstance(arg, newton_solver.NewtonLoops))
            return newton_envs, newton_loops

        def counted_newton_direction(*args):
            newton_envs, newton_loops = solver_structs(args)
            newton_direction(*args)
            if bool(qd_to_numpy(newton_envs.is_active)[0]):
                n_left = int(qd_to_numpy(newton_loops.n_pcg_iterations_left))
                self.n_pcg_iterations += self.pcg_max_iterations - n_left

        def counted_line_search(*args):
            newton_envs, _ = solver_structs(args)
            is_active = bool(qd_to_numpy(newton_envs.is_active)[0])
            line_search(*args)
            if is_active:
                self.n_line_search_trials += int(qd_to_numpy(newton_envs.n_line_search_iterations)[0])

        newton_solver.kernel_newton_direction = counted_newton_direction
        newton_solver.kernel_line_search = counted_line_search

    def take(self, scene) -> tuple[int, int, int]:
        """Return the Newton iterations, line-search trials and PCG iterations of the last step, and reset."""
        n_newton = int(qd_to_numpy(scene.sim.newton_solver.newton_envs.n_iterations)[0])
        counts = (n_newton, self.n_line_search_trials, self.n_pcg_iterations)
        self.n_line_search_trials = 0
        self.n_pcg_iterations = 0
        return counts
