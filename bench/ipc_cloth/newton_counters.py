"""Per-step counters of the Newton solve of Genesis for the parity benches: Newton iterations, line-search trials and
preconditioned conjugate gradient (PCG) iterations of environment 0, the counts CGQ's solver reports per frame
(newton_iters, total_ls_iters, total_pcg_iters).

The counters wrap the kernels of genesis/engine/solvers/newton_solver.py that end the PCG (the start of the line search)
and the Newton iteration, which the solver looks up in its module at every call. Import after gs.init.
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
        start_line_search = newton_solver.kernel_start_line_search
        end_iteration = newton_solver.kernel_end_iteration

        def counted_start_line_search(newton_rows, newton_envs, newton_loops, *args):
            if bool(qd_to_numpy(newton_envs.is_active)[0]):
                n_left = int(qd_to_numpy(newton_loops.n_pcg_iterations_left))
                self.n_pcg_iterations += self.pcg_max_iterations - n_left
            start_line_search(newton_rows, newton_envs, newton_loops, *args)

        def counted_end_iteration(newton_envs, *args):
            if bool(qd_to_numpy(newton_envs.is_active)[0]):
                self.n_line_search_trials += int(qd_to_numpy(newton_envs.n_line_search_iterations)[0])
            end_iteration(newton_envs, *args)

        newton_solver.kernel_start_line_search = counted_start_line_search
        newton_solver.kernel_end_iteration = counted_end_iteration

    def take(self, scene) -> tuple[int, int, int]:
        """Return the Newton iterations, line-search trials and PCG iterations of the last step, and reset."""
        n_newton = int(qd_to_numpy(scene.sim.newton_solver.newton_envs.n_iterations)[0])
        counts = (n_newton, self.n_line_search_trials, self.n_pcg_iterations)
        self.n_line_search_trials = 0
        self.n_pcg_iterations = 0
        return counts
