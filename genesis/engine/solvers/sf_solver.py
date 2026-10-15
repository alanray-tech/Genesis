from .base_solver import Solver, TimeBasedMixin, forward_to_system


class SFSolver(TimeBasedMixin, Solver):
    """
    Stable Fluid solver for eulerian-based gaseous simulation.

    The grid and its simulation belong to the System the engine builds for the solver (see SFSolverSystem), which the
    data attributes of the solver forward to.
    """

    grid = forward_to_system("grid_state")
    t = forward_to_system("t", 0.0)

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        if options is None:
            return

        self.n_grid = options.res
        self.dx = 1 / self.n_grid
        self.res = (self.n_grid, self.n_grid, self.n_grid)
        self.solver_iters = options.solver_iters
        self.decay = options.decay

        self.inlet_s = options.inlet_s

        self.jets = ()

    def reset_grad(self):
        pass

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return bool(self.jets)

    def set_jets(self, jets):
        assert isinstance(jets, (list, tuple))
        self.jets = tuple(jets)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward):
        return None

    def substep_pre_coupling(self, f):
        self._system.substep_pre_coupling(f)

    def substep_post_coupling(self, f):
        self._system.substep_post_coupling(f)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        pass

    def add_grad_from_state(self, state):
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self, f):
        pass

    def set_state(self, f, state, envs_idx=None):
        pass

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass
