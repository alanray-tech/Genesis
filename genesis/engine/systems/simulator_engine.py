"""Engine of the simulator, with host fallback and whole-scene graph pipelines."""

import genesis as gs
from genesis.engine.core import Engine, Pipeline, make_stage_schedule


class SimulatorEngine(Engine):
    """The engine of one simulator, which runs the substep phases of its solvers and of its coupler.

    Each System registers the function that creates it for this Engine from its own module (see register_system). The
    simulator keeps the step loop, the clock and the checkpoints, and launches one pipeline per entry point.
    """

    def build(self):
        super().build()
        # The Systems import this module to register, so they are imported here rather than declared on the class
        from .coupler_system import CouplerSystem
        from .initializer import Initializer
        from .rigid_solver_system import RigidSolverSystem
        from .substepper import Substepper, kernel_graph_substep

        self.init_pipeline = Pipeline(self.require(Initializer).initialize)
        substepper = self.require(Substepper)
        self.input_pipeline = Pipeline(substepper.process_input)
        coupler = self.require(CouplerSystem)
        self.coupler_reset_pipeline = Pipeline(coupler.run_reset)
        self.coupling_grad_pipeline = Pipeline(coupler.run_couple_grad)
        self.substep_takes_time = self.scene.options.sim.pipeline == "graph"
        self.graph_rigid_system = None
        if self.substep_takes_time:
            unsupported = [name for name, is_active in (("FEM", substepper.fem is not None),) if is_active]
            if gs.backend != gs.cuda:
                gs.raise_exception("The whole-scene graph pipeline currently requires the CUDA backend.")
            if self.scene.options.sim.requires_grad:
                gs.raise_exception("The whole-scene graph pipeline does not support differentiable simulation yet.")
            if not coupler.is_pairwise:
                gs.raise_exception("The whole-scene graph pipeline currently requires PairwiseCoupler.")
            if unsupported:
                gs.raise_exception(
                    "The whole-scene graph pipeline currently supports Rigid/Kinematic or Tool/MPM/SPH/PBD/SF scenes without "
                    f"{', '.join(unsupported)} Systems."
                )
            self.graph_kinematic_system = None
            self.graph_schedule = make_stage_schedule(substepper)
            self._graph_entries = (kernel_graph_substep,)
            self.substep_pipeline = None
            self.split_substep_pipelines = ()
        else:
            self.substep_pipeline = Pipeline(substepper.substep)
            self.split_substep_pipelines = ()
            self.graph_kinematic_system = None
        # A scene simulating rigid entities alone advances the rigid solver directly, skipping the inputs, which only
        # feed the differentiable tape, and the coupler, which has nothing to exchange
        self.rigid_substep_pipeline = Pipeline(self.require(RigidSolverSystem).substep)

    def finalize_device_data(self) -> None:
        """Build the graph query root after every System has initialized its Data, then bind graph Pipelines to it."""
        if not self.substep_takes_time:
            return
        self.context = self.create_context()
        self.substep_pipeline = Pipeline(self._graph_entries[0], self.context, self.graph_schedule)

    def prepare_graph_substep(self) -> None:
        """Run host-only cache invalidation before the one device graph launch."""
        if self.graph_rigid_system is not None:
            for action in self.graph_rigid_system.preparations.actions:
                action.invoke()
