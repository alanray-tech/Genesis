"""Engine of the simulator, with host fallback and whole-scene graph pipelines."""

import genesis as gs
from genesis.engine.core import Engine, Pipeline


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
        from .substepper import Substepper, kernel_particle_graph_substep

        self.init_pipeline = Pipeline(self.require(Initializer).initialize)
        substepper = self.require(Substepper)
        self.input_pipeline = Pipeline(substepper.process_input)
        coupler = self.require(CouplerSystem)
        self.coupler_reset_pipeline = Pipeline(coupler.run_reset)
        self.coupling_grad_pipeline = Pipeline(coupler.run_couple_grad)
        self.substep_takes_time = self.scene.options.sim.pipeline == "graph"
        if self.substep_takes_time:
            unsupported = [
                name
                for name, is_active in (
                    ("Rigid", substepper.rigid.is_active),
                    ("Kinematic", substepper.kinematic.is_active),
                    ("FEM", substepper.fem is not None),
                )
                if is_active
            ]
            if gs.backend != gs.cuda:
                gs.raise_exception("The whole-scene graph pipeline currently requires the CUDA backend.")
            if self.scene.options.sim.requires_grad:
                gs.raise_exception("The whole-scene graph pipeline does not support differentiable simulation yet.")
            if not coupler.is_pairwise:
                gs.raise_exception("The whole-scene graph pipeline currently requires PairwiseCoupler.")
            if unsupported:
                gs.raise_exception(
                    "The whole-scene graph pipeline currently supports Tool, MPM, SPH, PBD and SF scenes without "
                    f"{', '.join(unsupported)} Systems."
                )
            self.substep_pipeline = Pipeline(
                kernel_particle_graph_substep,
                substepper.sph is not None,
                substepper.sph,
                substepper.pbd is not None,
                substepper.pbd,
                substepper.sf is not None,
                substepper.sf,
                substepper.mpm is not None,
                substepper.mpm,
                substepper.rigid,
                substepper.rigid.rigid_sdf_system,
                substepper.mpm_sph_coupling is not None,
                substepper.mpm_sph_coupling,
                substepper.mpm_pbd_coupling is not None,
                substepper.mpm_pbd_coupling,
                substepper.mpm is not None and substepper.mpm.solver._constraints_initialized,
                substepper.tool is not None,
                substepper.tool,
                substepper.mpm_tool_coupling is not None,
            )
        else:
            self.substep_pipeline = Pipeline(substepper.substep)
        # A scene simulating rigid entities alone advances the rigid solver directly, skipping the inputs, which only
        # feed the differentiable tape, and the coupler, which has nothing to exchange
        self.rigid_substep_pipeline = Pipeline(self.require(RigidSolverSystem).substep)
