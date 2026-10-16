"""Manager System of the scene's coupling phases."""

from typing import TYPE_CHECKING

from genesis.engine.core import HostAction, System, register_system
from genesis.engine.couplers import PairwiseCoupler

from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class CouplerSystem(System):
    """Manager of the coupling schedule.

    SAP and IPC keep their monolithic host entry points. Pairwise coupling is composed from Systems, each of which
    registers one ordered preprocess or coupling Action and owns the data specific to that interaction.

    A collection boundary below exists only for a data dependency: MPM grid begin produces ``vel_out``; every MPM
    pair consumes and mutates that same velocity in sequence; grid end consumes their result; and FEM-MPM/FEM-SPH
    consume the FEM surface velocity after Rigid-FEM has updated it. Post-MPM interactions with rigid have no
    producer/consumer edge and therefore share one collection.
    """

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        selected_coupler = scene.sim.coupler
        self.is_pairwise = isinstance(selected_coupler, PairwiseCoupler)
        # Pairwise execution is fully composed below; retain no PairwiseCoupler reference on the new pipeline.
        self.coupler = None if self.is_pairwise else selected_coupler
        self.pairwise_options = selected_coupler.options if self.is_pairwise else None

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_mpm_rigid_preprocess(self):
        """CPIC preprocessing before solver pre-coupling."""

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_mpm_grid_begin(self):
        """Serial exception: produce MPM ``vel_out`` consumed by every MPM pair below."""

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_mpm_tool_couple(self):
        """Serial exception: consume and mutate ``vel_out`` produced by grid begin."""

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_mpm_rigid_couple(self):
        """Serial exception: consume and mutate ``vel_out`` after MPM-Tool."""

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_mpm_sph_couple(self):
        """Serial exception: consume and mutate ``vel_out`` after MPM-Rigid, and produce SPH velocity."""

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_mpm_pbd_couple(self):
        """Serial exception: consume and mutate ``vel_out`` after MPM-SPH, and produce PBD velocity."""

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_mpm_grid_end(self):
        """Serial exception: consume the final ``vel_out`` after all MPM pair transforms."""

    @System.action_collection(kind="host", call_args=("f",))
    def parallel_couples(self):
        """Default phase for independent pair couplers.

        Host fallback invokes them serially; graph mode launches the corresponding StageActions in parallel.
        """

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_fem_mpm_couple(self):
        """Serial exception: consume FEM velocity after Rigid-FEM and MPM ``vel_out`` after grid end."""

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_fem_sph_couple(self):
        """Serial exception: consume FEM velocity after FEM-MPM and SPH velocity after the parallel phase."""

    @System.action_collection(kind="host", call_args=("envs_idx",))
    def pairwise_resets(self):
        """Independent pairwise coupling state resets."""

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_mpm_grid_begin_grads(self):
        """Adjoint of MPM grid velocity initialization."""

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_mpm_tool_grads(self):
        """Adjoint of MPM-Tool coupling."""

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_mpm_rigid_grads(self):
        """Adjoint of MPM-Rigid coupling."""

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_mpm_sph_grads(self):
        """Adjoint of MPM-SPH coupling."""

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_mpm_pbd_grads(self):
        """Adjoint of MPM-PBD coupling."""

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_mpm_grid_end_grads(self):
        """Adjoint of MPM grid boundary conditions."""

    @System.action_collection(kind="host", call_args=("f",))
    def parallel_grads(self):
        """Adjoints of the default independent pair couplers."""

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_fem_mpm_grads(self):
        """Adjoint of FEM-MPM coupling."""

    @System.action_collection(kind="host", call_args=("f",))
    def pairwise_fem_sph_grads(self):
        """Adjoint of FEM-SPH coupling."""

    @System.protocol
    def on_pairwise_mpm_rigid_preprocess(self, *, action: HostAction) -> None:
        self.pairwise_mpm_rigid_preprocess += action

    @System.protocol
    def on_pairwise_mpm_grid_begin(self, *, action: HostAction) -> None:
        self.pairwise_mpm_grid_begin += action

    @System.protocol
    def on_pairwise_mpm_tool_couple(self, *, action: HostAction) -> None:
        self.pairwise_mpm_tool_couple += action

    @System.protocol
    def on_pairwise_mpm_rigid_couple(self, *, action: HostAction) -> None:
        self.pairwise_mpm_rigid_couple += action

    @System.protocol
    def on_pairwise_mpm_sph_couple(self, *, action: HostAction) -> None:
        self.pairwise_mpm_sph_couple += action

    @System.protocol
    def on_pairwise_mpm_pbd_couple(self, *, action: HostAction) -> None:
        self.pairwise_mpm_pbd_couple += action

    @System.protocol
    def on_pairwise_mpm_grid_end(self, *, action: HostAction) -> None:
        self.pairwise_mpm_grid_end += action

    @System.protocol
    def on_pairwise_couple(self, *, action: HostAction) -> None:
        """Register a dependency-free pair in the default parallel phase."""
        self.parallel_couples += action

    @System.protocol
    def on_pairwise_fem_mpm_couple(self, *, action: HostAction) -> None:
        self.pairwise_fem_mpm_couple += action

    @System.protocol
    def on_pairwise_fem_sph_couple(self, *, action: HostAction) -> None:
        self.pairwise_fem_sph_couple += action

    @System.protocol
    def on_pairwise_reset(self, *, action: HostAction) -> None:
        self.pairwise_resets += action

    @System.protocol
    def on_pairwise_mpm_grid_begin_grad(self, *, action: HostAction) -> None:
        self.pairwise_mpm_grid_begin_grads += action

    @System.protocol
    def on_pairwise_mpm_tool_grad(self, *, action: HostAction) -> None:
        self.pairwise_mpm_tool_grads += action

    @System.protocol
    def on_pairwise_mpm_rigid_grad(self, *, action: HostAction) -> None:
        self.pairwise_mpm_rigid_grads += action

    @System.protocol
    def on_pairwise_mpm_sph_grad(self, *, action: HostAction) -> None:
        self.pairwise_mpm_sph_grads += action

    @System.protocol
    def on_pairwise_mpm_pbd_grad(self, *, action: HostAction) -> None:
        self.pairwise_mpm_pbd_grads += action

    @System.protocol
    def on_pairwise_mpm_grid_end_grad(self, *, action: HostAction) -> None:
        self.pairwise_mpm_grid_end_grads += action

    @System.protocol
    def on_pairwise_grad(self, *, action: HostAction) -> None:
        """Register the adjoint of a dependency-free pair."""
        self.parallel_grads += action

    @System.protocol
    def on_pairwise_fem_mpm_grad(self, *, action: HostAction) -> None:
        self.pairwise_fem_mpm_grads += action

    @System.protocol
    def on_pairwise_fem_sph_grad(self, *, action: HostAction) -> None:
        self.pairwise_fem_sph_grads += action

    @System.action(kind="host")
    def preprocess(self):
        return self.run_preprocess

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    def run_preprocess(self, f: int) -> None:
        if not self.is_pairwise:
            self.coupler.preprocess(f)
            return
        for action in self.pairwise_mpm_rigid_preprocess.actions:
            action.invoke((f,))

    def run_couple(self, f: int) -> None:
        if not self.is_pairwise:
            self.coupler.couple(f)
            return
        # These are distinct lifecycle phases, not ranked peers: each line names what the next phase may consume.
        for collection in (
            self.pairwise_mpm_grid_begin,
            self.pairwise_mpm_tool_couple,
            self.pairwise_mpm_rigid_couple,
            self.pairwise_mpm_sph_couple,
            self.pairwise_mpm_pbd_couple,
            self.pairwise_mpm_grid_end,
            self.parallel_couples,
            self.pairwise_fem_mpm_couple,
            self.pairwise_fem_sph_couple,
        ):
            for action in collection.actions:
                action.invoke((f,))

    def run_reset(self, envs_idx) -> None:
        if not self.is_pairwise:
            self.coupler.reset(envs_idx=envs_idx)
            return
        for action in self.pairwise_resets.actions:
            action.invoke((envs_idx,))

    def run_couple_grad(self, f: int) -> None:
        if not self.is_pairwise:
            self.coupler.couple_grad(f)
            return
        # Reverse-mode follows the exact reverse of the explicit forward phases above.
        for collection in (
            self.pairwise_fem_sph_grads,
            self.pairwise_fem_mpm_grads,
            self.parallel_grads,
            self.pairwise_mpm_grid_end_grads,
            self.pairwise_mpm_pbd_grads,
            self.pairwise_mpm_sph_grads,
            self.pairwise_mpm_rigid_grads,
            self.pairwise_mpm_tool_grads,
            self.pairwise_mpm_grid_begin_grads,
        ):
            for action in collection.actions:
                action.invoke((f,))


@register_system(SimulatorEngine)
def create_coupler_system(scene: "Scene") -> CouplerSystem:
    """Create the coupling manager used by every scene stepped by the engine."""
    return CouplerSystem(scene)
