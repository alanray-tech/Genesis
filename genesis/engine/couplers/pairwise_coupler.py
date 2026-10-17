"""Configuration marker for the compositional pairwise coupling pipeline."""

from typing import TYPE_CHECKING

from genesis.options.solvers import LegacyCouplerOptions
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator


class PairwiseCoupler(RBC):
    """Host-visible selection and options for coupling performed entirely by pairwise Systems.

    This object owns no device data and executes no simulation phase. `CouplerSystem` and the registered pairwise
    coupling Systems own the explicit forward, backward and reset pipelines.
    """

    def __init__(self, simulator: "Simulator", options: LegacyCouplerOptions) -> None:
        self.sim = simulator
        self.options = options

    @property
    def active_solvers(self):
        return self.sim.active_solvers
