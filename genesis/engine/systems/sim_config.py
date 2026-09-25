from __future__ import annotations

import quadrants as qd

from .sim_system import SimSystem


@qd.data_oriented
class SimConfig(SimSystem):
    """Store the numerical controls shared by the global Newton pipeline."""

    def __init__(
        self,
        h: float,
        max_newton: int,
        max_pcg: int,
        max_line_search: int,
        newton_tolerance: float,
        pcg_tolerance: float,
    ) -> None:
        super().__init__()
        self.h = h
        self.max_newton = max_newton
        self.max_pcg = max_pcg
        self.max_line_search = max_line_search
        self.newton_tolerance = newton_tolerance
        self.pcg_tolerance = pcg_tolerance

    def build(self) -> None:
        pass
