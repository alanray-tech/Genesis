from __future__ import annotations

import numpy as np
import quadrants as qd

from .component_partitioner import ComponentPartitioner
from .masked_linear_pcg import MaskedLinearPCG
from .sim_system import SimSystem


@qd.data_oriented
class PartitionPCGSolver(SimSystem):
    """Fast-SV component partitioning plus progressively masked PCG."""

    def __init__(self, *, max_sv_iter: int = 64) -> None:
        super().__init__()
        if max_sv_iter <= 0:
            raise ValueError("PartitionPCGSolver max_sv_iter must be positive")
        self.max_sv_iter_host = int(max_sv_iter)
        self.is_initialized_host = False

    def do_build(self) -> None:
        self.partitioner = self.require(ComponentPartitioner)

    def init(
        self,
        total_dof: int,
        n_block_rows: int,
        pcg_tol_rate: float,
        static_edges: np.ndarray,
    ) -> None:
        if self.is_initialized_host:
            raise RuntimeError("PartitionPCGSolver is already initialized")
        self.partitioner.init(
            n_block_rows,
            self.max_sv_iter_host,
            static_edges,
        )
        self.linear_pcg = MaskedLinearPCG(total_dof, n_block_rows)
        self.pcg_tol_rate = qd.ndarray(qd.f64, shape=())
        self.pcg_tol_rate.from_numpy(np.array(pcg_tol_rate, dtype=np.float64))
        self.is_initialized_host = True

    @qd.func(requires_top_level=True)
    def solve(
        self,
        global_linear_system: qd.template(),
        rigid: qd.template(),
        rigid_forest: qd.template(),
        fem_preconditioner: qd.template(),
        has_rigid: qd.template(),
        has_rigid_forest: qd.template(),
        has_fem: qd.template(),
        max_iter,
    ):
        self.partitioner.partition()
        self.linear_pcg.initialize(
            global_linear_system,
            self.partitioner,
            rigid,
            rigid_forest,
            fem_preconditioner,
            has_rigid,
            has_rigid_forest,
            has_fem,
            self.pcg_tol_rate[()],
            max_iter,
        )
        while qd.graph.do_while(self.linear_pcg.condition):
            self.linear_pcg.iteration(
                global_linear_system,
                self.partitioner,
                rigid,
                rigid_forest,
                fem_preconditioner,
                has_rigid,
                has_rigid_forest,
                has_fem,
                max_iter,
            )
