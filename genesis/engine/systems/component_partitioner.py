from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import quadrants as qd

from .dynamic_exclusive_sum import DynamicExclusiveSum
from .sim_system import SimSystem


def _normalize_edge_pairs(
    edge_pairs: Iterable[tuple[int, int]] | np.ndarray | None,
    n_block_rows: int,
) -> np.ndarray:
    if edge_pairs is None:
        return np.empty((0, 2), dtype=np.int32)

    if isinstance(edge_pairs, np.ndarray):
        pairs = np.asarray(edge_pairs)
    else:
        pairs = np.asarray(list(edge_pairs))
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    if pairs.ndim == 1:
        if pairs.size % 2 != 0:
            raise ValueError("ComponentPartitioner edge data must contain row pairs")
        pairs = pairs.reshape(-1, 2)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("ComponentPartitioner edges must have shape (n_edges, 2)")
    if not np.issubdtype(pairs.dtype, np.integer):
        raise TypeError("ComponentPartitioner edge rows must be integers")

    pairs_i64 = np.asarray(pairs, dtype=np.int64)
    if np.any(pairs_i64 < 0) or np.any(pairs_i64 >= n_block_rows):
        raise ValueError(f"ComponentPartitioner edge rows must lie within [0, {n_block_rows})")
    return np.ascontiguousarray(pairs_i64, dtype=np.int32)


@qd.data_oriented
class ComponentPartitioner(SimSystem):
    """Graph-native connected components over 3x3 Hessian block rows."""

    def __init__(self) -> None:
        super().__init__()
        self.is_initialized_host = False

    def do_build(self) -> None:
        from .global_linear_system import GlobalLinearSystem

        self.global_linear_system = self.require(GlobalLinearSystem)

    def init(
        self,
        n_block_rows: int,
        max_sv_iter: int,
        static_edges: Iterable[tuple[int, int]] | np.ndarray | None = None,
    ) -> None:
        if self.is_initialized_host:
            raise RuntimeError("ComponentPartitioner is already initialized")
        if n_block_rows < 0:
            raise ValueError("ComponentPartitioner block-row count must be non-negative")
        if n_block_rows > np.iinfo(np.int32).max:
            raise ValueError("ComponentPartitioner block-row count exceeds i32")
        if max_sv_iter <= 0:
            raise ValueError("ComponentPartitioner max_sv_iter must be positive")

        edge_pairs = _normalize_edge_pairs(static_edges, n_block_rows)
        storage = max(n_block_rows, 1)
        edge_storage = max(edge_pairs.shape[0], 1)

        self.n_block_rows_host = n_block_rows
        self.max_sv_iter_host = max_sv_iter
        self.n_static_edges_host = edge_pairs.shape[0]

        self.static_labels = qd.ndarray(qd.i32, shape=(storage,))
        self.parent = qd.ndarray(qd.i32, shape=(storage,))
        self.is_root = qd.ndarray(qd.i32, shape=(storage,))
        self.comp_label = qd.ndarray(qd.i32, shape=(storage,))
        self.K = qd.ndarray(qd.i32, shape=())
        self.changed = qd.ndarray(qd.i32, shape=())
        self.loop_condition = qd.ndarray(qd.i32, shape=())
        self.iter_remaining = qd.ndarray(qd.i32, shape=())
        self.n_block_rows = qd.ndarray(qd.i32, shape=())
        self.n_static_edges = qd.ndarray(qd.i32, shape=())
        self.max_sv_iter = qd.ndarray(qd.i32, shape=())
        self.static_edges = qd.ndarray(qd.i32, shape=(edge_storage, 2))
        self.relabel_scan = DynamicExclusiveSum(storage)

        self.static_labels.from_numpy(np.zeros(storage, dtype=np.int32))
        self.parent.from_numpy(np.zeros(storage, dtype=np.int32))
        self.is_root.from_numpy(np.zeros(storage, dtype=np.int32))
        self.comp_label.from_numpy(np.zeros(storage, dtype=np.int32))
        self.K.from_numpy(np.array(0, dtype=np.int32))
        self.changed.from_numpy(np.array(0, dtype=np.int32))
        self.loop_condition.from_numpy(np.array(0, dtype=np.int32))
        self.iter_remaining.from_numpy(np.array(0, dtype=np.int32))
        self.n_block_rows.from_numpy(np.array(n_block_rows, dtype=np.int32))
        self.n_static_edges.from_numpy(np.array(edge_pairs.shape[0], dtype=np.int32))
        self.max_sv_iter.from_numpy(np.array(max_sv_iter, dtype=np.int32))
        uploaded_edges = np.zeros((edge_storage, 2), dtype=np.int32)
        uploaded_edges[: edge_pairs.shape[0]] = edge_pairs
        self.static_edges.from_numpy(uploaded_edges)
        self.is_initialized_host = True

    @qd.func
    def _find_representative(self, vertex):
        current = self.parent[vertex]
        if current != vertex:
            next_vertex = self.parent[current]
            while current > next_vertex:
                self.parent[vertex] = next_vertex
                vertex = current
                current = next_vertex
                next_vertex = self.parent[current]
        return current

    @qd.func
    def _union(self, left, right):
        if left != right:
            left_root = self._find_representative(left)
            right_root = self._find_representative(right)
            while left_root != right_root:
                high = qd.max(left_root, right_root)
                low = qd.min(left_root, right_root)
                previous = qd.atomic_cas(
                    self.parent[high],
                    high,
                    low,
                )
                if previous == high:
                    qd.atomic_max(self.changed[()], qd.i32(1))
                    left_root = low
                else:
                    left_root = self._find_representative(previous)
                    right_root = self._find_representative(low)

    @qd.func(requires_top_level=True)
    def _flatten(self):
        qd.loop_config(name="component_sv_flatten")
        for vertex in range(self.n_block_rows[()]):
            root = self.parent[vertex]
            next_vertex = self.parent[root]
            while root != next_vertex:
                root = next_vertex
                next_vertex = self.parent[root]
            if self.parent[vertex] != root:
                self.parent[vertex] = root
                qd.atomic_max(self.changed[()], qd.i32(1))

    @qd.func(requires_top_level=True)
    def _finish_iteration(self):
        qd.loop_config(name="component_sv_convergence")
        for _ in range(1):
            remaining = self.iter_remaining[()] - 1
            self.iter_remaining[()] = remaining
            if remaining <= 0 or self.changed[()] == 0:
                self.loop_condition[()] = 0

    @qd.func(requires_top_level=True)
    def _fallback_if_unconverged(self):
        qd.loop_config(name="component_sv_safe_fallback")
        for row in range(self.n_block_rows[()]):
            if self.iter_remaining[()] <= 0 and self.changed[()] != 0:
                self.parent[row] = 0
        qd.loop_config(name="component_sv_finalize_fallback")
        for _ in range(1):
            if self.iter_remaining[()] <= 0 and self.changed[()] != 0:
                self.changed[()] = 0

    @qd.func(requires_top_level=True)
    def _dense_relabel(self):
        qd.loop_config(name="component_mark_roots")
        for row in range(self.n_block_rows[()]):
            self.is_root[row] = qd.i32(self.parent[row] == row)

        self.relabel_scan.scan(
            self.is_root,
            self.comp_label,
            self.n_block_rows[()],
        )

        qd.loop_config(name="component_extract_K")
        for _ in range(1):
            if self.n_block_rows[()] == 0:
                self.K[()] = 0
            else:
                last = self.n_block_rows[()] - 1
                self.K[()] = self.comp_label[last] + self.is_root[last]

        qd.loop_config(name="component_scatter_labels")
        for row in range(self.n_block_rows[()]):
            self.comp_label[row] = self.comp_label[self.parent[row]]

    @qd.func(requires_top_level=True)
    def compute_static_labels(self):
        """Build the reusable component seed from uploaded host-static edges."""
        qd.loop_config(name="component_static_init_identity")
        for row in range(self.n_block_rows[()]):
            self.parent[row] = row

        qd.loop_config(name="component_static_set_loop")
        for _ in range(1):
            self.changed[()] = 0
            self.iter_remaining[()] = self.max_sv_iter[()]
            self.loop_condition[()] = qd.i32(self.n_block_rows[()] > 0)

        while qd.graph.do_while(self.loop_condition):
            qd.loop_config(name="component_static_clear_changed")
            for _ in range(1):
                self.changed[()] = 0

            qd.loop_config(name="component_static_sv_hook")
            for edge in range(self.n_static_edges[()]):
                self._union(
                    self.static_edges[edge, 0],
                    self.static_edges[edge, 1],
                )

            self._flatten()
            self._finish_iteration()

        self._fallback_if_unconverged()

        qd.loop_config(name="component_copy_static_labels")
        for row in range(self.n_block_rows[()]):
            self.static_labels[row] = self.parent[row]

        self._dense_relabel()

    @qd.func(requires_top_level=True)
    def partition(self):
        """Merge the static seed with the current linear-system BCOO graph."""
        qd.loop_config(name="component_runtime_init_labels")
        for row in range(self.n_block_rows[()]):
            self.parent[row] = self.static_labels[row]

        qd.loop_config(name="component_runtime_set_loop")
        for _ in range(1):
            self.changed[()] = 0
            self.iter_remaining[()] = self.max_sv_iter[()]
            self.loop_condition[()] = qd.i32(self.n_block_rows[()] > 0)

        while qd.graph.do_while(self.loop_condition):
            qd.loop_config(name="component_runtime_clear_changed")
            for _ in range(1):
                self.changed[()] = 0

            qd.loop_config(name="component_runtime_sv_hook")
            for edge in range(self.global_linear_system.bcoo_nnz[()]):
                left = self.global_linear_system.bcoo_row[edge]
                right = self.global_linear_system.bcoo_col[edge]
                if left >= 0 and left < self.n_block_rows[()] and right >= 0 and right < self.n_block_rows[()]:
                    self._union(left, right)

            self._flatten()
            self._finish_iteration()

        self._fallback_if_unconverged()
        self._dense_relabel()
