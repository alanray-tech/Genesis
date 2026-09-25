from __future__ import annotations

import numpy as np
import quadrants as qd
from quadrants.algorithms import (
    exclusive_scan_add,
    exclusive_scan_scratch_slots,
    sort,
    sort_scratch_slots,
)

import genesis as gs

from .sim_system import SimSystem


@qd.data_oriented
class GlobalLinearSystem(SimSystem):
    """Own the global vectors and the explicit symmetric three-by-three block operator."""

    def __init__(self, n_block_rows: int, n_triplets: int = 0) -> None:
        super().__init__()
        if n_block_rows < 0:
            raise ValueError("The number of global block rows must be non-negative")
        if n_triplets < 0:
            raise ValueError("The triplet capacity must be non-negative")

        self.sort_end_bit = 64
        self.sort_log256_max_n = 4
        self.scan_log256_max_n = 4

        self.n_block_rows = n_block_rows
        self.n_dofs = n_block_rows * 3
        self.n_triplets = n_triplets
        self.has_explicit_operator = n_triplets > 0
        self.is_cpu = gs.backend == gs.cpu
        self.triplet_capacity = max(n_triplets, 1)

        padded_capacity = max(((self.triplet_capacity + 63) // 64) * 64, 64)
        sort_scratch_size = max(sort_scratch_slots(padded_capacity, self.sort_log256_max_n), 1)
        scan_scratch_size = max(exclusive_scan_scratch_slots(padded_capacity, self.scan_log256_max_n), 1)

        self.n_dofs_device = qd.ndarray(qd.i32, shape=(1,))
        self.triplet_capacity_device = qd.ndarray(qd.i32, shape=(1,))
        self.n_live_triplets = qd.ndarray(qd.i32, shape=(1,))
        self.n_padded_live_triplets = qd.ndarray(qd.i32, shape=(1,))

        self.triplet_row = qd.ndarray(qd.i32, shape=(self.triplet_capacity,))
        self.triplet_col = qd.ndarray(qd.i32, shape=(self.triplet_capacity,))
        self.triplet_value = qd.ndarray(qd.f64, shape=(self.triplet_capacity * 9,))

        self.sort_keys = qd.ndarray(qd.u64, shape=(padded_capacity,))
        self.sort_keys_scratch = qd.ndarray(qd.u64, shape=(padded_capacity,))
        self.sort_permutation = qd.ndarray(qd.u32, shape=(padded_capacity,))
        self.sort_permutation_scratch = qd.ndarray(qd.u32, shape=(padded_capacity,))
        self.sort_scratch = qd.ndarray(qd.u32, shape=(sort_scratch_size,))
        self.sort_size = qd.ndarray(qd.i32, shape=())

        self.segment_flags = qd.ndarray(qd.u32, shape=(padded_capacity,))
        self.segment_ids = qd.ndarray(qd.u32, shape=(padded_capacity,))
        self.scan_scratch = qd.ndarray(qd.u32, shape=(scan_scratch_size,))

        self.bcoo_row = qd.ndarray(qd.i32, shape=(self.triplet_capacity,))
        self.bcoo_col = qd.ndarray(qd.i32, shape=(self.triplet_capacity,))
        self.bcoo_value = qd.ndarray(qd.f64, shape=(self.triplet_capacity * 9,))
        self.n_bcoo = qd.ndarray(qd.i32, shape=())

        self.rhs = qd.ndarray(qd.f64, shape=(max(self.n_dofs, 1),))
        self.solution = qd.ndarray(qd.f64, shape=(max(self.n_dofs, 1),))

    def build(self) -> None:
        self.n_dofs_device.from_numpy(np.array([self.n_dofs], dtype=np.int32))
        self.triplet_capacity_device.from_numpy(np.array([self.triplet_capacity], dtype=np.int32))
        self.n_live_triplets.from_numpy(np.array([self.n_triplets], dtype=np.int32))
        self.n_padded_live_triplets.from_numpy(np.array([((self.n_triplets + 63) // 64) * 64], dtype=np.int32))
        self.sort_size.from_numpy(np.array(self.n_triplets, dtype=np.int32))
        self.n_bcoo.from_numpy(np.array(0, dtype=np.int32))

    def reallocate_triplets(self, n_triplets: int) -> None:
        """Replace explicit-matrix workspaces with the requested capacity."""
        if n_triplets < self.n_triplets:
            raise ValueError("The new triplet capacity cannot be smaller than the fixed triplet extent")

        self.triplet_capacity = max(n_triplets, 1)
        padded_capacity = max(((self.triplet_capacity + 63) // 64) * 64, 64)
        sort_scratch_size = max(sort_scratch_slots(padded_capacity, self.sort_log256_max_n), 1)
        scan_scratch_size = max(exclusive_scan_scratch_slots(padded_capacity, self.scan_log256_max_n), 1)

        self.triplet_row = qd.ndarray(qd.i32, shape=(self.triplet_capacity,))
        self.triplet_col = qd.ndarray(qd.i32, shape=(self.triplet_capacity,))
        self.triplet_value = qd.ndarray(qd.f64, shape=(self.triplet_capacity * 9,))
        self.sort_keys = qd.ndarray(qd.u64, shape=(padded_capacity,))
        self.sort_keys_scratch = qd.ndarray(qd.u64, shape=(padded_capacity,))
        self.sort_permutation = qd.ndarray(qd.u32, shape=(padded_capacity,))
        self.sort_permutation_scratch = qd.ndarray(qd.u32, shape=(padded_capacity,))
        self.sort_scratch = qd.ndarray(qd.u32, shape=(sort_scratch_size,))
        self.segment_flags = qd.ndarray(qd.u32, shape=(padded_capacity,))
        self.segment_ids = qd.ndarray(qd.u32, shape=(padded_capacity,))
        self.scan_scratch = qd.ndarray(qd.u32, shape=(scan_scratch_size,))
        self.bcoo_row = qd.ndarray(qd.i32, shape=(self.triplet_capacity,))
        self.bcoo_col = qd.ndarray(qd.i32, shape=(self.triplet_capacity,))
        self.bcoo_value = qd.ndarray(qd.f64, shape=(self.triplet_capacity * 9,))
        self.triplet_capacity_device.from_numpy(np.array([self.triplet_capacity], dtype=np.int32))

    @qd.func(requires_top_level=True)
    def zero_assembly(self):
        for i_d in range(self.n_dofs_device[0]):
            self.rhs[i_d] = qd.f64(0.0)
        for i_value in range(self.triplet_capacity_device[0] * 9):
            self.triplet_value[i_value] = qd.f64(0.0)

    @qd.func(requires_top_level=True)
    def set_n_live_triplets(self, n_live_triplets):
        for _ in range(1):
            self.n_live_triplets[0] = n_live_triplets
            self.n_padded_live_triplets[0] = ((n_live_triplets + 63) // 64) * 64
            self.sort_size[()] = n_live_triplets

    @qd.func(requires_top_level=True)
    def sort_seed(self):
        for i_triplet in range(self.n_padded_live_triplets[0]):
            if i_triplet < self.n_live_triplets[0]:
                row = qd.u64(self.triplet_row[i_triplet])
                col = qd.u64(self.triplet_col[i_triplet])
                self.sort_keys[i_triplet] = (row << 32) | col
                self.sort_permutation[i_triplet] = qd.u32(i_triplet)
            else:
                self.sort_keys[i_triplet] = qd.u64(0xFFFFFFFFFFFFFFFF)
                self.sort_permutation[i_triplet] = qd.u32(i_triplet)

    @qd.func(requires_top_level=True)
    def sort_radix(self):
        sort(
            self.sort_keys,
            self.sort_keys_scratch,
            self.sort_permutation,
            self.sort_permutation_scratch,
            self.sort_scratch,
            self.sort_size,
            qd.u64,
            True,
            self.sort_end_bit,
            self.sort_log256_max_n,
        )

    @qd.func(requires_top_level=True)
    def sort_segment_flags(self):
        for i_triplet in range(self.n_padded_live_triplets[0]):
            if i_triplet < self.n_live_triplets[0]:
                is_tail = i_triplet == self.n_live_triplets[0] - 1
                if not is_tail:
                    is_tail = self.sort_keys[i_triplet] != self.sort_keys[i_triplet + 1]
                self.segment_flags[i_triplet] = qd.u32(is_tail)
            else:
                self.segment_flags[i_triplet] = qd.u32(0)

    @qd.func(requires_top_level=True)
    def sort_scan(self):
        exclusive_scan_add(
            self.segment_flags,
            self.segment_ids,
            self.scan_scratch,
            self.n_live_triplets[0],
            qd.u32,
            self.scan_log256_max_n,
        )

    @qd.func(requires_top_level=True)
    def sort_zero_bcoo(self):
        for _ in range(1):
            self.n_bcoo[()] = 0
        for i_value in range(self.n_live_triplets[0] * 9):
            self.bcoo_value[i_value] = qd.f64(0.0)

    @qd.func(requires_top_level=True)
    def sort_reduce(self):
        for i_triplet in range(self.n_live_triplets[0]):
            i_source = qd.i32(self.sort_permutation[i_triplet])
            i_segment = qd.i32(self.segment_ids[i_triplet])
            for i_value in range(9):
                qd.atomic_add(
                    self.bcoo_value[i_segment * 9 + i_value],
                    self.triplet_value[i_source * 9 + i_value],
                )

    @qd.func(requires_top_level=True)
    def sort_extract(self):
        for i_triplet in range(self.n_live_triplets[0]):
            if self.segment_flags[i_triplet] == qd.u32(1):
                i_segment = qd.i32(self.segment_ids[i_triplet])
                key = self.sort_keys[i_triplet]
                self.bcoo_row[i_segment] = qd.i32(key >> 32)
                self.bcoo_col[i_segment] = qd.i32(key & qd.u64(0xFFFFFFFF))
                if i_triplet == self.n_live_triplets[0] - 1:
                    self.n_bcoo[()] = i_segment + 1

    @qd.func(requires_top_level=True)
    def reduce_cpu(self):
        for _ in range(1):
            n_unique = 0
            for i_triplet in range(self.n_live_triplets[0]):
                row = self.triplet_row[i_triplet]
                col = self.triplet_col[i_triplet]
                i_unique = qd.i32(-1)
                for j_unique in range(n_unique):
                    if self.bcoo_row[j_unique] == row and self.bcoo_col[j_unique] == col:
                        i_unique = j_unique
                if i_unique < 0:
                    i_unique = n_unique
                    n_unique = n_unique + 1
                    self.bcoo_row[i_unique] = row
                    self.bcoo_col[i_unique] = col
                    for i_value in range(9):
                        self.bcoo_value[i_unique * 9 + i_value] = qd.f64(0.0)
                for i_value in range(9):
                    self.bcoo_value[i_unique * 9 + i_value] = (
                        self.bcoo_value[i_unique * 9 + i_value] + self.triplet_value[i_triplet * 9 + i_value]
                    )
            self.n_bcoo[()] = n_unique

    @qd.func(requires_top_level=True)
    def finalize_bcoo(self):
        if qd.static(self.is_cpu):
            self.reduce_cpu()
        else:
            self.sort_seed()
            self.sort_radix()
            self.sort_segment_flags()
            self.sort_scan()
            self.sort_zero_bcoo()
            self.sort_reduce()
            self.sort_extract()

    @qd.func(requires_top_level=True)
    def apply_bcoo(self, x: qd.template(), y: qd.template()):
        for i_entry in range(self.n_bcoo[()]):
            row = self.bcoo_row[i_entry]
            col = self.bcoo_col[i_entry]
            x0 = x[col * 3]
            x1 = x[col * 3 + 1]
            x2 = x[col * 3 + 2]
            b0 = self.bcoo_value[i_entry * 9]
            b1 = self.bcoo_value[i_entry * 9 + 1]
            b2 = self.bcoo_value[i_entry * 9 + 2]
            b3 = self.bcoo_value[i_entry * 9 + 3]
            b4 = self.bcoo_value[i_entry * 9 + 4]
            b5 = self.bcoo_value[i_entry * 9 + 5]
            b6 = self.bcoo_value[i_entry * 9 + 6]
            b7 = self.bcoo_value[i_entry * 9 + 7]
            b8 = self.bcoo_value[i_entry * 9 + 8]
            qd.atomic_add(y[row * 3], b0 * x0 + b1 * x1 + b2 * x2)
            qd.atomic_add(y[row * 3 + 1], b3 * x0 + b4 * x1 + b5 * x2)
            qd.atomic_add(y[row * 3 + 2], b6 * x0 + b7 * x1 + b8 * x2)

            if row != col:
                row_x0 = x[row * 3]
                row_x1 = x[row * 3 + 1]
                row_x2 = x[row * 3 + 2]
                qd.atomic_add(y[col * 3], b0 * row_x0 + b3 * row_x1 + b6 * row_x2)
                qd.atomic_add(y[col * 3 + 1], b1 * row_x0 + b4 * row_x1 + b7 * row_x2)
                qd.atomic_add(y[col * 3 + 2], b2 * row_x0 + b5 * row_x1 + b8 * row_x2)
