from __future__ import annotations

from typing import NamedTuple

import quadrants as qd

from .sim_system import SimSystem


class DofRange(NamedTuple):
    scalar_offset: int
    block_offset: int
    n_dofs: int
    n_blocks: int

    @property
    def n_storage_dofs(self) -> int:
        return self.n_blocks * 3


@qd.data_oriented
class GlobalLayout(SimSystem):
    """Allocate participant-owned scalar ranges in a uniform three-lane block layout."""

    def __init__(self) -> None:
        super().__init__()
        self.n_block_rows = 0
        self.n_dofs = 0
        self.is_built = False

    def allocate(self, n_dofs: int) -> DofRange:
        """Allocate a block-aligned range for a participant."""
        if self.is_built:
            raise RuntimeError("GlobalLayout cannot allocate ranges after build")
        if n_dofs < 0:
            raise ValueError("The number of participant DOFs must be non-negative")

        n_blocks = (n_dofs + 2) // 3
        block_offset = self.n_block_rows
        dof_range = DofRange(
            scalar_offset=block_offset * 3,
            block_offset=block_offset,
            n_dofs=n_dofs,
            n_blocks=n_blocks,
        )
        self.n_block_rows += n_blocks
        self.n_dofs = self.n_block_rows * 3
        return dof_range

    def build(self) -> None:
        self.is_built = True
