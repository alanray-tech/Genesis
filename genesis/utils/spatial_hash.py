"""Typed state and device functions of the particle spatial hash."""

import dataclasses
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V, DataKind

BLOCK_DIM = 256


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SpatialHasherState:
    """Slot counts, offsets and prefix-sum scratch of one particle spatial hash."""

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    slot_size: qd.Tensor
    slot_start: qd.Tensor
    cur_cnt: qd.Tensor


def get_spatial_hasher_state(hasher, n_batch: int) -> SpatialHasherState:
    return SpatialHasherState(
        slot_size=V(dtype=gs.qd_int, shape=(hasher.n_slots, n_batch)),
        slot_start=V(dtype=gs.qd_int, shape=(hasher.n_slots, n_batch)),
        cur_cnt=V(dtype=gs.qd_int, shape=(n_batch,)),
    )


@qd.func
def func_pos_to_grid(pos: qd.types.vector(3), hasher: qd.template()):
    return qd.floor(pos / hasher.cell_size, gs.qd_int)


@qd.func
def func_grid_to_slot(grid_id: qd.types.vector(3), hasher: qd.template()):
    return (
        grid_id[0] * hasher.grid_res[1] * hasher.grid_res[2] + grid_id[1] * hasher.grid_res[2] + grid_id[2]
    ) % hasher.n_slots


@qd.func
def func_pos_to_slot(pos: qd.types.vector(3), hasher: qd.template()):
    return func_grid_to_slot(func_pos_to_grid(pos, hasher), hasher)


@qd.func(requires_top_level=True)
def func_compute_reordered_idx(
    n: int,
    pos: qd.template(),
    active: qd.template(),
    reordered_idx: qd.template(),
    hasher_state: SpatialHasherState,
    hasher: qd.template(),
):
    for i_slot, i_b in qd.ndrange(hasher_state.slot_size.shape[0], hasher_state.cur_cnt.shape[0]):
        hasher_state.slot_size[i_slot, i_b] = 0
        hasher_state.slot_start[i_slot, i_b] = 0
    for i_b in range(hasher_state.cur_cnt.shape[0]):
        hasher_state.cur_cnt[i_b] = 0

    for i_n, i_b in qd.ndrange(n, hasher_state.cur_cnt.shape[0]):
        if active[i_n, i_b]:
            i_slot = func_pos_to_slot(pos[i_n, i_b], hasher)
            qd.atomic_add(hasher_state.slot_size[i_slot, i_b], 1)

    if qd.static(gs.backend == gs.cpu):
        for i_slot in range(hasher_state.slot_size.shape[0]):
            for i_b in range(hasher_state.cur_cnt.shape[0]):
                hasher_state.slot_start[i_slot, i_b] = qd.atomic_add(
                    hasher_state.cur_cnt[i_b], hasher_state.slot_size[i_slot, i_b]
                )
    else:
        n_blocks = (hasher_state.slot_size.shape[0] + BLOCK_DIM - 1) // BLOCK_DIM
        qd.loop_config(block_dim=BLOCK_DIM)
        for i in range(hasher_state.cur_cnt.shape[0] * n_blocks * BLOCK_DIM):
            tid = qd.simt.block.thread_idx()
            i_block = i // BLOCK_DIM
            i_b = i_block // n_blocks
            i_slot = (i_block % n_blocks) * BLOCK_DIM + tid
            slot_size = gs.qd_int(0)
            if i_slot < hasher_state.slot_size.shape[0]:
                slot_size = hasher_state.slot_size[i_slot, i_b]
            prefix = qd.simt.block.exclusive_add(slot_size, BLOCK_DIM, gs.qd_int)
            sh_base = qd.simt.block.SharedArray((1,), gs.qd_int)
            if tid == BLOCK_DIM - 1:
                sh_base[0] = qd.atomic_add(hasher_state.cur_cnt[i_b], prefix + slot_size)
            qd.simt.block.sync()
            if i_slot < hasher_state.slot_size.shape[0]:
                hasher_state.slot_start[i_slot, i_b] = sh_base[0] + prefix

    for i_n, i_b in qd.ndrange(n, hasher_state.cur_cnt.shape[0]):
        if active[i_n, i_b]:
            i_slot = func_pos_to_slot(pos[i_n, i_b], hasher)
            reordered_idx[i_n, i_b] = qd.atomic_add(hasher_state.slot_start[i_slot, i_b], 1)

    for i_slot, i_b in qd.ndrange(hasher_state.slot_size.shape[0], hasher_state.cur_cnt.shape[0]):
        hasher_state.slot_start[i_slot, i_b] -= hasher_state.slot_size[i_slot, i_b]


@qd.func
def func_for_all_neighbors(
    i_p: int,
    pos: qd.template(),
    task_range: float,
    ret: qd.template(),
    task: qd.template(),
    task_owner: qd.template(),
    i_b: int,
    hasher_state: SpatialHasherState,
    hasher: qd.template(),
):
    base = func_pos_to_grid(pos[i_p, i_b], hasher)
    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
        i_slot = func_grid_to_slot(base + offset, hasher)
        for j_p in range(
            hasher_state.slot_start[i_slot, i_b],
            hasher_state.slot_start[i_slot, i_b] + hasher_state.slot_size[i_slot, i_b],
        ):
            if i_p != j_p and (pos[i_p, i_b] - pos[j_p, i_b]).norm() < task_range:
                task(task_owner, i_p, j_p, ret, i_b)
