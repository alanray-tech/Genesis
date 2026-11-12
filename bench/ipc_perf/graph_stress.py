"""Stress the Quadrants graph features a device-driven Newton loop would rely on, outside compute-sanitizer.

Every round launches, in this order, and checks the result of each:
- a graph kernel with nested qd.graph.do_while loops (an outer Newton-like loop around an inner line-search-like loop),
  their trip counts read from device flags;
- a graph kernel with a qd.checkpoint inside a do_while loop that yields once (a capacity overflow), the host then
  resuming it from that checkpoint;
- a kernel compiled for this round alone, so that its module loads right after the graph kernels, the sequence that
  failed with CUDA_ERROR_UNKNOWN under compute-sanitizer (qd-lookup).
The offline cache is off, so every new kernel compiles and loads a new module.

    python graph_stress.py [--rounds 300]
"""

import argparse
import time

import numpy as np

import quadrants as qd

parser = argparse.ArgumentParser()
parser.add_argument("--rounds", type=int, default=300)
args = parser.parse_args()

qd.init(arch=qd.cuda, offline_cache=False)

N = 1024
x = qd.ndarray(qd.i32, shape=(N,))
outer_flag = qd.ndarray(qd.i32, shape=())
inner_flag = qd.ndarray(qd.i32, shape=())
outer_count = qd.ndarray(qd.i32, shape=())
inner_count = qd.ndarray(qd.i32, shape=())
limits = qd.ndarray(qd.i32, shape=(2,))
overflow = qd.ndarray(qd.i32, shape=())
yield_at = qd.ndarray(qd.i32, shape=())


@qd.kernel(graph=True)
def kernel_nested(
    outer_flag: qd.types.ndarray(qd.i32, ndim=0),
    inner_flag: qd.types.ndarray(qd.i32, ndim=0),
    outer_count: qd.types.ndarray(qd.i32, ndim=0),
    inner_count: qd.types.ndarray(qd.i32, ndim=0),
    limits: qd.types.ndarray(),
    x: qd.types.ndarray(),
):
    for _ in range(1):
        outer_count[()] = 0
        outer_flag[()] = 1
    while qd.graph.do_while(outer_flag):
        for _ in range(1):
            inner_count[()] = 0
            inner_flag[()] = 1
        while qd.graph.do_while(inner_flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
            for _ in range(1):
                inner_count[()] = inner_count[()] + 1
                inner_flag[()] = qd.i32(inner_count[()] < limits[1])
        for _ in range(1):
            outer_count[()] = outer_count[()] + 1
            outer_flag[()] = qd.i32(outer_count[()] < limits[0])


@qd.kernel(graph=True, checkpoints=True)
def kernel_yield(
    outer_flag: qd.types.ndarray(qd.i32, ndim=0),
    outer_count: qd.types.ndarray(qd.i32, ndim=0),
    overflow: qd.types.ndarray(qd.i32, ndim=0),
    yield_at: qd.types.ndarray(qd.i32, ndim=0),
    x: qd.types.ndarray(),
):
    for _ in range(1):
        outer_count[()] = 0
        outer_flag[()] = 1
    while qd.graph.do_while(outer_flag):
        with qd.checkpoint(0, yield_on=overflow):
            for _ in range(1):
                if outer_count[()] == yield_at[()]:
                    overflow[()] = 1
        for i in range(x.shape[0]):
            x[i] = x[i] + 1
        for _ in range(1):
            outer_count[()] = outer_count[()] + 1
            outer_flag[()] = qd.i32(outer_count[()] < 10)


def make_kernel(k: int):
    @qd.kernel
    def kernel_fresh(x: qd.types.ndarray()):
        for i in range(x.shape[0]):
            x[i] = x[i] + k

    return kernel_fresh


rng = np.random.default_rng(0)
failures = []
start = time.perf_counter()
for i_round in range(args.rounds):
    x.from_numpy(np.zeros(N, dtype=np.int32))
    n_outer, n_inner = (int(v) for v in rng.integers(1, 9, size=2))
    limits.from_numpy(np.array([n_outer, n_inner], dtype=np.int32))
    kernel_nested(outer_flag, inner_flag, outer_count, inner_count, limits, x)
    expected = n_outer * n_inner
    overflow[()] = 0
    yield_at[()] = int(rng.integers(0, 10))
    status = kernel_yield(outer_flag, outer_count, overflow, yield_at, x)
    n_yields = 0
    while status.yielded:
        n_yields += 1
        yield_at[()] = -1
        overflow[()] = 0
        status = kernel_yield.resume(outer_flag, outer_count, overflow, yield_at, x, from_checkpoint=status.checkpoint)
    expected += 10
    make_kernel(i_round + 1)(x)
    expected += i_round + 1
    values = x.to_numpy()
    if n_yields != 1 or not (values == expected).all():
        failures.append((i_round, n_yields, int(values.min()), int(values.max()), expected))
        print(f"round {i_round}: yields {n_yields}, x in [{values.min()}, {values.max()}], expected {expected}", flush=True)
    if i_round % 50 == 0:
        print(f"round {i_round} done, {time.perf_counter() - start:.1f} s", flush=True)
print(f"{args.rounds} rounds, {len(failures)} failures, {time.perf_counter() - start:.1f} s", flush=True)
