# Component-partitioned PCG

Genesis World provides two global linear solvers for the Quadrants Newton
pipeline:

```python
contact_config = {
    "linear_system/solver": "partition_pcg",  # default
    "linear_system/partition_sv_max_iter": 64,
}
```

Set `linear_system/solver` to `linear_pcg` to retain the original global-PCG
path for A/B measurements.

## Partition

`partition_pcg` treats each 3-DOF BCOO row as a graph vertex.  At engine
initialization it computes a static fast-SV seed from:

- QCloth triangle and bending-hinge stencils;
- Genesis rigid DOFs in the same environment;
- CGQ MinCoo root and joint DOFs in the same forest tree;
- both proxy rows and their owning mechanism.

At every Newton solve, the current sorted BCOO row/column pairs add elastic
and live contact edges to that seed.  Fast-SV, path compression, dense
relabeling, and the PCG loop all execute in the Quadrants CUDA graph.  There
is no host iteration or runtime dependency on `cuda-graph-qipc`.

The explicit rigid/forest/proxy edges are required because the rigid Hessian
is matrix-free: those couplings do not appear in BCOO.  They also connect the
physical proxy rows to the reduced generalized coordinates used by the
existing `P^T H P` operator.  A 3-DOF row that straddles two logical rigid
groups conservatively merges both groups.

If fast-SV reaches `linear_system/partition_sv_max_iter` before convergence,
the device path collapses to one component.  This preserves the global-PCG
mathematics instead of applying an unsafe partial mask.

## Masked PCG

Each component owns independent `rz`, `pAp`, `alpha`, and `beta` values.
Converged components are progressively removed from BCOO SpMV, FEM
preconditioning, Genesis rigid matrix-free work, MinCoo forest
expand/restrict and tree preconditioning, proxy projection, segmented dot
products, and vector updates.  Each forest tree stores one representative
block row; static connectivity guarantees that every generalized coordinate
and owned proxy row has the same live component label.  The initial threshold
follows CGQ:

```text
tol[c] = tol_rate * max(abs(rz[c]), tol_rate * sum(abs(rz)))
```

The global iteration count is the maximum work required by any component.
Existing `n_iterations`, `is_failed`, residual, and preconditioned-residual
diagnostics remain available through `engine.pcg_solver.linear_pcg`.

## Benchmark

Use the matched Franka-cloth benchmark to compare solvers:

```powershell
python examples/newton_coupling/franka_cloth_cgq_benchmark.py `
  --rigid-backend cgq_mincoo --linear-solver partition_pcg `
  --output output/partition.json
python examples/newton_coupling/franka_cloth_cgq_benchmark.py `
  --rigid-backend cgq_mincoo --linear-solver linear_pcg `
  --output output/linear.json
```

The output records frame timings, Newton/PCG counts, selected solver, and
per-frame component counts. Compare matching rigid backend, forest, BVH, contact,
warmup, and frame settings; partitioning is intended to reduce work after
components converge, not necessarily reduce the maximum PCG iteration count.

The Franka teleoperation examples accept the same
`--linear-solver {partition_pcg,linear_pcg}` switch for interactive checks.

## Current boundary

This implementation partitions only the Hessian solve.  CCD remains on the
existing global path; `contact/ccd_partition` is not wired to these labels in
this change.
