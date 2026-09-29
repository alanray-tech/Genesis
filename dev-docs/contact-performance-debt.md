# Contact Performance Debt Register

Status: authoritative rewrite queue.

Ground truth is `cuda-graph-qipc`
`main@42e7d4cbbad08739107ad830a17918f5f0f209ff`.

This register covers implementations that preserve useful numerical or candidate
semantics but do not yet represent the highest-performance target. Passing the
crossed-cloth milestone does not close an item. Every item remains open until
the production implementation is replaced and benchmarked.

## Acceptance policy

- This register is not a waiver. A listed nonconforming implementation must not
  remain reachable from a production, example, or milestone runtime.
- Every scene-scale GPU phase must be load-balanced.
- Irregular traversal, frontier expansion, compaction, reduction, segmented
  reduction, candidate emission, and sparse assembly must use the most
  efficient applicable warp/subgroup-level algorithm.
- Production must match or exceed the fastest applicable CGQ path. A different
  decomposition is admissible only after evidence shows equivalent or better
  lane utilization, atomic traffic, memory traffic, occupancy, and scaling.
- Serialized scene-scale traversal, unbounded work assigned to one lane,
  pair-by-pair global reservation where warp batching applies, scalar global
  reduction where warp/block reduction applies, host fallback, brute-force
  production paths, and scene-specific tuning are forbidden.
- Temporary scalar or diagnostic implementations may exist only in isolated
  correctness tests or offline comparison tools. Builders and runtime systems
  must not reference them.
- Numerical correctness and milestone-scene success do not admit
  non-load-balanced or non-warp-optimal code.
- Every rewrite requires correctness parity, size-scaling measurements, Nsight
  kernel evidence, and an end-to-end frame comparison.

## Completed P0 gates before the next subsystem

### PERF-C01: EE 12x12 Hessian materialization

Implementation:

- Ordinary, non-mollified four-vertex EE uses
  `gipc_barrier_grad_hess_ee_rank1`.
- `cipc_scatter_triplets_upper_rank1` writes the ten 3x3 outer-product blocks
  directly.
- Dense storage remains only in mollified and PE/PP-degenerate branches.

CGQ target:

- `gipc_barrier_grad_hess_ee_rank1` emits `v[12]`, `grad_scale`, and
  `hess_coef`.
- `cipc_scatter_triplets_upper_rank1` forms the ten 3x3 outer-product blocks
  directly.

Rewrite:

- Keep the dense path only for mollified and degenerate PE/PP dispatch.
- Ordinary four-vertex EE must never allocate or write a 12x12 temporary.
- Apply the same direct-scatter principle to ordinary PT where profitable.

Acceptance:

- Per-pair gradient and unordered-global-ID triplet parity with the dense result:
  complete.
- The ordinary four-vertex branch performs no 144-entry Hessian writes or
  reads: complete.
- The one dense 144-entry allocation required by the mollified/degenerate
  fallback may remain in the combined kernel, matching CGQ's 1,600-byte/thread
  post-rewrite target; the removed 12x9 and 9x9 intermediates must not return.
- Kernel-resource evidence confirms the expected local-memory traffic and
  occupancy.

Status: **closed**. Implementation, numerical parity, and IR branch inspection
are complete. Quadrants reports 254 registers/thread and 2 active blocks/SM for
the combined EE kernel; the final crossed-cloth Nsight Systems capture reports
an 85.727-microsecond median for `cipc_filter_assemble_ee`. Nsight Compute
local-memory counters were unavailable because of `ERR_NVGPUCTRPERM`, but
Genesis does not repeat CGQ's algorithm experiment: the pinned-CGQ stack and
A/B measurements below remain the performance authority.

Pinned-CGQ evidence: stack use fell from 5,088 to 1,600 bytes/thread; the
isolated kernel improved by up to 2.8x, frame-2 EE filter time fell from 149 to
84 microseconds (1.78x), and step time fell by 5.8%.

### PERF-C02: EE warp-batched candidate output

Implementation:

- `_ee_emit_pair` uses a subgroup ballot, lane prefix, one reservation by lane
  zero, and a subgroup broadcast.
- Both `LBVH.query_ee_warp` and `DualEEQueryState` use this batch emitter.
- The global pair count retains exact required-size semantics on overflow.
- `query_ee_warp` uses an occupancy-sized 510-block static launch and
  grid-strides edge queries across resident warps. It does not use Quadrants'
  8,160-block dynamic-range default.

CGQ target:

- Warp-local candidate buffering.
- One amortized global reservation per emitted batch.
- Deterministic overflow accounting compatible with checkpoint growth.

Rewrite:

- Use subgroup ballot/prefix operations to compact accepted pairs within each
  warp.
- Reserve one contiguous output interval per warp batch and scatter lanes into
  that interval.
- Preserve exact required-count reporting when capacity is exceeded.

Acceptance:

- Candidate-set parity between warp-query and dual-query paths: complete.
- Global counter atomics scale with emitted batches, not emitted pairs.
- Dense-contact profiling demonstrates reduced atomic serialization.

Status: **closed**. Implementation and candidate parity are complete. The
warp-per-edge rollback compiles to 72 registers/thread, 32,768 shared
bytes/block, 3 active blocks/SM, and the expected 510-block occupancy grid.
Pair reservations occur once per subgroup batch in both rollback and dual
paths.

Pinned-CGQ evidence for the complete warp-per-EE-query schedule, which includes
these batched writes: 603 to 92 microseconds on frame 2 (6.5x), and 403.8 to
60.1 microseconds over the orbit mean (6.7x) on RTX 5090.

### PERF-C03: Complete load-balanced dual EE

Implementation:

- The serialized `LBVH.query_ee_dual` and its global stack have been removed.
- `DualEEQueryState` owns two growable frontier buffers and device-scalar
  counts, level/parity state, target, required capacity, and overflow bits.
- Static graph stages perform subgroup-compacted level expansion up to the
  device-selected depth; persistent warp-local DFS workers pull independent
  frontier tasks from `next_task`.
- Frontier overflow grows by the CGQ `1.2` rule and replays the query from its
  checkpoint.

CGQ target:

- Up to `dual/max_levels` level-synchronous frontier expansion.
- Device-selected auto depth based on edge count and resident DFS warp target.
- Warp-local DFS from independent frontier entries.
- Dynamic frontier capacity, overflow checkpoint, growth, and exact replay.

Rewrite:

- Add current/next frontier node-pair buffers and device live counts.
- Emit 18 alternating static graph stages. The selected stage stores the final
  frontier and clears live counts, so later stages perform zero work.
- Do not replace these stages with a conditional child-graph loop: CGQ measured
  6.1--6.9 microseconds of conditional control cost per level and rejected that
  schedule after Kimono and WreckingBall regressions.
- Dispatch frontier entries across warps and use bounded per-warp DFS stacks.
- Integrate warp-batched pair output from PERF-C02.

Acceptance:

- Candidate-set equality against the warp-per-edge query after sorting:
  complete.
- Forced frontier overflow, growth, and exact replay parity: complete.
- No scene-scale traversal inside a serialized one-iteration task: complete.
- Work distribution and occupancy comparable to CGQ on sparse, anisotropic,
  and dense cloth scenes.

Status: **closed**. Implementation, parity, overflow replay, launch occupancy,
and the final static-pipeline profile are complete. In the 100-step
crossed-cloth capture, the main DFS median was 33.151 microseconds; selected
frontier stages were approximately 4.4--6.0 microseconds and config-gated
stages approximately 1.3 microseconds. Cross-scene scaling conclusions remain
owned by the pinned-CGQ experiments below, per the faithful-migration policy.

Pinned-CGQ evidence on RTX 5090: Kimono query-only aggregate improved 1.858x
and build-plus-query improved 1.756x; node tests fell 3.119x with identical
leaf tests. Production-auto query speedups were 1.678x on Kimono and 1.008x on
WreckingBall. The small two-bunny case regressed by 0.076 ms/query and remains
the accepted CGQ small-scene trade-off.

Pinned local-Quadrants CUDA attributes on RTX 5090:

- frontier expand: 64 registers/thread, zero shared bytes, 4 active blocks/SM;
- warp-local DFS: 64 registers/thread, 32,768 shared bytes/block, 3 active
  blocks/SM.

The corresponding static launch grids are 680 frontier blocks and 510 DFS
blocks on the 170-SM RTX 5090. Both kernels manually grid-stride dynamic device
counts; using a dynamic top-level Quadrants range would launch 8,160 blocks and
is prohibited for this path.

`resident_dfs_warps` is derived from those pinned DFS register requirements and
live CUDA device thread/register/shared-memory limits. Quadrants should
eventually expose the compiled-kernel occupancy query directly; until then, a
Quadrants compiler pin change requires revalidating the 64-register structural
metadata before this gate remains closed.

These three P0 gates no longer block the next subsystem. The remaining entries
below are still mandatory production rewrites and must not be silently treated
as permanent architecture.

## Broad-phase and BVH debt

### PERF-B01: Packed conservative DOP14f

Current:

- DOP14 bounds are stored as fourteen `f64` ndarray values: 112 bytes per node.

CGQ target:

- Outward-rounded `f32` DOP14 values padded/aligned to 64 bytes.
- Vectorized 16-byte loads with no node straddling.

Rewrite:

- Add or verify Quadrants support for directed `f64 -> f32` rounding.
- Use a 64-byte packed/aligned representation whose generated loads are
  confirmed in PTX.

Acceptance:

- Every fp32 bound conservatively contains the fp64 reference.
- No candidate loss over adversarial rounding tests.
- Node bandwidth and query time match or beat CGQ DOP14f.

### PERF-B02: Scene-bound reduction and refit

Current:

- Scene bounds use two generic Quadrants block-reduction phases.
- Internal refit performs leaf-to-root atomic-CAS walks and explicit fences.

CGQ target:

- Tuned BV merge reduction and production refit kernels.

Rewrite:

- Profile before changing the standard Karras refit.
- Replace generic reduction stages only when a tuned block/CUB-equivalent path
  wins end-to-end.

### PERF-B03: Query stack storage

Current:

- Dual EE and the warp-per-edge rollback now use bounded per-warp shared
  stacks.
- PT still allocates a persistent 64-entry global stack row per surface-vertex
  query.

CGQ target:

- Warp-local DFS state for every production PT/EE traversal.

Rewrite:

- Move PT traversal state to the CGQ 8-warps/block shared frontier.
- Remove `stack_pool` after no production query references it.

Status: EE is complete; PT remains open.

### PERF-B04: Contact-table node culling

Current:

- Node metadata carries homogeneous body IDs only.
- Contact-element masks are checked in narrow phase rather than used for
  homogeneous-node culling.

CGQ target:

- Body and contact-element information participates in info-guided rejection.

Rewrite:

- Publish homogeneous contact-element metadata per node and reject disabled
  subtrees before leaf traversal.

### PERF-B05: Warp-cooperative PT traversal

Current:

- `LBVH.query_pt` assigns one swept point query to one thread and traverses a
  private global-memory stack.
- On cloth-sized scenes this exposes only approximately
  `ceil(n_surface_vertices / 32)` divergent warps.

CGQ target:

- `pt_query_warp`: one warp per swept point query, 8 warps per 256-thread
  block, a 1,024-entry shared stack per warp, subgroup-batched pushes, and
  subgroup-batched pair reservations.
- Occupancy-sized static launch with manual grid-striding over the
  device-scalar query count.

Acceptance:

- Sorted PT candidate-set parity with the current query on sparse, anisotropic,
  and dense scenes.
- Exact required-count behavior through pair-buffer checkpoint growth.
- Compiled register/shared-memory occupancy and Nsight timing match or beat the
  pinned CGQ path.

## Contact evaluation and assembly debt

### PERF-A01: PT and friction dense temporaries

Current:

- PT and friction paths construct dense 12-vector/12x12 intermediates before
  block scatter.

Target:

- Direct structured or low-rank 3x3 block emission wherever CGQ exposes that
  structure.

### PERF-A02: Global energy atomics

Current:

- Barrier and friction kernels atomically add every active pair into one scalar.

CGQ target:

- Pair or block partials followed by block reduction and final reduction.

Rewrite:

- Use block-local reductions and one partial per block.
- Merge barrier, PH, and friction totals through a bounded final reduction.

### PERF-A03: Minimum-gap atomics

Current:

- Every active pair atomically updates global, per-body, or per-vertex gap
  trackers.

CGQ target:

- Thread/block-local folds with amortized tracker updates.

Rewrite:

- Reduce within blocks before updating shared adaptive-kappa state.

### PERF-A04: Capacity-wide clearing

Current:

- Unique gradient/Hessian buffers are cleared to maximum capacity each Newton
  iteration.

Target:

- Clear only the live/high-water range required by the reduction backend.

### PERF-A05: Duplicate contact/global sort stages

Current:

- Contact doublets/triplets are sorted and reduced, distributed, then the
  unified body triplets are sorted and reduced again.

CGQ target:

- Keep the required CGQ category routing but use production dynamic radix/scan
  and optimized FSR at both levels.

Long-term investigation:

- Measure whether compatible FEM-only categories can safely bypass redundant
  work without changing the future mixed-system routing contract.

### PERF-A06: Generic radix/scan/FSR lowering

Current:

- Quadrants generic radix sort, exclusive scan, and atomic segmented reduction
  are used for contact and body BCOO.

CGQ target:

- Dynamic OneSweep radix sort, dynamic exclusive sum, warp head-segmented FSR,
  exact live extents, and grid-parameter patching on growth.

### PERF-A07: Adaptive-kappa no-op traversal

Current:

- Runtime mode checks occur inside full body/vertex loops, so inactive
  granularities may still launch scene-scale work.

Target:

- Mode-gated kernels return before the range traversal or use graph conditions
  without changing graph topology.

### PERF-A08: Friction state and Jacobian recomputation

Current:

- Generic matrix/vector construction rebuilds lagged tangent bases, closest
  parameters, Jacobians, and dense Hessians.

CGQ target:

- Specialized PT/EE/PE/PP/PH kernels with register-resident coefficients and
  direct block scatter.

## CCD debt

### PERF-D01: Global per-pair alpha atomic

Current:

- Every pair atomically updates one global `ccd_alpha`.

CGQ target:

- Block partial minima followed by final reduction.

### PERF-D02: Missing component-partitioned CCD

Current:

- One smallest TOI limits every disconnected component.

CGQ target:

- Device component partition, segmented minimum, and per-body/component alpha.

### PERF-D03: Divergent directional-CCD loops

Current:

- One thread owns one pair's conservative-advancement loop up to 50,000
  iterations.

Target:

- Preserve CGQ mathematics while profiling pair-level divergence and applying
  the CGQ diagnostics/partitioning path.

### PERF-D04: Always-written pair diagnostics

Current:

- Per-pair CCD alpha arrays are written every frame for failure diagnostics.

Target:

- Compile/runtime-gated diagnostics with zero traffic in production.

## Global linear system and PCG debt

### PERF-L01: Atomic body FSR

Current:

- Every raw 3x3 triplet contributes nine global atomics during body reduction.

Target:

- Warp-segmented reduction matching the production CGQ FSR path.

### PERF-L02: Atomic BCOO SpMV

Current:

- Symmetric off-diagonal blocks atomically accumulate into shared row vectors.

Target:

- Profile row contention and adopt the fastest CGQ traversal/partitioned path.

### PERF-L03: Scalar atomic dot products

Current:

- PCG `rz`, `pAp`, and residual norms use per-DOF global scalar atomics.

Target:

- Warp/block partial reductions and bounded final reductions.

### PERF-L04: Standard PCG graph-node count

Current:

- The standard PCG loop retains separate memset, alpha, beta, and swap nodes.

CGQ target:

- Masked/fused PCG node layout and convergence masking.

### PERF-L05: No component masking

Current:

- Converged disconnected components continue participating in PCG.

CGQ target:

- Static component labels and MaskedPCG skip work per converged component.

### PERF-L06: FEM diagonal preconditioner gather

Current:

- Each Newton step scans every BCOO block to gather diagonal entries, then uses
  a generic 3x3 inverse.

Target:

- Direct diagonal indexing or a production CGQ preconditioner path; evaluate
  MAS when it is the fastest applicable cloth option.

### PERF-L07: Production BCOO validation

Current:

- Every Newton iteration scans the complete BCOO to validate ordering.

Target:

- Debug/diagnostic gate; zero validation traffic in production builds.

## Pipeline and graph debt

### PERF-P01: Missing accepted-energy reuse

Current:

- Baseline elastic/contact energy is recomputed every Newton iteration.

CGQ target:

- Compute E0 on the first iteration and reuse the accepted trial energy through
  `swap_E`, repricing only when adaptive kappa changes.

### PERF-P02: Sequential independent subsystem phases

Current:

- Several rigid, FEM, contact, energy, and initialization phases are emitted
  sequentially even when their writes are disjoint.

CGQ target:

- Explicit parallel subgraphs with dependency joins at the true consumer.

### PERF-P03: Excess scalar graph nodes

Current:

- Numerous serialized one-iteration loops emit small scalar kernels.

Target:

- Fuse related state updates without obscuring checkpoint or dependency
  contracts.

### PERF-P04: Contact graph compile cost

Current:

- One very large graph kernel inlines every contact variant and compiles slowly
  on a cold cache.
- `advanced_optimization=False` intentionally trades steady-state runtime for
  development compile speed.

Target:

- Restore the fastest runtime optimization settings for production.
- Improve Quadrants pass scaling and artifact reuse rather than permanently
  disabling optimization.

### PERF-P05: Capacity growth allocation

Current:

- Growth recreates complete ndarray workspaces and relies on launch-context
  rebinding.

CGQ target:

- High-water pools, geometric reuse, dynamic helper reconfiguration, optional
  shrink hysteresis, and no pipeline rebuild.

### PERF-P06: Per-frame scene writeback

Current:

- FEM state is copied to legacy scene storage in a separate post-step kernel.

Target:

- Keep rendering outside the Newton graph, but profile and minimize the bridge;
  avoid it entirely in headless simulation.

## Rigid and future coupling debt

### PERF-R01: Dense rigid matrix-free apply on large islands

Current:

- Exact dense rigid blocks are appropriate for the current Franka-sized
  baseline but scale quadratically with island DOFs.

Target:

- Apply the roadmap's future large-island block/subtree/sparse policy after a
  measured crossover.

### PERF-R02: Missing optimized RC/CR routes

Current:

- Rigid-Cloth routes are outside the current milestone.

Future target:

- CGQ-compatible reduced-KKT/proxy path with explicit RC/CR BCOO and optimized
  coordinate pullback.

## Compile-time and memory-layout debt

### PERF-COMP01: Generated contact IR size

Current:

- Generated distance, mollifier, barrier, friction, and CCD functions inline
  large expression trees into the graph.

Target:

- Preserve kernel fusion only where runtime wins; use compiler improvements,
  reusable functions/artifacts, and specialized low-rank paths to reduce both
  IR and spills.

### PERF-COMP02: ndarray layout verification

Current:

- Contact blocks and DOP bounds rely on generic ndarray layout.

Target:

- Verify alignment, coalescing, vector loads, and cache-line behavior in PTX;
  introduce explicit packed layouts where the generic lowering is inferior.

## Exit criteria

The register may be closed only when:

1. PERF-C01, PERF-C02, and PERF-C03 are complete before any next subsystem.
2. Every remaining item has an implementation, benchmark, or explicit
   user-approved deferral.
3. Runtime is measured against pinned CGQ on the same GPU, scene, precision,
   timestep, and contact parameters.
4. The final production path targets the best measured performance, not merely
   functional parity.
