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

The heterogeneous-body parity gate additionally requires Morton permutation
to reorder leaf `node_body_id` metadata together with AABBs and element IDs.
Without that reorder, dual traversal can incorrectly cull every cloth-rigid
EE subtree while warp traversal remains correct. The interleaved-body
dual/warp regression covers this exact failure.

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

Implementation:

- Production DOP14 bounds are fourteen active `f32` values in a sixteen-lane
  ndarray row: 64-byte row stride and 64-byte base alignment.
- Point projection and final radius expansion convert from `f64` with explicit
  round-down/round-up helpers. Every packed leaf bound conservatively contains
  the retained fourteen-`f64` reference.
- `extras/bvh/genesis_legacy_fp64_bounds=1` retains the former 112-byte node
  representation as an explicit A/B oracle.

CGQ target:

- Outward-rounded `f32` DOP14 values padded/aligned to 64 bytes.
- Vectorized 16-byte loads with no node straddling.

Correctness:

- Directed-conversion tests cover exact `f32` values, adjacent `f64` values,
  subnormals, signed zero, finite extremes, and overflow-adjacent values.
- Adversarial moving-triangle leaf tests prove outward containment against the
  retained `f64` path.
- Packed/direct, `f64`/direct, and `f64`/legacy Franka-Cloth paths all pass the
  broad-phase suite and one-frame production smoke gate. The packed and
  retained paths both execute Newton 2, total PCG 6, and line search 0.

Matched profile proof:

- Nsight Systems window: frames 20--28, 18 Newton evaluations, RTX 5090.
- Native-`f32` scene reduction costs `20.448 us/Newton` versus
  `48.816 us/Newton` for the retained `f64` bounds (`2.387x`).
- Direct internal refit costs `95.662 us/Newton` with packed bounds versus
  `132.655 us/Newton` with `f64` bounds (`1.387x`).
- Warp PT costs `42.911 us/Newton` with packed bounds versus
  `96.751 us/Newton` with `f64` bounds (`2.255x`).
- CGQ's corresponding PT kernel costs `65.551 us/Newton` in the pinned
  capture; the packed Genesis traversal is faster in this matched window.

Ground-truth discrepancy:

- CGQ `DOP14f::expand` rounds the radius upward and then performs
  round-to-nearest `f32` subtraction/addition. The final arithmetic can move a
  lower or upper bound inward by one ULP. Genesis deliberately rounds the
  final expanded values outward. This is a reported safety correction, not an
  unreported naming or algorithm deviation.

Status: **closed at the Genesis storage/algorithm layer**. A generated-PTX
probe of one complete sixteen-lane row load shows sixteen scalar `ld.b32`
instructions at offsets 0--60, not four vector loads. The ndarray has the
required alignment and stride, but Quadrants does not preserve that fact into
vectorized memory operations. Native vector-load parity remains open under
PERF-COMP02 and the Quadrants lowering layer.

### PERF-B02: Scene-bound reduction and refit

Implementation:

- Packed bounds use native-`f32` two-phase scene reduction; the retained
  `f64` path remains available for A/B comparison.
- Every completed second child directly writes all parent min/max lanes in one
  refit visit. The former initialize-parent plus two child-merge sequence is
  retained behind `extras/bvh/genesis_legacy_refit=1`.
- Body metadata propagates in the same leaf-to-root visit, so Genesis does not
  require CGQ's separate post-refit body-propagation phase.

CGQ target:

- Tuned BV merge reduction and production refit kernels.

Matched profile proof:

- With `f64` bounds held fixed, direct refit costs `132.655 us/Newton` versus
  `199.870 us/Newton` for initialize-plus-two-merge (`1.507x`).
- Packing then reduces direct refit from `132.655` to `95.662 us/Newton`
  (`1.387x`).
- Packed/direct named broad-phase work totals `422.971 us/Newton` versus
  `608.777 us/Newton` for `f64`/legacy (`1.439x`, 30.5% lower).
- CGQ refit is `61.984 us/Newton`, plus `40.512 us/Newton` for its separate
  body propagation. Genesis's combined `95.662 us/Newton` path is slightly
  faster in the pinned captures.
- Whole captured GPU-kernel time changes from `178.069` to `170.420 ms`
  (4.30% lower), but small contact/PCG trajectory differences make the
  equal-stage named timings above the acceptance evidence. Wall medians
  (`24.913` versus `24.888 ms`) are measurement noise and are not claimed as
  layer speedup.

Status: **closed at the Genesis algorithm layer**. The remaining small
scene-reduction and reorder differences are launch/grid/lowering costs assigned
to the Quadrants optimization layer.

### PERF-B03: Query stack storage

Current:

- Production dual EE, warp-per-edge EE, and PT traversal use bounded per-warp
  shared frontiers.
- The initialization-only exact ET checker currently reuses one persistent
  64-entry global stack row per surface-edge query.

CGQ target:

- Warp-local DFS state for every production PT/EE traversal.

Status: **PT and EE closed** by PERF-B05 and PERF-C03. Only the
initialization-only exact ET diagnostic stack remains open; it must be lowered
to CGQ's fixed per-thread representation after confirming generated local
memory traffic does not regress. The stale claim that production PT used a
global stack is removed.

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

Implementation:

- `query_pt_warp` assigns one warp to each swept point query, uses eight warps
  per 256-thread block and one 1,024-entry shared frontier per warp, and
  grid-strides over `n_surf_verts[()]`.
- Internal-node pushes and emitted PT pairs use subgroup ballot/prefix
  compaction. Each emitted batch performs one global count reservation while
  retaining exact required-count semantics on pair-buffer overflow.
- Exact one-leaf handling matches CGQ. Frontier saturation uses a device
  invariant, rather than truncating candidates or abusing pair-buffer growth.
- Query projection bounds, including rigid-proxy path inflation, are computed
  once per warp and reused for every DOP overlap.
- `bvh/pt_query=batched` retains the former one-thread/global-stack traversal as
  an explicit scene-config A/B path; production defaults to `warp`.

Correctness:

- Sorted candidate-set parity against the retained batched traversal is
  covered for AABB/DOP14, a one-leaf tree, and a dense non-power-of-two tree.
- The dense regression also forces pair-buffer overflow and proves both paths
  report the same complete required count.
- In the matched Franka-cloth runs, PT/EE/active-contact counts, Newton/PCG/line
  search/CCD sequences, and accepted steps are identical. State and proxy
  residual differences remain at floating-point roundoff.

Matched profile proof:

- Nsight Systems capture window: frames 20--28, 18 Newton evaluations, RTX
  5090, with the same updated predicate/path-inflation semantics on both paths.
- The retained batched PT kernel costs `2.731 ms`, or `151.721 us/Newton`.
- Warp PT costs `1.748 ms`, or `97.109 us/Newton`: `1.562x` faster, `36.00%`
  lower, and `0.983 ms` removed from the captured query stage.
- Nsight reports 128 registers/thread, zero local memory, 32,768 shared bytes
  per block, and a 340-block launch: exactly two resident blocks per SM on the
  170-SM RTX 5090. The earlier guessed 72-register launch metadata produced
  510 blocks and was rejected after profiling.
- Pinned CGQ warp PT is `65.649 us/Newton`, with 255 registers/thread, 32,768
  shared bytes, and 170 blocks. Genesis remains `1.479x` slower; the next
  packed-DOP14f layer addresses the 112-byte fp64 node loads that remain in
  this otherwise aligned traversal.
- Whole-window timings from the two separate captures are not used as the
  layer speedup: unchanged kernels varied by far more than the `0.983 ms`
  removed PT work, indicating run-wide clock/thermal variance. The equal-work
  named-kernel result above is the acceptance measurement.

Status: **closed at the traversal-schedule layer**. The one-thread production
path is gone. Remaining native-CGQ parity is tracked by PERF-B01/B02 and the
Quadrants grid/lowering layer.

### PERF-B06: LBVH Morton dynamic OneSweep

Implementation:

- Triangle and edge LBVH Morton keys use the shared `DynamicRadixSort` u64
  OneSweep path with `n_prims[()]` as the live device-scalar extent.
- Morton generation initializes only that live range. No mask or padded
  sentinel range participates in the production sort.
- `extras/sort_reduce/genesis_legacy=1` retains the former padded generic
  Quadrants radix sort as the explicit comparison path.
- Permutation storage is `i32`, matching CGQ's OneSweep accessor contract.

Correctness:

- Stable key and permutation parity with the retained generic path is covered
  with a non-power-of-two live count and duplicate keys.
- The full broad-phase suite and dynamic-radix suite pass.
- In the matched Franka-cloth trajectory, Newton, PCG, line-search, CCD, and
  contact-count sequences are unchanged. Displacements and proxy residuals
  agree to floating-point roundoff.

Matched profile proof:

- Nsight Systems capture window: frames 20--28, 18 Newton evaluations, RTX
  5090, the same scene/configuration before and after the rewrite.
- The two generic Morton sort stages cost `16.105 ms`, with 288 generated
  variants and 5,184 kernel instances.
- Dynamic OneSweep costs `6.888 ms`, with 72 generated variants and 1,296
  instances: `2.338x` faster, `57.23%` lower, and `9.217 ms` removed from the
  captured window.
- Whole-window GPU kernel time falls from `188.504` to `177.448 ms`
  (`1.062x`, `5.87%` lower); total kernel instances fall from 40,462 to
  36,538.
- Pinned CGQ Morton OneSweep costs `2.119 ms`, with 7 native variants and 756
  instances. The remaining `3.251x` sort-stage gap is generated-code,
  scalar bound-helper, and small-kernel lowering debt assigned to the
  Quadrants optimization layer; the occupancy-grid portion is closed by the
  later resident-grid rewrite. None is admissible as a scene-specific Genesis
  workaround.

Status: **closed at the Genesis application layer**. The generic production
sort is gone; native-CGQ parity remains open under the Quadrants
range/grid/lowering work.

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

Implementation:

- Contact doublets, contact triplets, and global body BCOO use a
  Quadrants implementation of CGQ's dynamic OneSweep radix sort, decoupled
  lookback exclusive sum, and warp head-segmented FSR.
- Every live extent is a zero-dimensional device scalar. The sort and scan
  launch ranges are derived from those scalars; allocation growth does not
  bake a Python capacity into the compiled graph.
- Contact assembly starts at CGQ's 4,865-entry floor, grows by checkpoint, and
  shrinks its active padded extent with hysteresis. One integer extent, not a
  mask, describes valid padded work.
- `extras/sort_reduce/genesis_legacy=1` retains the former generic
  radix/scan/atomic-reduce path as an explicit A/B oracle.

CGQ target:

- Dynamic OneSweep radix sort, dynamic exclusive sum, warp head-segmented FSR,
  exact live extents, and grid-parameter patching on growth.

Correctness:

- Stable u32/u64 key and permutation parity is covered at zero, warp, block,
  CGQ floor, 100k, 500k, and 1M live counts.
- Scan and FSR correctness are covered through 1M entries.
- A growth regression runs one compiled graph at capacity 4,865, replaces its
  storage with capacity 20,000, and verifies the complete stable sort. This
  caught and removed an invalid first implementation that froze the launch
  extent in Python; that implementation failed the Franka trajectory when
  contact storage grew.
- Contact legacy/new A/B, checkpoint growth, rigid-proxy assembly growth, and
  the 100-frame Franka-Cloth trajectory pass.

Performance:

- The matched nine-frame Franka window executes 18 Newton evaluations. The
  complete doublet/triplet/body sort-reduce stages fall from 24.881 ms in the
  retained current fallback to 12.534 ms (`1.985x`), while variants fall from
  432 to 131 and instances from 7,776 to 2,358.
- Against the pre-layer profile, the same stages fall from 30.421 to
  12.534 ms (`2.427x`). Total GPU time falls from 216.494 to 190.349 ms
  (`1.137x`).
- Synchronized 100-frame medians are 27.019 ms for the retained fallback and
  26.323 ms for OneSweep/scan/FSR (`1.026x`). Against the pre-layer
  29.165-ms result, the complete layer is `1.108x` faster.
- Numerical work remains matched: both paths execute exactly two Newton
  evaluations per frame; mean total PCG work is 72.44 versus 72.38.
- Device-only u64 sort coverage follows the CGQ-defined `<100 ms` domain
  through 100M entries. At the simulation-scale 1M point, Genesis is
  approximately 0.353 ms versus CGQ's 0.254 ms. The remaining standalone and
  in-scene gap is generated-code, scalar-helper, and small-kernel lowering
  policy, not an application fallback.

Status: **closed at the Genesis application layer**. The remaining native-CGQ
gap is tracked as Quadrants scalar dynamic-range helpers and lowering work;
the occupancy-grid portion is closed by the resident-grid rewrite. No
scene-specific launch or frozen-capacity specialization is admissible.

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

### PERF-D03: Divergent conservative-advancement loops

Current:

- One thread owns one directional or rigid-screw pair's
  conservative-advancement loop up to 50,000 iterations.

Target:

- Preserve CGQ mathematics while profiling pair-level divergence and applying
  the CGQ diagnostics/partitioning path.

### PERF-D04: Always-written pair diagnostics

Current:

- Per-pair CCD alpha arrays are written every frame for failure diagnostics.

Target:

- Compile/runtime-gated diagnostics with zero traffic in production.

### PERF-D05: Missing co-rotating screw certificate

Current:

- The standard unpartitioned screw CCD path includes the absolute
  arc-curvature certificate but not CGQ's second certificate in the frame of
  the faster-rotating side.

CGQ target:

- Evaluate absolute and co-rotating conservative bounds and advance by the
  larger certified fraction. Co-moving rigid pairs must certify without
  exhausting the iteration budget.

## Global linear system and PCG debt

### PERF-L01: Atomic body FSR

Implementation:

- Each warp performs a head-segmented reduction for all nine 3x3 components.
- Only segment heads issue global atomics, matching CGQ's production FSR
  decomposition.
- Sentinel lanes contribute zero and never write; segments spanning warps
  produce one partial atomic per warp.

Target:

- Warp-segmented reduction matching the production CGQ FSR path.

Evidence:

- Doublet, triplet, and flattened body layouts match NumPy segmented sums
  through 1M entries.
- At 1M entries Genesis FSR plus live-output clearing measures approximately
  0.182 ms. CGQ's broader flags+scan+clear+FSR+extract benchmark measures
  approximately 0.173 ms; these figures deliberately do not claim an
  isolated-kernel ratio because the measured stage boundaries differ.

Status: **closed**.

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
- `SimEngine` preserves the caller's Quadrants compile policy instead of
  overriding it globally.
- The interactive Franka teleop defaults to `advanced_optimization=False` for
  development and exposes `--advanced-optimization` for production profiling.
  Headless performance benchmarks retain the Quadrants production default.

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

### PERF-P07: Eager exact-L1 directional reduction

Current:

- Exact-L1 activation and trial evaluation are lazy, but the physical
  gradient-direction dot reduction is still evaluated once per Newton
  iteration before line search.

CGQ target:

- Request the dot reduction only after physical-energy rejection also reduces
  an in-tolerance equality residual.
- Replay the same trial alpha after the conditional merit probe without
  charging a line-search iteration.

Rewrite:

- Add the equivalent conditional graph region when Quadrants exposes the
  required graph predicate, or an equally efficient static alternative with
  measured zero-work overhead.
- Do not serialize the DOF reduction inside the scalar line-search decision.

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

- Proxy-Cloth and proxy-proxy gradient/Hessian routes are implemented through
  the reduced-KKT physical BCOO layout.
- Route classification expands each vertex 3x3 block into generic
  translation/rotation blocks before the global sort/reduce.

Target:

- Match CGQ's specialized RC/CR block emission, geometric-term fusion, and
  coordinate pullback traffic.

### PERF-R03: Mapped forest preconditioner

Current:

- The source-agnostic mapped articulated preconditioner gathers proxy BCOO,
  factors the Genesis forest by reverse depth, and applies reverse/root/forward
  substitution entirely on device.
- Eligible trees apply reverse/root/forward substitution in one
  one-block-per-tree shared-memory kernel matching CGQ. The factor stage
  remains level scheduled because it runs once per Newton rather than once per
  PCG iteration.
- Each body owns its 6x6 factor work in the level schedule. Fixed authored
  MJCF links use exact rigid transport without a scalar Schur pivot.
- Standard global PCG consumes this preconditioner during the explicitly
  authorized pre-partition milestone.

CGQ target:

- `rigid_forest/fused=1`: one fused shared-memory tree solve for supported
  trees, warp-cooperative 6x6 root inversion, and the CGQ singleton path.

Rewrite:

- Port the fused/shared tree factor and apply schedules after the standard-PCG
  conformance gate.
- Preserve the current source-agnostic BCOO gather and exact fixed-link
  transport.
- Replace generic per-body 6x6 inverse lowering with the CGQ warp inverse.

Acceptance:

- Preconditioner action parity against the unfused factorization.
- Equal or lower PCG iterations on Franka-Cloth and rigid-proxy stress gates.
- Nsight evidence for occupancy, synchronization, and shared/global traffic.

#### Matched Franka-Cloth evidence (2026-09-29)

The matched benchmark uses an RTX 5090, explicit GPU synchronization, 20
discarded warmup frames, and three repeats of 100 measured frames. CGQ was
forced from its production defaults to `linear_system/solver=linear_pcg` and
`linear_system/preconditioner=diag`, so this comparison does not credit
MaskedPCG or MAS:

- CGQ median: 14.314 ms.
- Genesis median: 64.832 ms.
- Genesis/CGQ median ratio: 4.529x.

The steady-state Nsight window contains nine frames. CGQ accumulated 128.149
ms across 39,768 kernel instances; Genesis accumulated 375.756 ms across
116,914 instances. The corresponding ratios are 2.932x GPU time and 2.940x
kernel instances. CGQ executed 525 PCG iterations in the window. The Genesis
`pcg_zero_operator_direction` node executed 1,058 times, proving 2.015x as many
actual PCG iterations; the public Genesis frame counter currently reports only
one solve's count and must not be interpreted as the frame total.

Named-node attribution identifies the primary implementation gap:

- Genesis reduced rigid-forest matvec: 178.669 ms, 34,914 launches, or 168.9
  microseconds and 33 launches per PCG iteration.
- CGQ reduced rigid-forest matvec: 24.864 ms, 5,250 launches, or 47.4
  microseconds and 10 launches per iteration.
- The Genesis forest matvec is therefore 3.57x slower per iteration before the
  additional 2.015x iteration-count difference.
- Quadrants dynamic-range bound helpers inside the Genesis PCG loop add 47.500
  ms and 50,784 scalar launches, equal to 44.9 microseconds and 48 launches per
  iteration.
- Genesis rigid constraint-Hessian application adds 30.107 ms, or 28.5
  microseconds per iteration.
- Generic PCG vector/reduction work is not the leading gap: Genesis uses 24.6
  microseconds per iteration versus 21.6 microseconds in CGQ.
- BCOO SpMV is not the bottleneck: Genesis uses 4.93 microseconds per iteration
  versus 6.04 microseconds in CGQ for this scene.

The first mapped-forest rewrite is now complete. `RigidJointForestSystem`
publishes CGQ's `tree_roots`, `tree_body_start`, `tree_body_list`,
`tree_local_index`, `depth_start`, and `depth_order` layouts. Live tree counts
and extents remain device scalars. Three build-time paths are retained:

- `extras/rigid_forest/genesis_legacy=1`: original scan-all Genesis baseline;
- `genesis_legacy=0, rigid_forest/fused=0`: CGQ compact level path;
- `genesis_legacy=0, rigid_forest/fused=1`: CGQ bounded fused tree path when
  eligible, otherwise the compact level path.

Franka algebraic tests compare all three implementations for `P`, `P^T`, and
the virtual-work identity. The fused path is bounded by CGQ's eight-tree and
64-body-per-tree eligibility rules; it retains the legacy path only as an
explicit non-production A/B oracle.

The first 100-frame post-warmup ablation on RTX 5090 measured:

- Genesis legacy median 65.399 ms;
- CGQ compact level median 67.181 ms;
- CGQ fused tree median 53.783 ms;
- fused-tree speedup over the retained baseline: 1.216x (17.8% lower median).

The nine-frame named-node capture normalizes away the small iteration-count
difference. Legacy expand/project cost 134.0 microseconds and 26 launches per
PCG iteration; fused tree cost 35.0 microseconds and two launches. This is a
3.83x per-iteration reduction and a 13x launch-count reduction for `P/P^T`.

The original ablation used Genesis's built-in Panda MJCF while CGQ used
`franka_panda_mjcf_v2`. The strict robot-asset rerun now loads CGQ's exact
`panda.xml` and collision meshes in both implementations. Genesis enables
optional MJCF fixed-link merging so both loaders expose 10 rigid links and 9
DOFs; convexification, decimation, and watertight wrapping are disabled for
this benchmark. Over 100 measured frames after 20 warmup frames, the Genesis
fused-tree median is 34.972 ms versus the existing CGQ LinearPCG+diag median of
14.314 ms, reducing the same-robot-asset ratio to 2.443x. Table/riser system
representation remains different: fixed rigid proxies in Genesis versus fixed
ABD in CGQ.

The fused preconditioner apply now costs approximately 27.0 microseconds per
application versus CGQ's 25.1 microseconds. A missing actuator velocity
augmentation (`-act_bias[2] * dt`, CGQ's `dt * kv`) was added to edge and
free-root pivots; this reduced a representative ten-frame total-PCG median
from 166.5 to 98.

The earlier reduced operator used forest `P^T M P`, controller damping, and
the contact operator in place of Genesis native `nt_H`. A later frozen-state
Franka regression proved that this was not a valid substitution while the
gradient still came from Genesis's native constraint objective. The state had
one active native constraint row and no active QIPC contact. The forest/native
operator relative Frobenius error was 17.16%; each finger diagonal was
`0.225000` instead of native `1.317152`, omitting `1.092152` of active
constraint curvature.

Production now applies Genesis native `nt_H` for the complete reduced rigid
block and uses the forest only to pull the physical contact BCOO action back
to generalized coordinates. The mapped articulated factorization remains the
PCG preconditioner. The reduced variable is configuration displacement
`delta_q = h^2 delta_qacc`: native gradient is scaled by `h^2`, native Hessian
is unscaled, and the accepted solution is divided by `h^2` when written back
to `qacc`. This is the required coordinate congruence, not a missing timestep
factor.

On the 30-frame tabletop teleop regression, the mismatched operator reached
60 Newton iterations. It also needed 24 and 20 iterations on frames with zero
active QIPC contacts. After restoring native curvature, all four initial
zero-contact frames take exactly two Newton iterations; the complete run has
worst Newton 7 and 10 across two runs, line search 0, and worst per-solve PCG
52. The remaining 3--10-iteration frames coincide with active cloth/contact
work rather than contact-free rigid convergence.

The latest strict 100-frame comparison, with CGQ's Panda asset and
minimal-coordinate fixed table/riser, measured:

- Genesis median 43.009 ms, mean 40.562 ms, p95 50.519 ms;
- CGQ LinearPCG+diag median 14.658 ms, mean 14.427 ms, p95 15.487 ms;
- median ratio 2.934x;
- median total PCG 43 in Genesis versus 76 in CGQ.

The remaining gap is therefore execution cost, not excess Krylov work.
Remaining priorities are fixed pipeline graph fragmentation, generic
range-bound helper nodes, contact/BVH kernels, and the still-level-scheduled
forest factor stage. The production graph now enables Quadrants advanced
optimization; cold compilation is slower, but normalized steady-state work is
lower. BCOO SpMV remains explicitly deprioritized.

The final same-asset Nsight window attributes the remaining GPU gap:

- total kernel time: Genesis 258.346 ms, CGQ 128.599 ms (2.009x);
- kernel instances: 56,990 versus 38,628 (1.475x);
- unique kernels: 917 versus 350 (2.620x);
- actual PCG iterations: 578 versus 540;
- PCG cost: 168.7 versus 108.9 microseconds per iteration;
- Newton-pipeline cost: 3.679 versus 1.817 ms per iteration;
- Genesis Newton-pipeline graph: 810 kernel variants executed 40 times;
- CGQ Newton-pipeline graph: 191 variants executed 18 times.

Thus 70% of the measured excess GPU time is outside the PCG loop. All serial
tasks together contribute 37.502 ms, but this includes real work. Restricting
the count to a serial task immediately followed by its same-count range task,
with at most 5 microseconds average serial time, identifies 420 Quadrants
dynamic-range bound helpers: 24,726 instances and 24.887 ms. The retained
native rigid contact preprocessing contains a separate serialized task
(`_step_kernel...kernel_22_serial`) costing 9.754 ms over nine frames and is
not counted as a range helper. Within PCG, proxy map/post-processing costs 35.8
microseconds and bound helpers cost 13.7 microseconds per iteration. Forest
kernels and BCOO SpMV are no longer priorities: Genesis forest work is 67.3
microseconds versus CGQ's 75.7, and Genesis BCOO SpMV is 5.7 microseconds per
iteration.

Named task boundaries give an exact sort/reduce attribution:

- contact doublets: 12.435 ms, 96 kernel variants;
- contact triplets: 26.444 ms, 168 variants;
- global body BCOO: 22.565 ms, 171 variants;
- combined Genesis sort/reduce: 61.489 ms, 436 variants, 17,004 instances;
- share of Genesis GPU time: 22.02%;
- share of the Genesis-over-CGQ excess GPU time: 38.48%;
- Genesis cost per Newton: 1.577 ms;
- matching CGQ OneSweep/scan/FSR cost: 0.196 ms per Newton.

At Genesis's 39 Newton iterations in the labeled capture, replacing this phase
with the CGQ schedule projects a 53.84 ms reduction over nine frames, or about
5.98 ms per frame.

Quadrants' LSB radix sort uses an even pass count for the u32/u64 pipelines, so
the sorted keys and permutation land back in the first input buffers. Genesis
now consumes those first buffers for flags, FSR, and extraction; the `_out`
arrays remain temporary ping-pong storage. The previous code happened to work
for current row/vertex ranges because the final high-byte pass did not change
their order, but that was not a valid sort contract.

The complete excess-time decomposition is mutually exclusive:

- sort/reduce gap: 57.958 ms, 38.48% of the excess;
- PCG-loop gap: 52.755 ms, 35.03%;
- all remaining fixed/frame pipeline gap: 39.901 ms, 26.49%.

Within the PCG gap, Genesis executes 658 iterations versus CGQ's 540. The
extra work contributes about 20.0 ms; the remaining 32.7 ms is implementation
cost at equal iteration count. Genesis proxy map/post kernels cost 34.4
microseconds per PCG, range-bound helpers 14.1 microseconds, and vector
zero/copy kernels 14.8 microseconds. Forest work is not the cause: Genesis
costs 68.6 microseconds per PCG versus CGQ's 75.7.

The proxy map/post layer now fuses tangent expansion with optional slack
expansion, and fuses proxy wrench restriction with proxy result writeback.
Fixed-work profiling measures:

- proxy map/post: 34.4 to 14.5 microseconds per PCG (2.37x);
- PCG graph variants: 41 to 37;
- complete PCG implementation cost: 169.6 to 156.4 microseconds per iteration
  (1.084x).

Whole-trajectory wall time is not used as the proof for this layer because the
different floating-point reduction order changed the number of PCG iterations
in the sampled trajectory. Algebraic, virtual-work, free-root, fixed-root, and
Franka-Cloth integration gates pass; the normalized profile is the performance
acceptance metric.

The remaining fixed-pipeline multiplier is not caused by slower PT/EE kernels
per invocation. In the matched convergence trace:

- CGQ executes two Newton iterations per frame; Genesis executes three or
  four;
- both report CCD alpha 1.0, so the difference is not a shortened step;
- both final displacement directions are below the same 5e-4 absolute
  tolerance;
- Genesis proxy residuals are 1e-10--1e-8, below the 1e-5 proxy tolerance;
- CGQ proxy residuals are near machine precision.

Genesis PT, EE, and friction assembly kernels are comparable to or faster than
CGQ per invocation, but are repeated with every extra Newton iteration.
Therefore no scene-specific tolerance, iteration cap, or contact disable is an
admissible optimization. The next prerequisite is frozen-state numerical
parity of contact candidates, energy, gradient, Hessian, and accepted Newton
direction. Native rigid contact preprocessing may only be removed after QIPC
owns the exact CGQ rigid-rigid contact filters.

The pure-rigid conformance gate found the source of the extra Newton
iterations before contact algebra was changed. CGQ's MJCF loader interprets
the authored finger `biasprm="0 -100 -10"` as `kp=100, kv=10` even though
MuJoCo reports `biastype=NONE`; Genesis therefore retained zero finger bias
while both loaders agreed on `act_gain=100`. With the same explicit kp/kv
vector applied in the benchmark:

- first-frame pure-rigid joint-position maximum error is 1.85e-6;
- 100-frame maximum error is 6.76e-6;
- the coupled Genesis Newton work drops from three/four evaluations to two;
- the ten-frame median drops from approximately 48.1 ms to 30.2 ms.

This is parameter conformance, not a scene-tuned stopping rule. The global
Genesis MJCF parser retains MuJoCo semantics; the CGQ conformance benchmark
publishes the exact controller parameters explicitly.

The subsequent contact-free QCloth gate exposed two more CGQ-conformance
errors:

- Genesis used the inherited `QCloth.nu=0.3` in bending rigidity, but CGQ's
  Poisson-free Baraff-Witkin `Cloth` fixes `bending_nu=0`. Genesis now uses
  `bending_E * (2*thickness)^3 / 12`.
- Genesis detected convergence from the current Newton direction and then set
  `alpha=0`, discarding that final direction. CGQ applies the converged
  direction once and exits only after the second Newton evaluation. Genesis
  now has the same accepted-step and public counter semantics.

With matching quadratic bending, the first contact-free 25x25 cloth system has
identical row/column arrays, maximum transformed Hessian error
`6.94e-18` (relative Frobenius error `1.89e-16`), and maximum transformed RHS
error `1.02e-21`. Before the pipeline correction, the discarded direction
left `1.89e-8` of non-rigid displacement after frame one; frame-two PCG work
was 73 iterations versus CGQ's 21 and later frames reached 90--97. After the
correction, a 20-frame `tol_rate=1e-10` gate has:

- maximum transformed trajectory error `1.55e-15`;
- exactly two Newton evaluations per frame in both implementations;
- an identical total-PCG sequence, including 21 iterations on 17 of 20 frames.

This is a general Newton acceptance fix. It does not special-case free fall,
cloth size, gravity, or the milestone scene. On the matched Franka-Cloth
benchmark, both implementations now execute exactly two Newton evaluations
per measured frame. Mean total PCG work is 72.40 in Genesis versus 72.54 in
CGQ, so iteration count no longer explains the runtime gap. The synchronized
100-frame medians are 30.136 ms and 14.491 ms respectively
(`2.080x` Genesis/CGQ). A controller-aligned ten-frame Genesis trace improves
from 34.799 ms before the final-step correction to 29.465 ms after it
(`1.181x`).

`FEM.QCloth.nu` remains an inherited Genesis material-base field and is not a
CGQ `Cloth` parameter. The QIPC path deliberately ignores it for the
Baraff-Witkin preset; this is the sole API naming/storage exception in this
gate.

The next frozen-state gate enables contact without a rigid proxy, isolating
Consistent IPC from reduced-KKT routing. Two exact CGQ/Genesis scene pairs
were checked:

- A 25x25 cloth at `xi + 0.5*d_hat` above one halfplane produces exactly 625
  PH candidates and 625 active pairs in both implementations. Maximum
  per-pair gradient and Hessian errors are `5.42e-20` and `6.66e-16`;
  aggregate barrier-energy error is `6.44e-20`; CCD alpha is exactly 1.
- A free triangle at the same normalized gap above a fixed triangle produces
  exactly 3 PT and 6 EE candidates in both implementations. The same 3 PT
  pairs are active and all 6 EE pairs are inactive. Aggregate contact-gradient
  and Hessian maximum errors are `4.07e-20` and `3.33e-16`; barrier-energy
  error is `3.64e-23`.

For the fixed-triangle case, the complete first-Newton block system also
matches: row/column arrays are identical, block-value maximum error is
`3.33e-16` (relative Frobenius error `1.81e-16`), and RHS maximum error is
`4.07e-20`. Reconstructing CGQ's production PCG iterate from its dumped final
residual gives a Newton-direction maximum error of `3.60e-19` against
Genesis (`1.68e-15` relative). No contact-algebra change was required.

This gate deliberately represents the obstacle as fully fixed FEM surface
vertices. It proves PH/PT/EE candidates, active filtering, barrier energy,
gradient, Hessian, BCOO assembly, PCG, and accepted direction before adding
the rigid-proxy pullback. It does not claim reduced-KKT coupling parity; that
is the next gate.

The one implementation defect exposed by the minimal triangle was unrelated
to contact: a valid zero-live-hinge `QuadraticBending` allocation has capacity
one, but its initializer uploaded a `(0,4)` array. Zero-live-hinge buffers are
now initialized with zero-filled capacity storage while the authoritative
device scalar remains `n_hinges[()] == 0`.

The fixed-rigid-proxy gate then replaces the fixed FEM obstacle with a
watertight tetrahedron owned by CGQ/Genesis minimal-coordinate rigid systems.
CGQ allocates rigid global vertices before FEM while Genesis allocates FEM
before rigid proxies, so comparison uses the exact coordinate-derived vertex
bijection rather than assuming equal global IDs. Results are:

- both produce 9 PT and 6 EE candidates, with the same coordinate stencils;
- both activate the same 3 PT pairs and no EE pairs;
- coordinate-mapped aggregate physical gradient and Hessian maximum errors are
  `4.58e-18` and `2.59e-14` (`2.95e-14` and `2.49e-14` relative);
- barrier-energy error is `1.32e-21`, and CCD alpha is exactly 1;
- after one complete frame, cloth-position maximum error is `2.39e-18`;
- both report Newton 2, total PCG 2, maximum PCG 1, and zero line-search
  backtracks.

This closes the fixed rigid-proxy contact-routing gate. Together with the
previous gates, rigid dynamics, cloth elasticity, PH/PT/EE contact, physical
contact assembly, fixed-proxy pullback, Newton acceptance, and PCG iteration
counts are numerically aligned before profiling the remaining implementation
cost.

With that gate closed, the QIPC scene engine no longer runs Genesis native
rigid broadphase/narrowphase in parallel with QIPC contact. The retained
`build_rigid_engine()` comparison path still uses native collision. Unified
scenes default to QIPC-only collision and expose the explicit A/B fallback
`extras/rigid_contact/genesis_collision=1`; this key has no CGQ counterpart
and exists only to preserve the requested legacy comparison path.

The optimization followed the fixed profile/fix/profile sequence:

- Baseline `_step_kernel...kernel_22_serial`, between native
  `traverse_valid` and `clamp_prune_contacts`, cost `9.755 ms / 9 frames`
  (`1.084 ms/frame`).
- QIPC-only fixed-rigid frozen outputs are bitwise equal to the fallback for
  candidates, active pairs, contact gradient/Hessian, energy, CCD, final
  cloth state, and Newton/PCG counters.
- After the fix, the same task ID is only the zero-contact scalar path:
  `0.0073 ms / 9 frames`. The targeted native preprocessing cost is reduced
  by more than `99.9%`.
- The synchronized ten-frame wall median changes from `29.139 ms` to
  `29.088 ms`; total captured GPU kernel time changes from `220.518 ms` to
  `219.272 ms`. This small end-to-end delta is not used to deny the isolated
  removal: deleting duplicate native constraints changes the trajectory and
  raises the captured PCG work from 645 to 669 iterations.
- Normalized repeated-PCG cost improves from `158.44` to
  `155.61 microseconds/iteration`; non-PCG captured work falls from
  `118.325` to `115.168 ms`.

The post-removal nine-frame profile has a `1.705x` GPU-kernel-time ratio to
CGQ (`219.272` versus `128.599 ms`). The remaining compiler-owned costs are
already dominant named categories: contact/body sort-reduce costs
`27.102 ms` in Genesis versus `3.531 ms` in CGQ, and 586 dynamic-range helper
variants cost `20.962 ms` (sort `4.953`, PCG `8.918`, other pipeline `7.090`
ms). Remaining PCG work also differs (669 versus 540 iterations), so the
final optimization pass must keep numerical work count separate from
equal-work implementation cost.

The next PCG profile found one remaining application-side forest mismatch:
Genesis allocated CGQ's `edge_basis` and `edge_arm` buffers but recomputed
each scalar joint basis by traversing link joints and DOFs inside every PCG
expand/restrict. CGQ caches those values once per Newton. The compact
`cgq_level` and `cgq_tree` paths now consume the cache; the explicit
`genesis_legacy` path retains the recomputation oracle. Fixed child links
(`parent_edge == -1`) retain rigid transport without indexing an edge.

The matched late-window before/after profiles contain 656 and 658 PCG
iterations respectively:

- `forest_expand_tree_p`: `14.981 -> 12.661 microseconds/call` (`1.183x`);
- `forest_project_tree_Ap`: `18.110 -> 14.861 microseconds/call` (`1.219x`);
- combined expand/project: `33.091 -> 27.522 microseconds/PCG` (`1.202x`,
  16.8% lower).

The complete equal-work capture changes from `220.306` to `216.494 ms` GPU
time (`1.018x`) and from `29.304` to `28.868 ms` wall median (`1.015x`).
Cached level/tree `P`, `P^T`, virtual work, and the shared-memory
preconditioner match the retained legacy oracle; the full Franka-Cloth
reduced-KKT step also passes.

### Final equal-work profile and stop condition

The final late window contains 18 Newton evaluations in both implementations
and nearly equal PCG work (Genesis 658, CGQ 676). Results:

- GPU kernel time: `216.494` versus `145.500 ms` (`1.488x`);
- synchronized wall median: `28.868` versus `14.067 ms` (`2.052x`);
- kernel instances: `46,657` versus `42,300`;
- unique kernels: `1,296` versus `350`.

The remaining GPU excess is `70.994 ms`. Its dominant sources are now
Quadrants-owned lowering/algorithm gaps:

- contact/body sort-reduce: `30.421` versus `5.526 ms`, a `24.895 ms`
  gap; Genesis emits 438 variants versus CGQ's 35;
- PCG: `159.070` versus `106.683 microseconds/iteration`; the iteration body
  has 36 Genesis variants (14 helpers plus 22 real kernels) versus 26 native
  CGQ variants;
- dynamic-range bound helpers: sort `4.086 ms`, PCG `8.249 ms`, other
  pipeline `6.645 ms`;
- after subtracting PCG helpers, the equal-work generated-kernel gap is
  `39.85 microseconds/iteration`. Forest algebra, BCOO semantics, vector
  updates, and reductions already follow the corresponding CGQ schedule, so
  the remaining gap is code generation/kernel lowering rather than another
  scene-specific algorithm.

Sort/reduce, non-sort helpers, and equal-work PCG lowering account for about
93% of the remaining GPU excess without overlap. This satisfies the requested
stop condition: further Genesis-side native replacements would be workarounds
for Quadrants rather than framework optimizations.

The final 100-frame synchronized run reports Genesis median `29.165 ms`,
Newton mean 2.0, and total-PCG mean 72.29. The matching CGQ run reports
`14.491 ms`, 2.0, and 72.54, for a `2.013x` wall ratio with aligned numerical
work. Relative to the earlier `43.009 ms` Genesis run, the completed
conformance/optimization sequence is `1.475x` faster (32.2% lower median).
Single-run wall medians were not monotonic across every micro-optimization;
the per-layer acceptance figures above therefore use equal-work named-kernel
profiles.

### Post-OneSweep profile correction

The earlier `646 versus 676` statement compared Genesis frames 20--28 with a
reused late CGQ window. It was not a same-window PCG comparison and is
withdrawn. The current early-window CGQ capture contains 530 PCG iterations;
Genesis contains 646. Historical 75-frame means remain close (71.6 versus
72.4 iterations/frame), but the complete dynamic trajectories are not
framewise synchronized. The 116-iteration early-window difference is therefore
tracked as numerical-work debt, not charged to implementation speed.

The isolated frozen-state FEM and fixed-proxy gates above still prove PCG
algebra and iteration parity for those isolated operators. The complete
contact-free Franka-plus-free-cloth trajectory does **not** have framewise PCG
parity; the diagnosis below narrows that difference to the rigid contribution
to the shared Krylov recurrence. Per-iteration timing remains useful for
implementation profiling, but it is not evidence of framewise trajectory
parity.

### Quadrants resident occupancy grid

Quadrants commit `12119f03e` serializes a `grid_stride` task property, marks
CUDA `range_for` tasks at code generation, queries the compiled kernel with
`cuOccupancyMaxActiveBlocksPerMultiprocessor`, and clamps graph and streaming
launches to one resident block wave. An explicitly smaller grid remains
unchanged. Live extents remain device scalars; this changes no graph topology,
allocation capacity, or valid range.

The validation probe preserves all output values while changing a representative
dynamic launch from `8160x128` to `2040x128`. No `grid=8160` range launch
remains in the production profile. CUDA range/graph tests report 55 passes,
offline-cache tests report 27 passes, and the Genesis broad-phase/contact gate
reports 12 passes.

The matched Genesis before/after window is frames 20--28, with exactly 18
Newton evaluations and 646 PCG iterations in both captures. Contact-count and
accepted-step sequences are unchanged; state differences remain at
floating-point roundoff:

- total GPU kernel time: `170.420 -> 131.012 ms` (`1.301x`, 23.1% lower);
- synchronized wall median: `24.888 -> 20.230 ms` (`1.230x`, 18.7% lower);
- standard PCG: `149.006 -> 108.603 us/iteration` (`1.372x`);
- all paired dynamic-range work: `65.347 -> 36.108 ms` (`1.810x`);
- sort ranges: `12.196 -> 8.808 ms`;
- PCG ranges: `71.053 -> 43.061 ms`;
- BVH/contact ranges: `20.616 -> 17.442 ms`;
- other ranges: `14.350 -> 11.038 ms`.

The largest individual improvements include `drs_memset_lookback`
(`4.065 -> 0.963 ms`) and `forest_control_matvec`
(`3.247 -> 1.281 ms`). The serial bound helpers are deliberately separated:
they remain `8.064 -> 8.062 ms`, proving that occupancy removes excess worker
blocks but does not hide the scalar helper debt.

CGQ's current standard-PCG reference is approximately
`102.7 us/iteration`. The remaining equal-work PCG difference is now about
5.7%, and the unchanged Quadrants helper kernels are its dominant named
source. Eliminating or fusing those helpers belongs in Quadrants; a
Genesis-only fixed range, mask, host extent, or scene specialization is
forbidden. The framewise PCG mismatch is characterized below; strict parity
requires changing the rigid-model contract rather than tolerance changes.

### Framewise PCG workload diagnosis

The framewise difference is now resolved as a model-boundary difference, not
as an unlocated StandardPCG defect:

- A 40-frame free-cloth-only scene produces exactly the same `pcg` and
  `total_pcg` arrays in CGQ and Genesis: sums `920` and `1080`,
  respectively.
- With every cloth vertex fixed, the Franka/rigid-proxy subsystem needs at
  most one PCG iteration in both implementations. CGQ reports total PCG `80`;
  Genesis reports `76` because two zero-residual solves exit without an
  iteration.
- Combining the two uncoupled blocks with contact models disabled produces
  different shared-PCG histories: CGQ total PCG `1390` and Genesis `1101`
  over 40 frames. Both still execute exactly two Newton evaluations per frame,
  all candidate counts remain zero, and CCD alpha remains one.
- At frame zero the rigid joint-position maximum error is `1.85e-6` and the
  axis-mapped cloth-position maximum error is `9.44e-7`; at frame 39 those
  values are `1.35e-5` and `1.37e-4`. The movable-link masses and principal
  inertias agree to numerical precision, so this is not an asset or inertia
  mismatch.

The reason an uncoupled block can change the cloth iteration history is that
standard PCG uses one global `r^T M^-1 r`, `p^T A p`, alpha, and beta. It is
not two independent block solves. Genesis retains the native RigidSolver
prediction and gradient
`Ma - qf_smooth - qfrc_constraint`, while CGQ constructs its rigid BDF1 and
controller residual directly in the physical-body/forest pipeline. Those
rigid formulations remain close but are not the same trajectory. Their
different rigid Krylov component therefore changes the scalar recurrence
shared with an otherwise identical FEM block.

Two rejected explanations are recorded:

- Rewriting the Genesis forest unknown from native `qacc` units to CGQ
  displacement units is an exact congruence. Across 40 contact-free frames it
  leaves Newton, PCG, total-PCG, line-search, and CCD sequences unchanged;
  joint positions change by at most `8.53e-14`. The displacement form is kept
  because it matches CGQ semantics, not as a performance claim.
- The FEM operator, diagonal preconditioner, Newton acceptance, and StandardPCG
  recurrence are not the source: the isolated 40-frame FEM sequence is
  element-for-element identical.

Strict framewise CGQ trajectory parity would therefore require a selectable
CGQ rigid BDF1/state implementation (or direct use of CGQ's rigid runtime),
not tolerance tuning and not another PCG workaround. Replacing Genesis
RigidSolver is outside this migration's current authoritative-state contract.
Performance comparisons consequently report both actual trajectory work and
equal-work per-iteration cost.

The final same-window Nsight capture (frames 20--28, 18 Newton evaluations)
measures:

- Genesis: `146.637 ms`, `37,611` kernel instances, `759` variants, and
  `675` actual PCG iterations in that capture.
- CGQ: `110.796 ms`, `21,897` instances, `297` variants, and `520` PCG
  iterations.
- Genesis's 36 repeated PCG kernels cost `123.373 us/iteration`.
  CGQ's 19 iteration kernels plus seven preconditioner kernels cost
  `114.051 us/iteration`, so the equal-work PCG ratio is `1.082x`.
- Fourteen Quadrants scalar helper kernels inside each Genesis iteration cost
  `13.376 us/iteration`. They are the dominant remaining named PCG lowering
  cost; no fixed extent, mask, or scene-specific replacement is admissible.
- Normalizing Genesis to 520 PCG iterations gives `127.514 ms` of projected
  kernel work, or `1.151x` CGQ. Normalized kernel-instance count remains
  approximately `1.46x`, exposing generated graph-node fragmentation and
  launch scheduling beyond summed kernel-active time.

The synchronized 100-frame run, whose long-run mean PCG work is already close
(`87.27` Genesis versus `86.80` CGQ), reports warm-frame medians
`20.003 ms` and `12.927 ms` (`1.547x`). This wall ratio is not assigned to
extra Krylov work alone: the Nsight evidence above separates trajectory work
from the remaining Quadrants helper/node-lowering overhead.

### Cloth-only contact branch concurrency

A robot-free benchmark now isolates non-algorithm contact scheduling:
`examples/newton_coupling/multilayer_cloth_benchmark.py` contains four cloth
layers (8,804 vertices, 17,000 triangles) and one halfplane. Genesis and CGQ
use the same generated vertices/triangles, material values, timestep, contact
parameters, LinearPCG, and diagonal preconditioner. Both execute exactly two
Newton evaluations per frame with no line-search backtracking. No rigid
integration or reduced-KKT work is present.

The pre-change Nsight profile showed that Genesis serialized all PT, EE,
halfplane, BVH, filter, energy, and CCD branches: summed kernel time equalled
the kernel interval union, while CGQ overlapped approximately `2.797
ms/frame`. The blocker was not a Genesis algorithm. Quadrants rejected
`qd.graph.parallel_context()` inside a `checkpoints=True` graph, and its CUDA
graph builder could not keep dynamic-range bound helpers in a
checkpoint-owned fork/join.

Quadrants commit `3b0f9c2ad` adds checkpoint-owned parallel regions with
yield/resume coverage. The runtime admits the interleaved `checkpoint_id=-1`
pure range-bound helpers into the active checkpoint's section chain without
changing their unconditional resume semantics. The complete graph-parallel
suite passes (`33/33`). Genesis then reproduces CGQ's independent subgraphs
for:

- friction snapshot PT/EE/halfplane filters;
- active-pair counting;
- barrier and friction filter/assembly branches;
- barrier and friction energy branches;
- triangle and edge BVH builds;
- PT, EE, and halfplane trajectory queries; and
- PT, EE, and halfplane CCD.

The implementation is general scheduling over contact channels; it contains
no scene size, geometry, contact-count, or benchmark-specific condition. The
old ordering remains available only as the explicit A/B oracle
`extras/pipeline/genesis_serial=1`; production defaults to the CGQ-parallel
path.

Matched unprofiled frames 20--39 keep the contact counts identical and nearly
equalize LinearPCG work (`6,583` serial versus `6,570` parallel iterations):

- synchronized wall median: `26.711 -> 25.656 ms` (`1.041x`, 4.0% lower);
- synchronized wall mean: `25.359 -> 23.916 ms` (`1.060x`, 5.7% lower);
- relative to the pinned CGQ median `22.425 ms`, the Genesis ratio changes
  from `1.191x` to `1.144x`.

The long-window validation uses Quadrants `3b0f9c2ad`, CGQ
`42e7d4cbbad08739107ad830a17918f5f0f209ff`, 20 warmup frames, and 200
measured frames per implementation:

- CGQ median/mean/p95: `23.762 / 23.116 / 27.305 ms`;
- Genesis median/mean/p95: `27.071 / 26.651 / 30.465 ms`;
- Genesis/CGQ median/mean/p95 ratios: `1.139x / 1.153x / 1.116x`;
- both execute 400 Newton evaluations, zero line-search backtracks, and no CCD
  truncation;
- total PCG work is `78,733` in CGQ and `81,894` in Genesis (`1.040x`);
- total wall time divided by total PCG work gives a coarse `1.108x` ratio.
  This is not a kernel-only PCG metric because it includes fixed contact and
  graph work, but it removes the leading long-window Krylov-count difference;
- 38 frames whose PCG counts agree within 1% have a median per-frame wall
  ratio of `1.117x` and a mean ratio of `1.126x`.

The four consecutive 50-frame median ratios are `1.113x`, `1.121x`,
`1.143x`, and `1.196x`; their PCG-work ratios are respectively `1.008x`,
`1.023x`, `1.052x`, and `1.078x`. The increasing raw ratio therefore includes
trajectory-dependent work, not only implementation cost. After all 220
simulated frames, position differences remain small (`1.30e-4` maximum,
`1.85e-5` RMS), but broad-phase candidate populations have diverged. The
long-window conclusion is consequently:

- actual trajectory wall gap: approximately `1.14x--1.15x`;
- near-equal-work / coarsely normalized gap: approximately `1.11x--1.13x`;
- the remaining implementation gap is in the expected 10--15% range already
  attributed to Quadrants helper and graph-node lowering. It does not justify
  a Genesis scene specialization or fixed-size workaround.

The independent Nsight frames 20--28 prove that the serialized bottleneck is
gone:

- mean CUDA-graph kernel span: `22.609 -> 20.576 ms/frame` (`1.099x`, 9.0%
  lower);
- summed kernel work: `20.359 -> 20.697 ms/frame`;
- kernel interval union: `20.359 -> 18.161 ms/frame` (`1.121x`, 10.8%
  lower);
- measured overlap: `0.000 -> 2.536 ms/frame`, or 12.3% of summed parallel
  kernel work;
- active CUDA streams: `13 -> 25`.

The profile windows are not strict aggregate equal-work captures: atomic
assembly ordering perturbs the later PCG count while preserving the converged
state. A near-equal-work frame (`411` versus `412` PCG iterations) independently
shows kernel union `25.363 -> 22.352 ms` and graph span `28.173 -> 25.406 ms`.
The synchronized 20-frame result above is the primary end-to-end comparison
because its total PCG difference is only 0.2%.

The pinned CGQ profile has a `16.849 ms/frame` kernel union, so the remaining
Genesis GPU-union ratio is `1.078x`. Genesis still launches 19,290
`grid(1) x block(1)` scalar/helper kernels over the nine-frame capture
(`1.979 ms/frame`, 9.6% of summed kernel work), including dynamic-range
helpers and graph/checkpoint control. That compiler-owned bucket is already
larger than the `1.312 ms/frame` remaining union gap. Further removal requires
Quadrants range-helper fusion/lowering; a Genesis fixed extent, host size,
mask, or scene-specific replacement remains forbidden.

### Quadrants dynamic-range bound inlining

Quadrants commit `bf3a00645` removes the separate CUDA helper kernel and
global-temporary round trip for safe dynamic `range` bounds. The offloader
clones a side-effect-free bound-expression DAG into the consuming range task;
every worker evaluates the live device value before entering the grid-stride
loop. This preserves zero-dimensional device-scalar sizes and graph replay.
It introduces no host readback, mask, frozen capacity, or scene-dependent
specialization. Volatile expressions and reverse-mode tasks whose launcher
needs the exact extent for adstack sizing deliberately retain the old path.

This complements, rather than replaces, the resident-grid work in Quadrants
commit `12119f03e`. CUDA
`cuOccupancyMaxActiveBlocksPerMultiprocessor` and the live SM count determine
the resource-optimal resident grid; the worker's grid-stride loop determines
the logical dynamic extent. Reading a device scalar on the host merely to
choose the logical grid would recreate the synchronization and helper cost
that this change removes.

The regression changes both dynamic begin and end values between graph
replays and verifies that the graph contains exactly one worker task/node,
instead of a scalar helper plus worker. Dynamic-range/graph replay,
adstack-fallback, graph-parallel, and checkpoint-resume coverage reports 49
passes. The native extension rebuild and the complete formatting/lint suite
also pass.

The matched Nsight Systems windows cover frames 20--28 and 18 Newton
evaluations. Their PCG work differs (`2,606` before versus `2,495` after), so
aggregate time changes are reported separately from direct node evidence:

- the repeated standard-PCG body changes from 16 kernels per iteration to 11;
  the five removed `grid(1) x block(1)` variants are precisely the range-bound
  helpers, while all eleven numerical PCG kernels remain;
- those five helpers account for 13,030 launches and `10.340 ms` in the
  baseline window (`3.97 us/PCG iteration`);
- all graph kernel instances fall from `52,600` to `36,474`;
- `grid(1) x block(1)` instances fall from `19,290` to `4,385`, and their
  time falls from `1.979` to `0.492 ms/frame`; the remaining high-frequency
  scalar kernel is the required graph-do-while condition;
- mean graph span falls from `20.576` to `18.659 ms/frame` (`1.103x`), and
  kernel-interval union falls from `18.161` to `16.751 ms/frame` (`1.084x`).

The 200-frame post-warmup result provides the long-window wall measurement:

- Genesis before: median/mean/p95
  `27.071 / 26.651 / 30.463 ms`, 81,894 total PCG iterations;
- Genesis after: median/mean/p95
  `23.933 / 23.285 / 27.466 ms`, 76,239 total PCG iterations;
- raw before/after speedup: `1.131x / 1.145x / 1.109x`;
- pinned CGQ: median/mean/p95
  `23.762 / 23.116 / 27.303 ms`, 78,733 total PCG iterations;
- raw Genesis/CGQ ratios after the change:
  `1.007x / 1.007x / 1.006x`;
- total wall time divided by total PCG work gives a coarse Genesis/CGQ ratio
  of `1.040x`, down from `1.108x` before the change. This remains a
  whole-frame normalization, not a pure PCG-kernel metric.

Both Genesis runs execute exactly 400 Newton evaluations, zero line-search
backtracks, and full CCD steps. Final vertical position sum/min/max differ by
at most `5.03e-5 / 4.64e-8 / 1.88e-7`. The helper removal changes CUDA graph
scheduling and therefore floating-point reduction order, so it also changes
the later PCG trajectory. The direct acceptance proof is the disappearance of
the five helper launches per PCG iteration; the complete raw wall improvement
must not be attributed exclusively to their measured kernel-active time.

### 250-frame cloth cross-case validation

The range-bound inlining and occupancy-grid changes were revalidated on four
matched, rigid-free cloth workloads. Each backend received the same generated
vertices and triangles, material and contact parameters, constraints, `dt`,
LinearPCG solver, and diagonal preconditioner. Every run used 20 unmeasured
warmup frames followed by 250 measured frames:

- `pinned_drape` (6,561 vertices, 12,800 triangles, no contact) is the
  equal-work control. Genesis/CGQ had the same 648 total Newton iterations and
  112,804/112,791 total PCG iterations. Their median/mean/p95 frame times were
  `18.153/19.057/23.634 ms` and `18.094/19.745/25.146 ms`, respectively.
  The mean wall ratio and wall-per-PCG-work ratio were both `0.965x`; 249 of
  250 frames were within 1% PCG work and the final maximum position difference
  was `3.29e-6`.
- `inclined_drop` (6,561 vertices, 12,800 triangles, halfplane impact) recorded
  Genesis/CGQ median/mean/p95 times of `19.877/20.604/21.548 ms` and
  `20.255/20.828/22.672 ms`. Total Newton work was 513/508 and total PCG work
  was 101,981/100,879. The raw mean ratio was `0.989x`; normalizing the total
  wall time by total PCG work gave `0.979x`. The final vertical-coordinate
  difference was `4.62e-11`, while the translation-independent maximum shape
  difference was `2.11e-3`.
- `multilayer` (8,804 vertices, 17,000 triangles, four heterogeneous layers)
  recorded Genesis/CGQ median/mean/p95 times of `25.008/24.938/28.734 ms` and
  `23.295/23.095/27.398 ms`. Both executed 500 Newton iterations, but total PCG
  work was 103,919/96,749. The raw mean ratio was `1.080x`, while the
  wall-per-PCG-work ratio was `1.005x`. On the 17 frames whose PCG work was
  within 1%, the median ratio was `1.019x`. The observed raw gap therefore
  comes predominantly from trajectory-dependent solver work, not a comparable
  per-unit implementation overhead.
- `crossed_drop` (5,202 vertices, 10,000 triangles, two vertical sheets) is a
  trajectory-divergence stress case, not an equal-work timing case. Genesis/CGQ
  medians were `25.865/19.108 ms`, but means were `34.224/35.240 ms`; total
  Newton work was 971/1,081 and total PCG work was 157,472/168,193. Only 2 of
  250 frames were within 1% PCG work, and high-work phases occurred at
  different frames. Consequently, the raw `1.354x` median ratio is not an
  implementation comparison. The coarse wall-per-PCG-work ratio was `1.037x`.
  Both endpoints remained above the halfplane, but the final centered shapes
  differed by `0.993` after frictionless collapse.

The interpretable normalized ratios are therefore `0.965x`, `0.979x`, and
`1.005x`, with the trajectory-divergent stress case giving only a coarse
`1.037x` bound. This validates that the optimizations are general rather than
specific to the original multilayer stack; no remaining cross-case evidence
supports a large helper/grid implementation tax. Ratios within a few percent
must not be treated as a throughput claim without repeated runs and matched
kernel profiles.

Impact trajectories still need a separate frozen-state parity test. During the
measured window, minimum reported CCD alpha was `1.0` in Genesis versus
`0.219745` for CGQ in `inclined_drop`, and `1.0` versus `0.051993` in
`crossed_drop`. In addition, the two public line-search counters do not expose
the same semantics, so line-search counts were excluded from the comparison.
These observations prohibit using the impact cases as strict equal-work
evidence; they do not change the no-contact control result.

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

- Packed DOP14f has a verified 64-byte row stride and 64-byte-aligned base
  address, but Quadrants lowers a complete sixteen-lane row copy to sixteen
  scalar `ld.b32` instructions at offsets 0--60.
- Contact blocks continue to rely on generic ndarray lowering.

Target:

- Teach Quadrants alias/alignment analysis and CUDA lowering to preserve packed
  ndarray row alignment and emit four 16-byte vector loads where profitable.
- Re-profile DOP refit and traversal after vector lowering; do not introduce a
  Genesis-only native storage workaround.

### PERF-COMP03: Fastcache misses arithmetic static-property dependencies

Observed:

- `qd.static(range(self.n_levels_host - 1))` did not include the nested
  `n_levels_host` dependency in the fastcache specialization key.
- A graph first compiled for a singleton/free-body forest (`n_levels=1`) was
  reused for Franka (`n_levels=10`), so every mapped-forest factor/apply level
  was absent. Rigid preconditioned residuals and directions were exactly zero.
- Host `n_links` embedded in mechanism `%`/`//` indexing was likewise reused
  from an 11-link Franka graph in a 13-link Franka+table+cube scene. Table and
  cube proxies were decoded as Franka links 0/1. Runtime indexing now reads
  the zero-dimensional `RigidSystem.n_links[()]` scalar.
- The runtime fix does not retain a host live size. `max_depth` and `n_levels`
  are zero-dimensional device scalars. The graph emits one stage per link
  capacity and each stage reads `max_depth[()]` to select its live level.
  The one-frame Franka-Cloth gate then converged in 3 Newton iterations,
  9 PCG iterations, and 0 line-search backtracks.

Quadrants target:

- Used-property discovery must recurse through arithmetic expressions inside
  static `range` arguments.
- Add a fastcache regression that compiles `n_levels=1`, then `n_levels=10`,
  and verifies distinct graph task counts and execution.
- Keep graph topology based on allocation capacity and all live topology
  extents device-resident, independent of that compiler fix.

## Exit criteria

The register may be closed only when:

1. PERF-C01, PERF-C02, and PERF-C03 are complete before any next subsystem.
2. Every remaining item has an implementation, benchmark, or explicit
   user-approved deferral.
3. Runtime is measured against pinned CGQ on the same GPU, scene, precision,
   timestep, and contact parameters.
4. The final production path targets the best measured performance, not merely
   functional parity.
