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
- The initialization-only exact ET checker currently reuses one persistent
  64-entry global stack row per surface-edge query.

CGQ target:

- Warp-local DFS state for every production PT/EE traversal.

Rewrite:

- Move PT traversal state to the CGQ 8-warps/block shared frontier.
- Lower ET's fixed 64-entry stack to the same per-thread local representation
  as CGQ, after confirming Quadrants does not introduce worse spills.
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
  in-scene gap is generated-code/grid policy, not an application fallback.

Status: **closed at the Genesis application layer**. The remaining native-CGQ
gap is tracked as Quadrants dynamic-range/grid and lowering work; no
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

The reduced operator no longer mixes CGQ contact pullback with Genesis native
`nt_H`. It applies the matching forest `P^T M P`, controller damping, and
contact operator. A cached `body_inertia` field is an explicit Genesis storage
exception: CGQ stores the same physical 6x6 blocks in global BCOO, while
Genesis's compact generalized rows require retaining the unfactored body
blocks beside the mutated articulated factors. With this correction,
representative total PCG work reached 57.5 versus CGQ's 60.

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

### Post-OneSweep residual and stop condition

The standalone primitive gate and production integration described in
PERF-A06/PERF-L01 were applied after the preceding profile. The matched
nine-frame window now reports:

- 18 Newton evaluations in both Genesis and CGQ;
- 646 Genesis PCG iterations versus 676 in the pinned CGQ window;
- Genesis GPU time 190.349 ms versus CGQ 145.500 ms (`1.308x`);
- Genesis kernel instances 40,462, down from 46,657 before this layer;
- complete Genesis sort-reduce 12.534 ms versus 30.421 ms before the layer
  and 5.526 ms in CGQ.

The remaining Genesis GPU excess is 44.849 ms. The profile attributes:

- 422 confirmed Quadrants dynamic-range bound-helper variants, 16,127
  instances, and 16.056 ms;
- a 7.008-ms residual sort-reduce gap, dominated by per-pass dynamic range
  helpers, generated OneSweep code, and small-kernel launch floors;
- 149.46 microseconds per standard-PCG iteration across 36 Genesis variants,
  versus the pinned CGQ 106.68 microseconds across 26 variants.

The forest is not an application-owned regression: current expand, project,
control/inertia, and shared tree-preconditioner work remains comparable to the
pinned CGQ decomposition. BCOO SpMV is likewise not a leading cost. After
subtracting bound-helper time, Genesis GPU work is 174.293 ms (`1.198x` CGQ);
the rest is explained by the additional generated range kernels and their grid
policy/launch floor. This is the requested stop condition for Genesis-side
optimization: further native or scene-specialized replacements would hide
Quadrants compiler/runtime work instead of improving the simulation
architecture.

The final synchronized 100-frame Genesis median is 26.323 ms with Newton mean
2.0 and total-PCG mean 72.38. Relative to the pre-layer 29.165 ms it is
`1.108x` faster; relative to CGQ's 14.491 ms it is `1.816x`. The wall ratio is
larger than the GPU-kernel ratio because Python/graph submission and
synchronization are outside the captured kernel sum.

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
