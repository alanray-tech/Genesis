# P1 / P1b: IPC contact formulas in vector form, parity with CGQ

Library: `genesis/engine/ipc/` (and two generic helpers in `genesis/utils/geom.py`) on branch `ipc-formulas` (worktree
`genesis-world-ipc-formulas`). Harness: this folder, on branch `ipc-bvh-bench`. Ground truth: CGQ **`aeb58956`**
(locked, worktree `cuda-graph-qipc-ref-aeb58956`, `cases.GROUND_TRUTH`). Its inputs, outputs and caches are in
`data/aeb58956`, its tables in `results/aeb58956`. The tables of the previous ground truth `0496c4fa` (P1) stay as they
were in `results/`, and the P1 sections further down describe them.

## P1b result (ground truth aeb58956)

**PASS: 0 failing rows of 1,452,423 on CUDA and on CPU** (`results/aeb58956/table_acceptance.md`), fast_math off, same
rule as P1 (below). No row needed the referee: the pass1 and order clauses alone already leave 0 failing rows
(`table_acceptance-noreferee.md`). The order rows are exactly those of P1 (`ee_distance_type` 140, `closest_ee` 179,
`ccd_directional_pt` 1, `ccd_additive_ee` 19, `ccd_directional_ee` 100, screw PT 3 + 1, screw EE 20 + 7; CPU identical,
`results/aeb58956/clauses_acceptance.md`). Every one of the 92,827 `barrier_ee` rows passes by pass1:

| `barrier_ee` set | rows | max \|g - c\| / s |
|---|---|---|
| random / degenerate | 4,000 / 5,280 | 2.1e-13 / 3.9e-12 |
| real_cloth_stack / real_kimono (sampled active pairs) | 30,000 / 30,000 | 7.4e-15 / 2.0e-16 |
| real_cloth_stack_mollified / real_kimono_mollified (every active pair below the mollifier threshold) | 18,457 / 514 | 2.0e-14 / 2.5e-15 |
| mollifier_sweep (I1 / eps_x log-uniform over 1e-6..1) | 4,000 | 1.7e-13 |
| parallel (exactly parallel dyadic edges, and the same pairs rotated) | 576 | 4.7e-2, see below |

The 1,452,423 rows are P1's minus the dropped diagnostic case `barrier_ee_unprefixed` (69,280) plus the four new
mollified sets (23,547). Genesis with fast_math on (information): 1,761 failing rows on CUDA, 391 on CPU
(`table_genesis-fast-math.md`), the same kinds as in P1 plus, on CUDA, `barrier_ee` rows of the new sets (526
mollifier_sweep, 3 parallel) and 215 degenerate ones: contraction moves the mollified Hessian beyond CGQ's spread.

### CGQ contact formulas changed in 0496c4fa..aeb58956

`git diff 0496c4fa..aeb58956` (two squash merges, #432 and #433, which carry 1c630db1 and 79f0dcbb) over
`contact_function/*`, `consistent_ipc_contact.cu`, `halfplane_contact_kernels.cu`, `make_pd_lowrank_scalar.h`, friction,
CCD and candidate reach. Changes touching a function of the inventory:

1. **Mollified EE barrier** (`consistent_ipc_contact.cu`, `contact_export_system.cu`, new
   `contact_function/ee_mollified_barrier.h`): the gradient and Hessian are now the full product rule of m(I1) b(D),
   `G = b gm + m b' gD`, `H = b Hm + b' (gm gD^T + gD gm^T) + m (b'' gD gD^T + b' HD)`, in the natural vertex order for
   every flag (the PE/PP permutations are gone), with `HD = ee_flagged_distance2_hessian` and the mollifier derivatives
   of `ee_mollifier` (I1's Hessian included). The Hessian has no gassThreshold clamp. It is projected by
   `make_pd_12x12_lowrank_scalar` (range of A from Gram-Schmidt over its columns, tolerance 1e-12 of the Frobenius norm,
   rank budget 6, cyclic Jacobi on V^T A V), falling back to `gipc_make_pd<double, 12>` (Eigen self-adjoint solver,
   negative eigenvalues clamped, the matrix kept when already PSD) above rank 6, then symmetrized (79f0dcbb).
   Ported, see below.
2. **Mollified distance choice** (`ee_mollified_distance2`, shared by energy and derivative kernels): interior pairs
   (four active vertices) take `gipc_d_EE` while it lies in the barrier band, the flagged distance otherwise. Behavior
   unchanged from 0496c4fa, only factored. Both are the same plane distance in exact arithmetic, and Genesis has a
   single expression for it (`func_edge_edge_distance2`), so nothing to port.
3. **Rank-2 mollified family removed** (`gipc_barrier_gradient_hessian_rank2.h` deleted). The un-prefixed 2x2 variants
   of `gipc_barrier_gradient_hessian.h` remain in CGQ but production no longer calls them.

Outside the inventory: `adhesive_ipc_contact.cu` (free-DOF block projection), `variational_adhesion.h` (its mollifier
moved to `ee_mollifier`), `spatial_query.cu` (watertight ray/triangle and conservative ray/box tests of the spatial
query API). `halfplane_contact_kernels.cu`, `make_pd_lowrank_scalar.h`, friction, CCD and candidate reach are unchanged.

### The port

- `barrier.py`: `func_mollified_barrier_gradient_hessian(flag, ea0, ea1, eb0, eb1, D, grad D, threshold, d_hat, xi,
  kappa, rank_tolerance, eigen_tolerance, max_sweeps)` is the product rule above, then `gu.qd_project_psd`, then
  `0.5 (H + H^T)`. The rank-2 F-space form and its closed-form 2x2 projection (the P1 fix) are removed.
- `distance.py`: `func_edge_edge_flagged_distance2_hessian(flag, ...)` dispatches on the flag like the flagged distance
  (shared slot helper) to `func_edge_edge_distance2_hessian` (plane distance, Hessian over (r, u, v) as
  `(2 / q) w w^T + 2 a Hess s - a^2 Hess q`, bounded as the lines meet), `func_point_edge_distance2_hessian` and the
  constant PP block. `func_edge_edge_cross_norm2` now also returns I1's Hessian. All are written over the difference
  vectors and mapped to the vertices by one static helper.
- `genesis/utils/geom.py`, generic over the size: `qd_symmetric_eigen(A, tolerance, max_sweeps)` (cyclic Jacobi, CGQ's
  rotation and convergence test, runtime loops with runtime indices into local matrices) and `qd_project_psd(A,
  max_rank, rank_tolerance, tolerance, max_sweeps)` (CGQ's low-rank projection, its range basis and small eigenproblem
  zero-padded to the full size so that one Jacobi instance serves both paths; above the rank budget the whole matrix is
  decomposed, and returned untouched if PSD). The fallback uses Jacobi where CGQ uses Eigen's tridiagonal QR: the PSD
  projection is unique, so only rounding differs. Constants: 1e-12, 1e-15, 12 sweeps, rank 6 (`cases.py`).
- Harness: `case_barrier_ee` composes aeb58956 production (`ee_mollified_distance2` then `ee_mollified_barrier_grad_hess`),
  `barrier_ee_unprefixed` is dropped, the referee evaluates the product rule and a 120-digit eigendecomposition with
  every derivative a central difference of the plain distance, cross norm and barrier (independent of both sides'
  analytic Hessians).

### Defect found and fixed by the harness (P1b)

- `qd_symmetric_eigen` first computed its convergence measure as the total squared norm minus the squared diagonal,
  which cancels to rounding noise, so the sweeps stopped with off-diagonal entries near 1e-10 of the norm. Every
  mollified Hessian of rank above 6 (the full-decomposition path) was then 1e-10 to 7e-9 of the scale off (1,429 rows
  of cloth_stack and the sweep failed pass1), where CGQ is exact to 1e-16. Now the off-diagonal squares are summed
  directly, as CGQ does; the projection matches the 120-digit value to 4e-15 on those rows.

### Accuracy of CGQ aeb58956 (for chat KGxtQv051LpR)

No row of the acceptance needed the referee, so there is no row where Genesis is closer than CGQ beyond the pass1
bound (`results/aeb58956/cgq_accuracy.md` is empty). One place where CGQ (both builds) and Genesis are all inaccurate:
the **gradient** of the mollified barrier on the rotated parallel rows of `barrier_ee/parallel` (I1 ~ 1e-32, parallel to
the rounding of the rotation). There the gradient is dominated by b m' grad I1, and grad I1 = 2 (v x c, c x u) with
c = u x v pure cancellation, so every float64 evaluation is 1e-2 to 8e-2 of the set's gradient scale (6.5e-6, against a
Hessian scale of 1.4e11) away from the exact value: rows 466 (CGQ fma 7.3e-2, CGQ ieee 8.0e-2, Genesis 8.0e-2), 393
(4.1e-2 / 7.9e-2 / 7.9e-2), 420 (4.1e-3 / 2.0e-2 / 2.0e-2), 548, 382. That is conditioning of the formula at I1 -> 0,
with forces of order 1e-6, not a porting difference: Genesis equals CGQ's IEEE build there, and the rows pass pass1
because CGQ's two builds differ by as much. The Hessians of these rows agree with the exact value to rounding.

### Cost of the mollified path (Genesis, CUDA, float64, fast_math off)

`bench_mollified.py`, under an exclusive GPU lease, fresh offline cache. Each kernel evaluates the flag, the flagged
distance and the barrier gradient and Hessian of every pair and writes them (12 + 144 doubles): the plain path on the
26,871 non-mollified active EE pairs of cloth_stack, the mollified path on its 18,457 mollified ones, each tiled to
500,000 pairs, median of 20 calls.

| path | first call (compile + run) | per pair |
|---|---|---|
| plain EE (rank-1 Hessian) | 0.78 s | 4.2 ns |
| mollified EE (product rule, PSD projection) | 9.58 s | 208.7 ns |

The mollified path costs about 50 times the plain one per pair and adds about 9 s of compile time to a kernel that
contains it, almost all of it the projection (Gram-Schmidt, Jacobi and the reconstruction on 12x12 local matrices). The
plain kernel is bound by the 1.25 KB it writes per pair. Mollified pairs are rare in practice (I1 below 1e-3 of the rest
lengths' product), so the runtime cost scales with their count. The compile time is paid by every kernel that inlines the
mollified path, which argues for keeping it in one kernel.

## Edge-triangle intersection fix (both sides)

The coplanar branch of `tri_edge_intersect` / `func_edge_triangle_intersect` tested an edge against each triangle side
by 3D segment-segment closest points, skipping a side only when `|e x s|^2 < 1e-30`. A flat cloth (box_on_cloth, CPU,
frame 8, edge [145, 161], triangle [143, 144, 160], reproduced on Genesis 7224d8c4) reported a false crossing:
- in the plane, the edge and the side 144-160 are parallel to 5e-17, 2.65 cm apart. `|e x s|^2 = 3.11e-30` measures the
  tilt of the edge out of the triangle plane (heights -2.6e-14 / 7.2e-15, sin 6.3e-13), which the coplanar band admits;
- the crossing coordinates then hang on the normal component of e x s, 1.3e-19 exactly, from products of size 1.4e-3.
  Float64 gives 0, hence u = 0.892 where the exact value on the same float inputs is 5.8e7 (rational arithmetic).

A threshold relative to |e|^2 |s|^2 does not close it: the band admits tilts up to about 2e-12 L / |e|, and the same
edge moved to heights of 0.5 to 0.95 of the band (tilt 1.0e-12 to 1.9e-12) still crosses with tol = 1e-12. Fix (lead's
decision), the same on both sides: the side test in the plane of the triangle, with its normal n,
`denom = (e x s) . n`, side skipped when `denom^2 < rel^2 |e|^2 |s|^2 |n|^2` (rel = 1e-12, the header's tolerance),
`u = ((B - A) x s) . n / denom`, `v = ((B - A) x e) . n / denom`. The tilt no longer enters. Point-in-triangle had the
same scale problem (`|uv^2 - uu vv| < 1e-30` rejects every triangle with sides under ~3e-8 m) and now skips when
`|denom| < rel^2 uu vv`. A collinear overlap stays detected by the endpoints or by the two other sides.

Proof, ET only (nothing else rerun): input set `intersect/regression` (`inputs.py`, 8 rows: the real case, its tilted
copy, an edge exactly parallel to a side in the plane and the same edge tilted inside the band (+-1e-12), two collinear
overlaps, a 1e-8 m triangle crossed and missed). Crossing reported per row, in that order:

| | real | real tilted | parallel | parallel tilted | overlap 1 | overlap 2 | small crossed | small missed |
|---|---|---|---|---|---|---|---|---|
| CGQ aeb58956, nvcc --fmad=true | 0 | 0 | 0 | **1** | 1 | 1 | **0** | 0 |
| CGQ aeb58956, nvcc --fmad=false | **1** | **1** | 0 | **1** | 1 | 1 | **0** | 0 |
| CGQ fixed (both builds), Genesis fixed (CUDA, CPU) | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 |

Whether the real case crosses depends on contraction (with fused multiply-adds no tilted copy of it crosses, without
them 44% of 200,000 do), while the tilted exact-parallel case crosses in any arithmetic: the old test returns the closest
points of the two lines (u = v = 0.5) and never checks that the lines meet. On all 4,808 intersection rows, the fixed
Genesis (CUDA and CPU, fast_math off) and both fixed CGQ builds agree on every row, and the fixed CGQ differs from
aeb58956 on none of the 4,800 rows of the other sets (`results/aeb58956/table_et-fix.md`). CGQ test
`tests/test_edge_triangle_intersect.py` (branch `fix/et-relative-threshold`): with the header of aeb58956, 2 of 3 fail
(the tilted exact-parallel case, the small triangle); with the fix, 3 of 3 pass.

## P1 result (ground truth 0496c4fa, tables in results/)

**PASS: 0 failing rows of 1,498,156 on CUDA and on CPU** (`results/table_acceptance.md`), Genesis compiled with
`fast_math` off, under the lead's rule:

- reference c = CGQ as production builds it (nvcc default, `--fmad=true`), c_ieee = CGQ built with `--fmad=false`;
- pass1(x): every real output within `max(1e-12 * s, 2 * |c_ieee - c|)` of c (s = largest magnitude of that output
  over the input set, |.| the max over its components), every discrete output equal to c or c_ieee;
- a row passes by **pass1(g)**, else by **order** = pass1(g_order), g_order the same Genesis code with only the
  summation order of 3-term dot products switched to Eigen's, else by the **referee**: 50-digit exact value e,
  `|g - e| <= max(|c - e|, 1e-12 * s)`.

One command: `powershell -File bench\ipc_formulas\run.ps1` (inputs, both CGQ builds, the six Genesis runs, referee,
tables). Per-function clause counts with example inputs: `results/clauses_acceptance.md` (summary below, CUDA; CPU is
identical).

| Function (case) | rows | pass1 | order (max err/s) | referee (max err/s) |
|---|---|---|---|---|
| every distance, barrier PT/PE/PP/PH, pair, friction, frame, PE/PP CCD, PH CCD, candidate, intersection case | 996,460 total | all | 0 | 0 |
| `ee_distance_type` | 67,632 | 67,492 | 140 (flags) | 0 |
| `closest_ee` | 67,632 | 67,453 | 179 (7.8e-2) | 0 |
| `ccd_directional_pt` | 68,096 | 68,095 | 1 (1.8e-12) | 0 |
| `ccd_additive_ee` | 68,560 | 68,541 | 19 (4.5e-11) | 0 |
| `ccd_directional_ee` | 68,560 | 68,460 | 100 (9.9e-8) | 0 |
| `ccd_screw_pt` / `_interval_pt` | 6,048 / 6,048 | 6,045 / 6,047 | 3 / 1 (3.6e-1) | 0 |
| `ccd_screw_ee` / `_interval_ee` | 5,280 / 5,280 | 5,260 / 5,273 | 20 / 7 (3.0e-1) | 0 |
| `barrier_ee` (and the diagnostic `barrier_ee_unprefixed`) | 69,280 | 69,214 | 0 | 66 (2.6e-10) |

Every order row is a near-parallel edge pair or a knife-edge CCD stop decision (examples in the clauses file); every
referee row is a mollified edge-edge Hessian of the real frames (sin^2 of the edge angle between 4e-4 and 1e-3), where
Genesis is closer to the exact value than CGQ: e.g. kimono row 9244, CGQ 4e-8 of the row's largest entry, Genesis 3e-15.

## Why some rows need the order clause or the referee

1. **CGQ's mixed summation order** (all order rows). CGQ sums 3-term dot products in two orders: its distance
   classifiers' `nofma_dot` as (x0 + x1) + x2, which is Quadrants' order, and everything using Eigen's `.dot()` /
   `.squaredNorm()` as x0 + (x1 + x2). On near-parallel edges (a*c - b*b cancels completely) and at knife-edge CCD
   stops this decides the result. Experiment: rerunning Genesis with only Quadrants' reduction replaced by Eigen's
   (`genesis_side.py --sum-order eigen`) clears every CCD, `closest_ee` and `ee_distance_type` difference and creates
   new ones in exactly the functions built on `nofma_dot` (EE distance flags, barrier_ee degenerate). Compiler
   contraction never changes this order, which is why CGQ's two builds agree on these rows. The library keeps
   Quadrants' order (lead's decision).
2. **CGQ's own float64 error on mollified edge-edge Hessians** (all referee rows). Near the mollifier threshold the 2x2
   block of that Hessian has diagonal entries 10 to 20 orders of magnitude apart, and CGQ's results are 1e-12 to 1e-10
   of the output scale away from the exact value in both builds alike. Genesis is closer to it on every such row.

The 66 rows where Genesis is closer to the exact value than CGQ are recorded with their inputs in
`results/cgq_accuracy.md` (`cgq_accuracy.py`), to improve CGQ later: CGQ is 1e-11 to 2e-8 of the row's largest entry
away from the exact value there, in both builds, Genesis 4e-16 to 1e-13. CGQ's own un-prefixed algebra fails on the
same rows, so the loss is in the shared steps, presumably the 2x2 eigensolve (Eigen's `computeDirect`) of a block whose
diagonal entries are 10 to 20 orders of magnitude apart; the fix of Genesis (eigenvalue of the sign opposite to the
half trace from the determinant, eigenvector from the cancellation-free column) carries over.

## Defects found and fixed by the harness

- Projected 2x2 block of the mollified EE Hessian, eigenvector: my first closed form `(M - lambda_min I) * lambda_max /
  (lambda_max - lambda_min)` left the small diagonal entry with an absolute error of the rounding of lambda_max. Genesis
  was 5e-11 of s off where CGQ is exact to 1e-16 (50-digit check). Now the eigenvector is the cancellation-free column.
- Same block, eigenvalue: `lambda_max = half_trace + radius` cancelled when the block has a huge negative entry
  (-6.8e14 against 2e4), losing ~10 digits (kimono row 9244: 3e-6 of the row). Now the eigenvalue of the sign opposite
  to the half trace comes from the determinant; Genesis is 3e-15 of the row there.
- Additive CCD: the mean displacement was summed left to right where CGQ pairs it; the CCD accumulates that rounding
  over its steps (1e-11 where CGQ's builds agree to 3e-13). Now the pairs follow a fixed order per stencil.

## Fused multiply-add (check 4)

- Quadrants offers no per-operation or per-function control of contraction. `fast_math` is one global compile flag:
  it sets `AllowFPOpFusion = Fast` on the target machine (CUDA and CPU) and `reassoc`, `ninf`, `nsz`, `afn` on every
  instruction, so even an error-free emulation of an exact product is folded. Genesis' `gs.init` sets
  `fast_math = not debug`.
- With `fast_math` on (Genesis' default), on CUDA, real data flips distance flags: cloth_stack 75 PT flags
  (`pt_distance`), 112 in `barrier_pt`, 4 EE flags, 12 in `barrier_ee` (`results/table_genesis-fast-math.md`, 1,614
  failing rows under pass1; CPU 523 rows, no real-data flag flip). With it off: 0.
- Setting used for the parity: `fast_math` off, switched after `gs.init` in `genesis_side.py`, each setting with its own
  offline cache. How Genesis handles it for IPC scenes is left to the next phase.

## Information tables

- `table_ieee-strict.md`: Genesis against CGQ's IEEE build at 1e-12 alone (1,183 rows: the two causes above).
- `table_cgq-contraction.md`: CGQ's production build against its own IEEE build at 1e-12 alone: 3,085 failing rows,
  more than Genesis has against either.
- `table_acceptance-noreferee.md`: pass1 and order clauses only (132 rows left, all mollified EE, then refereed).

## Inputs

Random (unit box, band parameters over the whole active band), degenerate (exact dyadic coordinates on region
boundaries, coplanar, parallel, collinear, coincident, zero-area; r below the Hessian clamp, r near 1, D near xi^2;
near-parallel edges in the mollifier region; the three friction regimes, rest and rigid translation; CCD approaching,
grazing, inside the thickness, at rest; screw motions with rotation, co-moving sides, 1e-16 noise rotations), and real
stencils of T0's cloth_stack and kimono frames (candidates for distances, CCD and candidate reach, active pairs for
barrier and friction, PE/PP sub-features from their flags). Not covered by real data: screw CCD (the frames carry no
rotation data), half-plane cases (plane positions not captured), intersection; those are synthetic only. Friction on
real data uses the frame positions as lagged positions and the trajectory end as current ones, mu = 0.5, eps_vh = 1e-4;
mollifier rest positions are frame 0 (x_bar was not captured).

## Unsure / deviations

- Mollified EE (P1b): the full-decomposition fallback of the projection is Jacobi in Genesis and Eigen's tridiagonal QR
  in CGQ; the projection is unique, and the parity shows only rounding differences.
- Additive CCD reuses `func_point_triangle_distance_flag` where CGQ uses `gipc_dType_PT` (same Cramer system, Eigen
  determinants, contractible arithmetic); no row needed more than pass1 there.
- PP / PH tangent bases come from `genesis.utils.geom.qd_orthogonals`, a different orthonormal basis than CGQ's; the
  friction terms are invariant to it, and the harness compares the projector T T^T.
- Thresholds are parameters (Hessian clamp ratio 1e-6, mollifier ratio 1e-3, EE parallel tolerances 1e-6 and 1e-14,
  intersection tolerance 1e-12, CCD eta and iteration budget), to come from info fields in the solver; underflow guards
  (1e-300, 1e-30) stay literals.
- 12x12 matrices: the harness kernels compile and run in about 40 s per backend for all cases. The mollified path
  dominates compile time (see its cost above); the solver integration should measure its own kernels.
- The half-plane cull `func_point_halfplane_candidate_reach` was added after the brief (production reaches the cull in
  `halfplane_query_kernel`).

## Module layout

| Module | Content |
|---|---|
| `distance.py` | distance flags (PT, EE, PE, the parallel-projecting EE type of the additive CCD), squared distances of the primitives and their gradients, flagged distance and gradient per stencil, Hessians of the EE, PE and PP squared distances and the flagged EE distance Hessian, edge cross norm with its gradient and Hessian, closest points (PT, EE, PE, generic over the stencil shape) |
| `barrier.py` | Consistent IPC barrier and its derivatives, normal force, mollifier threshold and mollified barrier, PSD gradient/Hessian (plain rank-1, mollified product rule), half-plane gradient/Hessian, pair parameters (d_hat, xi, area weight) |
| `genesis/utils/geom.py` | generic `qd_symmetric_eigen` (cyclic Jacobi) and `qd_project_psd` (low-rank PSD projection with full fallback) |
| `friction.py` | tangent frame and closest-point weights per stencil (PT, EE, PE, PP, PH), friction energy and PSD gradient/Hessian shared by all stencils |
| `ccd.py` | additive, directional and screw (interval form) conservative advancement, each generic over the stencil shape; half-plane CCD, linear and screw; broad-phase distance rule (simplex and half-plane) |
| `intersection.py` | edge-triangle intersection |

The split follows the brief. Two choices shape it:
- Every barrier gradient/Hessian is one function of the squared distance D and its gradient over the stencil,
  `func_barrier_gradient_hessian(D, grad D, ...)`: CGQ's col 8 of PFPx is grad sqrt(D) / sqrt(dH), and its rank-1
  Hessian equals `lambda0 / (4 (D - xi^2) dH) * grad D grad D^T`. So the per-type CGQ functions (PT, PE, PP, the EE
  rank-1 fast path) become flag classification + `func_*_flagged_distance2` (D and its gradient, scattered over the
  4-vertex stencil) + one barrier function. The mollified EE barrier (P1b) takes the flag, the edges, D and its
  gradient, and builds the distance and cross-norm Hessians itself for the product rule.
- Friction is one energy and one gradient/Hessian function of a tangent basis T, vertex weights w and the vertex
  displacements, CGQ's Jacobian being `J = w (x) T^T`; each stencil only contributes its frame.
- CCD routines take each side of the stencil as the rows of a matrix, and the shapes (1x3, 2x2, 1x2, 1x1) select the
  closest-point routine statically: one function per CCD flavour instead of four.

## Inventory: what CGQ's production Consistent IPC path reaches, and where it lives in Genesis

Traced from `consistent_ipc_contact.cu` (filter/assemble/energy/friction kernels), `contact_kernels.cu` (count and
friction pair filters), `halfplane_contact_kernels.cu` (query, CCD), `ccd_kernels.h` (per-pair TOC dispatch),
`candidate_reach.h` and `edge_triangle_intersect.h`, under the default config (`consistent_ipc`, `ccd_bound =
directional`, `directional_reach = 1`, `ccd_eta = 0.5`, `friction/eps_v = 1e-2`). Harness case in the last column.

### Distances (`distance.py`)

| CGQ | Genesis | Case |
|---|---|---|
| `pt_distance_flag` (nofma Cramer) | `func_point_triangle_distance_flag` | `pt_distance` |
| `ee_distance_flag` (near-parallel guard 1e-6) | `func_edge_edge_distance_flag(..., parallel_tolerance)` | `ee_distance` |
| `detail::point_edge_distance_flag_3` | `func_point_edge_distance_flag` | `pe_distance` |
| `ee_distance_type` + `ee_parallel_distance_type` (additive CCD, 1e-14) | `func_edge_edge_distance_type` (same flag encoding as the EE flag) | `ee_distance_type` |
| `gipc_dType_PT` (additive CCD) | `func_point_triangle_distance_flag` (same Cramer system; CGQ evaluates it with Eigen determinants) | through `ccd_additive_pt` |
| `pp/pe/pt/ee_distance2`, `gipc_d_PP/PE/PT/EE` (identical expressions) | `func_point_edge_distance2`, `func_point_triangle_distance2`, `func_edge_edge_distance2`; PP is `(a - b).norm_sqr()` | all distance and barrier cases |
| `pt/ee_flagged_distance2`, `flag_active_offsets(_for_ee)` | `func_point_triangle/edge_edge/point_edge_flagged_distance2` (D and its gradient over the stencil) | `*_distance` |
| distance gradients: PFPx col 8 (production), `*_flagged_distance2_gradient` | `func_*_distance2_gradient` | `*_distance`, `barrier_*` |
| `(p1-p0).cross(p3-p2).squaredNorm()`, `ee_mollifier` (I1, its gradient and Hessian) | `func_edge_edge_cross_norm2` (I1, its gradient and Hessian) | `mollifier`, `barrier_ee` |
| `ee_flagged_distance2_hessian` (`pp/pe/ee_distance2_hessian`, SymPy code in `details/*.inl`) | `func_edge_edge_flagged_distance2_hessian`, `func_edge_edge_distance2_hessian`, `func_point_edge_distance2_hessian` (closed forms) | `barrier_ee` |
| `detail::closest_pt/ee/pe` | `func_point_triangle_closest_point`, `func_edge_edge_closest_points`, `func_point_edge_closest_point`, `func_closest_points` | `closest_*` |

### Barrier (`barrier.py`)

| CGQ | Genesis | Case |
|---|---|---|
| `gipc_barrier_energy`, `_first_derivative`, `_second_derivative` | `func_barrier` | `barrier` |
| `gipc_normal_force` | `func_barrier_normal_force` | `barrier`, friction |
| `gipc_compute_eps_x` | `func_edge_edge_mollifier_threshold(..., mollifier_ratio)` | `mollifier` |
| `gipc_barrier_energy_mollified` | `func_mollified_barrier` | `mollified_barrier` |
| `gipc_barrier_grad_hess_pt/pe/pp`, `_ee_rank1`, `gipc_pfpx_*`, `_barrier_lambda0` (gassThreshold clamp) | `func_barrier_gradient_hessian(D, grad D, ..., hessian_clamp_ratio)` | `barrier_pt/ee/pe/pp` |
| `ee_mollified_distance2`, `ee_mollified_barrier_grad_hess` (aeb58956) | `func_edge_edge_flagged_distance2` (one plane-distance expression), `func_mollified_barrier_gradient_hessian` | `barrier_ee` |
| `make_pd_12x12_lowrank_scalar`, `gipc_make_pd<double, 12>`, the symmetrization after them | `gu.qd_project_psd` (with `gu.qd_symmetric_eigen`), `0.5 (H + H^T)` | `barrier_ee` |
| `halfplane_signed_distance`, `_distance_sq` | `normal.dot(x - plane_point)` at the caller | `barrier_ph` |
| `halfplane_barrier_gradient`, `halfplane_barrier_hessian` | `func_point_halfplane_barrier_gradient_hessian` | `barrier_ph` |
| `pair_d_hat_*`, `pair_thickness_*`, `cipc_pair_area_weight` | `func_pair_barrier_parameters`; PH takes the vertex values and `w * d_hat` | `pair_*` |

### Friction (`friction.py`)

| CGQ | Genesis | Case |
|---|---|---|
| `f0`, `f1_div_rel_dx_norm`, `f2_term`, `friction_energy/gradient/hessian` (three regimes) | inside `func_friction_energy`, `func_friction_gradient_hessian` | `friction_*` |
| `*_friction_tangent_basis`, `*_friction_closest_point`, `*_tan_rel_dx`, `*_jacobi` | `func_point_triangle/edge_edge/point_edge/point_point/point_halfplane_friction_frame` (basis and vertex weights) | `frame_*` |
| `gipc_friction_energy/grad_hess_pt/ee/pe/pp`, `halfplane_friction_energy/grad_hess` | frame + `func_barrier_normal_force` + energy / gradient-Hessian | `friction_*` |
| friction pair filter: EE dropped when `cross^2 < 1e-3 |ea|^2 |eb|^2` | `func_edge_edge_cross_norm2 < func_edge_edge_mollifier_threshold(lagged, 1e-3)` | input domain of `friction_ee` |

### CCD and broad phase (`ccd.py`)

| CGQ | Genesis | Case |
|---|---|---|
| `gipc_point_triangle/edge_edge/point_edge/point_point_ccd<false>` | `func_additive_ccd` | `ccd_additive_*` |
| `directional_point_triangle/edge_edge/point_edge/point_point_ccd` | `func_directional_ccd` | `ccd_directional_*` |
| `screw_*_ccd`, `screw_*_ccd_interval`, `ScrewVert`, `screw_detail::*` | `func_screw_ccd` (interval form; the whole step is [0, 1] with the distance at 0) | `ccd_screw_*`, `ccd_screw_interval_*` |
| `ccd_ph_pair_toi`, linear branch | `func_point_halfplane_ccd` | `ccd_ph` |
| `screw_halfplane_ccd` | `func_point_halfplane_screw_ccd` | `ccd_ph_screw` |
| `pt/ee/pe/pp_candidate_reach`, `candidate_directional_reach`, `candidate_vertex_travel`, `candidate_within_reach` | `func_candidate_reach` (returns the verdict) | `candidate_*` |
| cull of `halfplane_query_kernel` | `func_point_halfplane_candidate_reach` | `candidate_ph` |

### Intersection (`intersection.py`)

| CGQ | Genesis | Case |
|---|---|---|
| `tri_edge_intersect`, `point_in_triangle`, `segment_segment_intersect` | `func_edge_triangle_intersect`, `func_point_in_triangle` | `intersect` |

### Left out, with the reason

- Spin friction (`spin_friction.h`): rod twist degrees of freedom, active only with `has_spin_friction` (rods), neither
  cloth nor rigid.
- `pair_vert_kappa_scale` (adaptive kappa controller) and `ct_query*` (contact tabular): data and controller state;
  the barrier functions take kappa and mu as values.
- Articulated (POE) CCD and the Lipschitz `path_kind` branches of CCD and candidate reach: out of scope (brief).
- Split CCD orchestration (`ccd_screw_split_first_pass`, `CCDSplitQueue`), the CCD partitioner (`ccd_emit_*`, swept
  14-DOP `*_ccd_broadphase`) and the co-motion union of `candidate_resolve`: scheduling and data structures around the
  formulas; the interval CCD and the verdict are ported.
- GIPC legacy variants (`ee_distance_flag_gipc_legacy`, `gipc_dType_EE_legacy`, `Legacy = true` ACCD) and the rank-2
  barrier family: GIPC-alignment tests only.
- `pairwise_ccd.h`: not selectable in production.
- SymPy distance gradients (`details/*.inl`): production uses none of them. Of the Hessians it uses the EE, PE and PP
  ones (mollified path, aeb58956), which Genesis writes in closed form.
- Adhesion, bond, variational adhesion, tangential adhesion, B-spline, MPM/ABD-specific contact: out of scope.
