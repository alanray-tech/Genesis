# P1: IPC contact formulas in vector form, parity with CGQ

Library: `genesis/engine/ipc/` on branch `ipc-formulas` (worktree `genesis-world-ipc-formulas`). Harness: this folder,
on branch `ipc-bvh-bench`. Ground truth: CGQ `origin/main@0496c4fa` (`cuda-graph-qipc-bvh-bench`).

## Result

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

- Mollified EE: CGQ production calls the rank-2 (dH^2-prefixed) algebra with kappa / dH^2; Genesis uses the un-prefixed
  algebra, identical by expansion (checked against CGQ's own un-prefixed form, case `barrier_ee_unprefixed`).
- Additive CCD reuses `func_point_triangle_distance_flag` where CGQ uses `gipc_dType_PT` (same Cramer system, Eigen
  determinants, contractible arithmetic); no row needed more than pass1 there.
- PP / PH tangent bases come from `genesis.utils.geom.qd_orthogonals`, a different orthonormal basis than CGQ's; the
  friction terms are invariant to it, and the harness compares the projector T T^T.
- Thresholds are parameters (Hessian clamp ratio 1e-6, mollifier ratio 1e-3, EE parallel tolerances 1e-6 and 1e-14,
  intersection tolerance 1e-12, CCD eta and iteration budget), to come from info fields in the solver; underflow guards
  (1e-300, 1e-30) stay literals.
- 12x12 matrices: the harness kernels (12x12 Hessians, CCD loops) compile and run in about 30 s per backend for all 52
  cases, no compile-time concern observed; the solver integration should still measure its own kernels.
- The half-plane cull `func_point_halfplane_candidate_reach` and the `barrier_ee_unprefixed` case were added after the
  brief (production reaches the cull in `halfplane_query_kernel`).

## Module layout

| Module | Content |
|---|---|
| `distance.py` | distance flags (PT, EE, PE, the parallel-projecting EE type of the additive CCD), squared distances of the primitives and their gradients, flagged distance and gradient per stencil, edge cross norm and its gradient, closest points (PT, EE, PE, generic over the stencil shape) |
| `barrier.py` | Consistent IPC barrier and its derivatives, normal force, mollifier threshold and mollified barrier, PSD gradient/Hessian (plain and mollified), half-plane gradient/Hessian, pair parameters (d_hat, xi, area weight) |
| `friction.py` | tangent frame and closest-point weights per stencil (PT, EE, PE, PP, PH), friction energy and PSD gradient/Hessian shared by all stencils |
| `ccd.py` | additive, directional and screw (interval form) conservative advancement, each generic over the stencil shape; half-plane CCD, linear and screw; broad-phase distance rule (simplex and half-plane) |
| `intersection.py` | edge-triangle intersection |

The split follows the brief. Two choices shape it:
- Every barrier gradient/Hessian is one function of the squared distance D and its gradient over the stencil,
  `func_barrier_gradient_hessian(D, grad D, ...)`: CGQ's col 8 of PFPx is grad sqrt(D) / sqrt(dH), and its rank-1
  Hessian equals `lambda0 / (4 (D - xi^2) dH) * grad D grad D^T`. So the per-type CGQ functions (PT, PE, PP, the EE
  rank-1 fast path) become flag classification + `func_*_flagged_distance2` (D and its gradient, scattered over the
  4-vertex stencil) + one barrier function. The mollified EE barrier is the same with the cross norm and its gradient
  (CGQ's col 4 of the mollified PFPx is grad sqrt(I1)), all in EE stencil order: CGQ's PE/PP subcase permutations
  disappear.
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
| PFPx col 4 of the mollified PFPx, `(p1-p0).cross(p3-p2).squaredNorm()` | `func_edge_edge_cross_norm2` (I1 and its gradient) | `mollifier`, `barrier_ee` |
| `detail::closest_pt/ee/pe` | `func_point_triangle_closest_point`, `func_edge_edge_closest_points`, `func_point_edge_closest_point`, `func_closest_points` | `closest_*` |

### Barrier (`barrier.py`)

| CGQ | Genesis | Case |
|---|---|---|
| `gipc_barrier_energy`, `_first_derivative`, `_second_derivative` | `func_barrier` | `barrier` |
| `gipc_normal_force` | `func_barrier_normal_force` | `barrier`, friction |
| `gipc_compute_eps_x` | `func_edge_edge_mollifier_threshold(..., mollifier_ratio)` | `mollifier` |
| `gipc_barrier_energy_mollified` | `func_mollified_barrier` | `mollified_barrier` |
| `gipc_barrier_grad_hess_pt/pe/pp`, `_ee_rank1`, `gipc_pfpx_*`, `_barrier_lambda0` (gassThreshold clamp) | `func_barrier_gradient_hessian(D, grad D, ..., hessian_clamp_ratio)` | `barrier_pt/ee/pe/pp` |
| `gipc_rank2::gipc_barrier_grad_hess_ee/pe/pp_mollified` with `kappa / dH^2` (what production calls), `_barrier_grad_hess_mollified_impl`, `gipc_make_pd<2>` | `func_mollified_barrier_gradient_hessian` (the un-prefixed algebra, identical by expansion; the `barrier_ee_unprefixed` case checks CGQ's own un-prefixed form) | `barrier_ee` |
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
- SymPy distance gradients/Hessians (`details/*.inl`): production uses none of them.
- Adhesion, bond, variational adhesion, tangential adhesion, B-spline, MPM/ABD-specific contact: out of scope.
