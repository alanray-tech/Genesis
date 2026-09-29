# Reduced-KKT Contact Proxy Migration Contract

Status: current implementation contract.

Ground truth is `cuda-graph-qipc`
`main@42e7d4cbbad08739107ad830a17918f5f0f209ff`.

Genesis World performs no reduced-KKT algorithm experiments. Formulas, names,
defaults, graph order, failure behavior, and accepted/rejected production
paths are migrated from that revision.

## Authoritative CGQ sources

- `docs/rigid_contact_proxy_kkt.md`
- `qipc/_src/native/solver/rigid_contact_proxy_context.h`
- `qipc/_src/native/solver/rigid_contact_proxy.h`
- `qipc/_src/native/solver/rigid_contact_proxy.cu`
- `qipc/_src/native/solver/rigid_contact_proxy_kkt.h`
- `qipc/_src/native/solver/rigid_contact_assemble.h`
- `qipc/_src/native/solver/rigid_contact_assemble.cu`
- `qipc/_src/native/solver/rigid_joint_forest_context.h`
- `qipc/_src/native/solver/rigid_joint_forest.h`
- `qipc/_src/native/solver/rigid_joint_forest.cu`
- `qipc/_src/native/solver/sim_engine_pipeline.cu`
- `tests/rigid_proxy_kkt_reference.py`
- `tests/test_rigid_contact_proxy.py`
- `tests/test_rigid_contact_proxy_kkt.py`
- `examples/rigid_proxy_kkt_restoration.py`
- `examples/minimal_wrecking_ball_ground.py`

## Production model

The solved problem is

```text
minimize E_A(x, g) + E_B(q)
subject to c(g, q) = 0
```

`x` contains non-rigid physical unknowns, `g` contains independent massless
proxy poses, and `q` contains Genesis minimal coordinates.

The ordinary hard-KKT direction is

```text
d_p = [0, -A^-1 c, 0]
d = d_p + P y
(P^T H P) y = -P^T (gradient + H d_p)
```

The implementation must not form `P^T H P`. PCG applies it as

```text
reduced direction
-> mechanism/link expansion
-> proxy expansion through tangent_map
-> physical BCOO plus native matrix-free apply
-> proxy restriction through tangent_map^T
-> mechanism/minimal restriction
-> reduced result
```

The proxy pose remains an independent nonlinear state. Only its linearized
increment is eliminated. Contact, friction, broad phase, trajectory
publication, and CCD consume only proxy geometry and proxy screw paths.

## Forbidden alternatives

- Persistent finite augmented Lagrangian coupling.
- Proxy soft-joint energy or user coupling stiffness.
- State-level substitution of proxy pose by FK.
- Explicit formation of the reduced dense contact matrix.
- Post-line-search proxy snap or FK projection.
- Articulated product-of-exponentials CCD.
- Omitting `H d_p` from the reduced RHS.
- Committing a frame while equality feasibility, restoration, or hard-probe
  verification is incomplete.
- A simplified hard-only runtime in place of production merit and restoration.

## Storage-owner adaptation

CGQ appends mechanism and proxy bodies to one growable
`RigidBodyDynamics`. Genesis native `RigidSolver` link arrays are immutable
after scene build and cannot accept appended proxy bodies.

Therefore:

- `RigidSystem` remains the owner of Genesis minimal dynamics and FK link
  state;
- `RigidContactProxySystem` owns independent proxy translation, quaternion,
  accepted history, trial state, twist, and KKT workspace;
- `mechanism_body` identifies the Genesis link/environment owner;
- `proxy_body` identifies the appended global contact-body row, not an entry
  fabricated inside native `RigidSolver`;
- scene initialization derives each proxy pose from the built link pose and
  its inertial transform. Genesis `links.i_pos/i_quat` are not valid staging
  sources before the first native predict/FK pass;
- proxy vertices and surfaces are published by
  `RigidContactProxySystem` into the existing global managers;
- this ownership difference must not change any CGQ KKT field name, map,
  tolerance, contact route, or graph dependency.

No host object map is permitted. Host geometry extraction is build-time
staging only; all live counts, modes, capacities, residuals, and iteration
state are zero-dimensional device scalars accessed with `[()]`.

Genesis currently solves for native trial acceleration, whereas CGQ's forest
direction is already a configuration-space body twist. For ordinary
semi-implicit joint integration the tangent factor is `dt^2`; the forest
applies that factor symmetrically in expansion and restriction. Every
proxy-owned mechanism link is marked in native `links.is_constrained` after
native candidate assembly. This disables Genesis's unconstrained free-root
midpoint override, so delegated roots and articulated joints use the standard
constraint integration map whose position tangent is `dt^2`. Integrator
conformance tests must verify this gate before `build_scene_engine` selects
the proxy path; an integrator with a different constrained tangent must provide
its exact map or be rejected explicitly.

Genesis retains authored fixed MJCF links that CGQ's minimal forest does not
represent as scalar joint edges. The mapped articulated preconditioner
therefore:

- stores the CGQ scalar `edge_*` state only for one-DOF revolute/prismatic
  edges;
- transports a fixed child's complete spatial block into its parent without a
  scalar Schur elimination;
- seeds each Genesis link's spatial mass/inertia block before gathering
  physical proxy BCOO, because Genesis's native generalized rigid Hessian is
  matrix-free and is not duplicated in physical BCOO;
- adds Genesis's authored armature and timestep-scaled joint damping to the
  matching scalar/root pivots, preserving the native mass augmentation;
- uses the mandatory CGQ names `articulated_inertia`, `edge_u`,
  `edge_hessian`, `edge_basis`, `edge_arm`, `edge_d`, `root_inverse`,
  `precond_force`, `precond_a`, `precond_velocity`, and
  `kkt_proxy_diagonal`.

This is a storage/source adaptation, not a different preconditioner: reverse
factorization, root solve, forward substitution, proxy tangent pullback, and
restoration slack block remain the CGQ operations. Unsupported multi-DOF
non-root joints and non-free moving roots are rejected at build time rather
than entering an approximate path.

Three Python/Genesis-only names are explicit exceptions to the CGQ field
manifest:
`root_dof_index` maps each native six-DOF free root into Genesis's compact
generalized rows, and `body_twist` stores expanded link twists that CGQ stores
in growable `RigidBodyDynamics::dq`; `lambda_` is the Python spelling of CGQ's
`lambda` because `lambda` is a Python keyword. The first two exist only
because Genesis's native generalized/link storage cannot be expanded or
relaid out.

## RigidContactProxySystem state

The following CGQ names are mandatory:

- mappings: `mechanism_body`, `proxy_body`, `pair_of_body`,
  `surface_radius`;
- KKT data: `constraint`, `tangent_map`, `normal_map`, `particular`,
  `slack`, `reaction`;
- restoration data: `lambda`, `metric`, `dual_update_flag`,
  `restoration_active`, `restoration_triggered`,
  `restoration_hard_probe`, `restoration_reprice_flag`,
  `restoration_entries`, `restoration_newton_epochs`,
  `restoration_dual_epochs`, `restoration_hard_probes`,
  `restoration_max_slack`;
- path and convergence data: `path_limit`, `max_surface_residual`,
  `fk_alpha`, `trial_max_residual`, `solve_tolerance`,
  `physical_converged`, `frame_failed`;
- globalization data: `filter_h`, `filter_energy`, `filter_size`,
  `filter_retry`, `merit_gradient`, `merit_dot_partial`,
  `merit_control_gradient`, `merit_active`, `merit_probe`,
  `merit_evaluated`, `merit_gtd`, `merit_rho`, `merit_slope`,
  `merit_entries`, `merit_probes`, `merit_accepts`, `merit_trials`,
  `filter_retries`, `merit_max_rho`, `merit_max_gtd`,
  `ls_exhaust_accept_count`;
- reduction workspaces: `energy_partial`, `residual_partial`.

Production config is fixed:

- `rigid_proxy/globalization = "merit"`;
- `rigid_proxy/restoration = 1`;
- `rigid_proxy/test_merit_energy_bias = 0.0`.
- `rigid_forest/fused = 1`.

`"watchdog"`, restoration-off, and nonzero test bias are diagnostic controls,
not production alternatives.
`extras/ls_forensics/test_energy_bias` is the pinned CGQ one-shot
line-search-exhaustion test control; production fixes it to zero.

## Constraint maps

For each proxy pair:

```text
c_t = t_proxy - t_fk
c_w = Log(R_proxy R_fk^T)
```

With `phi = c_w`:

```text
A = diag(I, J_l(phi)^-1)
B = [J_v; J_r(phi)^-1 J_w]
```

The stored maps are:

```text
tangent_map = A^-1 B_link
normal_map  = A^-1
particular  = -A^-1 c
```

For matching world-frame link/proxy charts, CGQ simplifies these to:

```text
tangent_map.rotation = J_l(phi) J_r(phi)^-1
normal_map.rotation  = J_l(phi)
particular.translation = -c_t
particular.rotation    = -J_l(phi) c_w
```

`expand_proxy_twist`, `restrict_proxy_wrench`, `expand_slack_twist`, and
`restrict_slack_wrench` must be mathematical transpose twins and share one
implementation of each stored map.

## Graph-static layout

- Reduced rows contain Genesis minimal coordinates and FEM vertices.
- Every proxy keeps two three-DOF dummy block rows in the graph-static global
  storage.
- Expanded physical `p` and `Ap` workspaces contain FEM, mechanism-link, and
  proxy rows.
- In hard mode, proxy dummy rows are identity rows and are not independent
  reduced unknowns.
- In restoration mode, those rows become explicit slack coordinates through
  `normal_map` and receive the restoration metric.
- Physical contact routes write BCOO in expanded proxy/FEM coordinates.
- The reduced operator expands before BCOO SpMV and restricts afterward.

## Component partition contract

One minimal robot is an indivisible solve and trajectory component.

- Register a forced component edge for every mechanism/proxy pair.
- Register every parent/child edge in the Genesis joint forest.
- Masked PCG must assign one convergence mask to all reduced coordinates,
  mechanism-link work rows, and proxy dummy/slack rows in that component.
- The CCD component partitioner must assign one alpha to the same complete
  component.
- Component labels are device-resident and may grow only through the approved
  checkpoint protocol.

The current authorized migration stage may select reduced KKT with the
standard global `LinearPCG` and one global screw-CCD alpha. This is the
functional standard path, not a claim of component-level performance or
independent-component progress. `ComponentPartitioner`,
`CCDComponentPartitioner`, forced forest/proxy edges, and masked PCG remain
mandatory follow-up work and must not change the reduced-KKT mathematics.

## Pipeline order

Frame open:

1. predict Genesis and FEM state;
2. reset merit/filter/restoration frame state;
3. prepare the spatial-inertia restoration metric;
4. initialize proxy pose/history consistently on scene init and reset.

Each Newton iteration:

1. derive solve scope;
2. evaluate exact Genesis FK;
3. reset and prepare `constraint`, `tangent_map`, `normal_map`,
   `particular`, and `max_surface_residual`;
4. assemble only physical objective gradient/Hessian;
5. build physical BCOO and sort/reduce;
6. snapshot the raw physical gradient when merit is enabled;
7. prepare physical `d_p`;
8. compute physical `H d_p`;
9. build `-P^T (gradient + H d_p)`;
10. gather and factor the mapped contact-aware preconditioner;
11. solve the matrix-free reduced operator;
12. expand the solved direction, add `particular`, and optional slack;
13. apply world-displacement clamp;
14. tighten the FK-defect prefix;
15. publish proxy screw endpoints;
16. build swept BVHs and run ordinary proxy screw CCD;
17. run shared line search with exact proxy/FK trial guard and physical
    energy fast path;
18. lazily evaluate exact-L1 merit only after an energy rejection that
    reduces equality residual;
19. finalize equality convergence and optional one-step elastic restoration;
20. run hard-probe verification before frame commit.

Before commit:

- physical stationarity is converged;
- `max_surface_residual <= solve_tolerance`;
- restoration is inactive;
- no hard probe is pending;
- `frame_failed == 0`.

Failure yields before previous-state and velocity commit. Proxy state is never
silently snapped to FK.

## Contact ownership and geometry

- Native Rigid-Rigid and Rigid-world routes remain enabled unless explicitly
  delegated.
- Cloth-Cloth uses ordinary IPC.
- Rigid-Cloth uses proxy-Cloth IPC.
- Delegated rigid geometry is removed from the matching native pair route.
- Every geometry pair has exactly one owner.

Rigid collision meshes are extracted from the built Genesis rigid geometry at
build time, transformed into their owning link frame, merged per delegated
link/environment, and published as proxy-local vertices plus global
triangle/edge topology. Runtime world positions and trajectory endpoints come
only from the independent proxy pose and twist.

## Tolerances and path safety

No user proxy tolerance or coupling stiffness is introduced.

```text
epsilon_path  = 0.1 * contact.d_hat
epsilon_solve = min(sim.abs_tol, 0.01 * contact.d_hat)
```

The FK-defect limiter uses the CGQ cancellation-safe positive-root branches
for

```text
h(alpha) <= h0 + (h_slack - h0) alpha + 0.5 L alpha^2.
```

The exact trial guard is a device invariant. It is not ordinary line-search
rejection.

## Migration order

This order is dependency order, not a sequence of reduced algorithms:

1. Port the complete state manifest and SE(3) KKT functions.
2. Publish independent proxy pose and surface state.
3. Add physical proxy/FEM contact distribute routes.
4. Add particular-step SpMV and reduced RHS.
5. Add matrix-free expansion/restriction around physical SpMV.
6. Add the source-agnostic mapped-tree preconditioner.
7. Add forced forest/proxy component edges, component CCD, and masked PCG.
8. Add proxy screw trajectory publication, FK-defect clamp, and exact guard.
9. Add lazy merit, filter, frame-local restoration, hard probe, and failure
   checkpoint.
10. Port the pinned CGQ conformance tests and examples.

No intermediate item may be registered by `builders.py` as a production
runtime until all dependencies required by its selected path are complete.

## Acceptance

Port the pinned CGQ gates rather than designing new experiments:

- finite-difference pose constraint Jacobians;
- `C P = 0` and `C d_p = -c`;
- dense-KKT parity with nonzero residual and mandatory `H d_p`;
- elastic-slack dense augmented-system parity;
- reduced matvec symmetry and positive definiteness;
- conservative FK-defect prefix tests;
- feasible-anchor and perturbed-proxy integration;
- massless proxy/no-contact inertia behavior;
- contact ownership and proxy geometry ownership;
- coherent component labels, PCG masks, and CCD alpha across every mechanism,
  forest edge, and proxy;
- table press and Cloth grasp;
- restoration then hard-probe verification;
- Newton-budget failure without commit;
- singleton minimal-coordinate scale;
- dump/recover with no persistent restoration dual.

The original Genesis native runtime remains the behavior and performance
oracle whenever the new engine is not selected.
