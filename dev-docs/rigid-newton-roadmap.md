# Rigid Newton Framework Roadmap

Status: authoritative roadmap for branch `newton/rigid-only-baseline`.

Read [rigid-newton-development.md](rigid-newton-development.md) before editing code.

## Agent rules

- Requirements marked **Accepted** are user decisions and must be preserved.
- Items marked **Open** must not be decided by an agent. Present evidence and trade-offs, then wait for user approval.
- Do not infer architecture from the current Genesis couplers. Legacy, SAP, and IPC coupling are counterexamples for this work.
- Do not introduce a workaround for an upstream limitation or an early simplified scene.
- Do not create an implementation that must be replaced when Cloth or cross-system contact is added.
- Keep formulas and numerical behavior traceable to one implementation. Native wrappers and the graph path must call the
  same numerical functions.

## Final outcome

The eventual application is tabletop Cloth manipulation:

- Genesis supplies the robot model, minimal-coordinate Rigid dynamics, controller, and actuators.
- The Cloth is simulated by an IPC solver.
- A unified Newton system performs two-way strong Rigid-Cloth coupling.
- Cloth reaction affects the robot dynamics.

## Accepted system constraints

- Preserve the Genesis articulated minimal-coordinate Rigid representation.
- Preserve the result and performance of pure Genesis Rigid behavior.
- Keep Rigid-Rigid and Rigid-world contact in the Genesis native contact implementation.
- Use the QIPC/CGQ system model: systems, global layout, explicit BCOO plus matrix-free operators, PCG, and one nonlinear
  lifecycle.
- Treat `qipc` and `cuda-graph-qipc` as references only. Future code is not based on either repository.
- Use the local Quadrants checkpoint implementation rooted at commit
  `459a3eb57a36781e3abcd6272232c074a98f87a7` and `double` only. Update the
  release pin when that checkpoint support is released; do not implement a
  fallback. `float` is outside the PoC.
- Environments are hard coupling boundaries. Connectivity is global inside each environment and never crosses
  environments.
- Use one numerical advance unit, called a timestep with duration `h`. The new framework has no Genesis
  `step/substep` hierarchy.
- Reuse only the Genesis Rigid numerical kernels/functions and the typed data they require. Do not inherit the existing
  Scene, Simulator, or coupler runtime lifecycle.
- A Genesis scene may be an authoring/build-time data source. It does not schedule the new Newton solver.
- Prohibit workarounds and one-off minimal implementations.

## Completed Rigid milestone

Complete the Genesis RigidSolver QIPC frameworkization as one end-to-end vertical slice.

The real milestone scene is `examples/newton_coupling/franka_cube.py`. It must complete the Franka hold, grasp, and
lift sequence through the new runtime in double precision while preserving native Rigid contact, friction, controllers,
and actuators. The original `examples/rigid/franka_cube.py` remains the native behavior and performance reference.

The first executable implementation must be graph-native:

```text
predict Rigid state
assemble native candidate rows

while Newton:
    classify active rows and cone zones
    compute exact gradient
    build native Rigid Cholesky preconditioner

    initialize PCG
    while PCG:
        explicit BCOO SpMV
        + Rigid matrix-free Hessian apply
        + participant preconditioner apply

    check Newton convergence

    alpha = 1
    while binary line search:
        evaluate exact total energy
        accept or alpha *= 0.5

    accept trial

step_forward configuration and FK
update_velocity
copy_x_prev
```

The timestep pipeline uses `@qd.kernel(graph=True)`. Newton, PCG, and line search use `qd.graph.do_while` from the
first executable version.

The completed Rigid milestone:

- does not use checkpoint;
- has no eager or host-loop solver path;
- does not split matrix-free apply or native preconditioner verification into separate implementation milestones.

Matrix-free and direct-factor comparisons remain required unit tests inside the complete vertical slice.

## Accepted implementation organization

- The independent framework lives under `genesis/engine/systems`.
- Every new framework owner lives in that folder. Existing Rigid files expose shared numerical functions and never
  depend on the new framework.
- New numerical systems use `@qd.data_oriented` objects with rebindable buffers.
- Existing Genesis Rigid typed dataclasses remain the Rigid participant's internal numerical payload.
- A built `RigidSolver` publishes its numerical data to the new runtime once. The runtime never calls the old
  Simulator or coupler lifecycle.
- The global PCG is a new graph-native composable implementation supporting BCOO, matrix-free participant operators,
  and participant preconditioners from its first version.
- Every framework component derives from `SimSystem`. Numerical participants register only the engine stages they
  implement; `SimSystem` inheritance alone does not imply the whole Newton interface.
- `SimEngine.initialize()` lowers timestep lifecycle functions directly onto the engine. `LinearPCG` independently
  lowers matrix-free operator and preconditioner contributions onto itself; there is no intermediate solve-plan object.
- Genesis scalar generalized DOFs use the CGQ scalar packing contract: three scalar DOFs per 3x3 block, one valid DOF
  count, and identity diagonal with zero gradient in the tail lanes. No padding mask is used.
- All built instances occupy one flattened global unknown space. Disconnected instances are ordinary block-diagonal
  components of the same linear system.
- There is one `SimSystem` architecture. When only Genesis Rigid is active, build-time static function selection emits
  the pure-Rigid specialization directly.
- Pure-Rigid behavior is not a second engine or lifecycle. The old `Scene.step() -> Simulator -> RigidSolver` path is
  retained only as a regression and performance oracle while the specialization reaches parity.

## Rigid participant requirements

The Rigid participant must expose graph-callable numerical behavior for:

- smooth prediction;
- native candidate-row assembly;
- Newton initialization;
- exact gradient;
- matrix-free Hessian apply;
- preconditioner factorization and apply;
- exact trial energy;
- trial acceptance;
- timestep finalization.

The initial Rigid graph path must retain:

- articulated kinematics and dynamics;
- equality constraints;
- joint limits;
- native Rigid contact;
- friction loss;
- pyramidal friction;
- elliptic friction and its coupled cone Hessian;
- joint-specific integration and retraction;
- batched environments.

## Linear solve requirements

The global operator is:

$$
H
=
H_{\mathrm{BCOO}}
+
\sum_s H_{\mathrm{mf},s}.
$$

The entire RR block uses matrix-free apply. Future Rigid-Cloth/Cloth blocks use explicit BCOO where already
decided by the coupling design.

The current Genesis dense Rigid matrix is factored once per Newton iteration:

$$
P_R
=
L_RL_R^T.
$$

Every PCG preconditioner call performs triangular apply:

$$
z_R
=
L_R^{-T}L_R^{-1}r_R.
$$

Direct solve is a preconditioner, not the global solver.

For the Rigid-only system, `P_R = H_R`; the Rigid block should therefore converge in one PCG iteration up to numerical
roundoff. This is an assertion of the complete graph solve, not a separate milestone.

## Line search requirements

Use one global step length `alpha`. Start from `alpha = 1`.

Accept only when the exact trial merit decreases:

$$
\Phi(\alpha)
\le
\Phi(0).
$$

Otherwise:

$$
\alpha
\leftarrow
\frac{1}{2}\alpha.
$$

- Use binary backtracking only.
- Do not use Genesis derivative-based bracketing.
- Do not use an Armijo condition.
- Reject the Newton step if the backtracking budget or minimum alpha is exhausted.
- A rejected trial must not mutate accepted state.

## First real acceptance point

One real scene containing articulation, native contact, and friction advances a full timestep through the new graph
path and satisfies all of the following:

- CUDA graph is built with no non-graph fallback.
- Newton, PCG, and line search all run in device graph loops.
- CPU and GPU pass in `double`.
- Single and batched environments pass.
- Every required native constraint/contact mode remains available.
- Matrix-free apply matches the assembled native Hessian.
- Native Cholesky apply matches the direct native solve.
- The untouched native baseline preserves its result and performance.

## Current milestone sequence

Only after the Rigid frameworkization is complete:

1. Add Cloth inertia and elasticity to the same framework, without self-contact. **Complete.**
2. Add the complete Cloth IPC self-contact and CCD pipeline. **Complete.**
3. Migrate the complete reduced-KKT rigid contact-proxy stack from pinned CGQ. **Current milestone.**
4. Validate Rigid-Cloth coupling conformance against the pinned CGQ scenes.

The completed contact acceptance scene contains one analytical ground halfplane
and two free-falling vertical cloth sheets. One sheet spans XZ and the other
spans YZ. Their heights are staggered so frame 0 is intersection-free; the
lower sheet reaches the ground before the upper sheet collides with it and
forms the crossed pile.

The completed contact milestone includes:

- Consistent IPC PT/EE with PE/PP degeneracy dispatch;
- analytical halfplane contact;
- swept CCD;
- lagged friction;
- contact tabular parameters;
- per-body adaptive kappa with the Newton grow tick;
- checkpoint/yield/resume for pair, assembly, and global-triplet capacity.

All parameter names, units, defaults, and growth rules are fixed by
[cgq-contact-parameter-manifest.md](cgq-contact-parameter-manifest.md).

### Contact performance acceptance

Numerical success in the crossed-cloth scene is necessary but not sufficient.
During the current migration stage, every scene-scale contact phase must use a
load-balanced GPU implementation and the most efficient applicable
warp/subgroup-level algorithm. Exact CGQ launch topology is optional only when
the alternative preserves semantics and evidence shows that it matches or
exceeds the applicable CGQ production path. A merely reasonable temporary
runtime implementation is not admissible.

No non-load-balanced implementation may enter the milestone runtime. For
irregular traversal, candidate compaction, output emission, reductions, and
sparse assembly, the implementation must use the most efficient applicable
warp/subgroup-level algorithm. A scalar or thread-local algorithm is rejected
when CGQ provides a warp-frontier, warp-DFS, warp-batched, or warp-segmented
production path.

The production objective is stricter: match or exceed the highest-performance
CGQ path on representative contact workloads. Current-stage acceptance of a
different parallel decomposition does not approve it as the final architecture.
Performance debt must remain explicit until it is eliminated.

The complete rewrite queue and exit criteria are maintained in
[contact-performance-debt.md](contact-performance-debt.md). PERF-C01 (EE
rank-1 scatter), PERF-C02 (warp-batched EE writes), and PERF-C03
(load-balanced dual EE) are closed P0 gates. Their parity, overflow replay,
occupancy, and Nsight evidence must remain regression requirements.

Acceptance requires:

- parallel BVH construction and query with no single lane traversing the whole
  scene;
- load-balanced EE work distribution under anisotropic and dense-contact
  workloads;
- warp/subgroup compaction and batched output for candidate streams;
- warp/block partial reductions instead of per-item scalar global atomics where
  applicable;
- parallel contact filter, assembly, sort/reduce, CCD, and BCOO distribution;
- no brute-force production path or host fallback;
- compile/runtime profiles and size-scaling data against the pinned CGQ
  reference.

The serialized `LBVH.query_ee_dual` prototype has been removed. Milestone
acceptance now requires the replacement `DualEEQueryState` path to retain
candidate parity, exact frontier-overflow replay, warp-batched writes, and
representative-workload scaling under profiling; implementation alone does not
close the gate.

### Reduced-KKT proxy migration

Ground truth is `cuda-graph-qipc`
`main@42e7d4cbbad08739107ad830a17918f5f0f209ff`. Genesis performs no algorithm
experiments and introduces no staged substitute. The production path includes:

- one explicit massless contact proxy for every delegated Genesis mechanism
  link;
- independent accepted/trial proxy SE(3) state and ordinary proxy screw
  trajectories;
- `constraint`, `tangent_map`, `normal_map`, `particular`, `slack`,
  `reaction`, path-limit, filter, merit, and restoration state with CGQ names
  and defaults;
- hard reduction `d = d_p + P y` and
  `P^T H P y = -P^T (gradient + H d_p)`;
- dummy proxy rows in the graph-static global layout, expanded physical
  SpMV, exact restriction, and source-agnostic mapped-tree preconditioning;
- proxy/FK curvature limiting before trajectory publication, ordinary screw
  CCD, exact trial residual guard, lazy exact-L1 merit, frame-local elastic
  restoration, hard-probe verification, and a pre-commit failure yield;
- no persistent finite AL path, proxy soft-joint energy, user coupling
  stiffness, state-level FK substitution, post-line-search snap, or
  articulated CCD.

CGQ appends proxies to its growable maximal `RigidBodyDynamics`. Genesis native
`RigidSolver` link storage cannot grow after scene build, so
`RigidContactProxySystem` owns the independent proxy SE(3) state while
preserving the CGQ public field names and publishes that state to the global
contact managers. This storage-owner adaptation must not change the KKT maps,
pipeline order, contact routes, or proxy screw contract.

Contact ownership is fixed:

- Rigid-Rigid and Rigid-world: Genesis native contact unless a pair is
  explicitly delegated to a proxy route.
- Cloth-Cloth: IPC.
- Rigid-Cloth: IPC through the explicit proxy surface.

Matrix representation is fixed:

- RR: matrix-free.
- Physical proxy-proxy, proxy-Cloth, and Cloth-Cloth contact: explicit BCOO.
- Reduced PCG matvec: expand through `P`, apply physical BCOO plus native
  matrix-free terms, then restrict through `P^T`.
- The reduced matrix is never formed explicitly.

## Current exclusions

- Explicit codimensional rod/particle PE/PP broad phase.
- Adhesion, variational adhesion, bonds, and contact topology mutation.
- Existing Genesis Scene/Simulator/coupler runtime integration.
- `float`.

## Checkpoint requirement

[Quadrants #750](https://github.com/Genesis-Embodied-AI/quadrants/issues/750)
tracked support for a checkpoint region containing child `qd.graph.do_while`
nodes. The local Quadrants commit pinned above contains that support and is now
required by dynamic contact-capacity handling.

The contact graph must yield only on a real capacity overflow, reallocate the
owning buffers, clear the triggering device scalar, and resume from the exact
CGQ phase. Do not add fixed-capacity failure behavior or flatten nested graph
loops as a fallback.

## Open decisions

- Large-island policy for replacing the native dense preconditioner with block, subtree, or sparse alternatives.
