# Rigid Newton Framework Roadmap

Status: authoritative roadmap for the current checkout.

Read [rigid-newton-development.md](rigid-newton-development.md) before editing code.

## Agent rules

- Requirements marked **Accepted** are user decisions and must be preserved.
- Items marked **Open** must not be decided by an agent. Present evidence and trade-offs, then wait for user approval.
- Do not infer architecture from the current Genesis couplers. Legacy, SAP, and IPC coupling are counterexamples for this work.
- Do not introduce an undocumented workaround for an upstream limitation. A
  temporary workaround requires an upstream issue, a regression test, measured
  cost, and explicit approval.
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
- Keep native Rigid collision code unchanged. The coupled first version disables
  that collision pass to avoid duplicate work and supports Rigid surfaces only
  as Cloth-contact proxies; Rigid-Rigid and Rigid-world contact are not part of
  the public coupled support matrix.
- Use one graph-native system model: systems, global layout, explicit BCOO
  plus matrix-free operators, LinearPCG, and one nonlinear lifecycle.
- External numerical implementations are optional references only, with no
  assumed checkout name or location.
- Use the Quadrants version pinned by `pyproject.toml` and `double` only.
  `float` is outside the first version.
- Environments are hard coupling boundaries. Connectivity is global inside each environment and never crosses
  environments.
- Use one numerical advance unit with duration `h`. The first version requires
  one numerical advance per `Scene.step()`.
- Reuse only the existing Rigid numerical kernels/functions and typed data.
  Keep the Rigid implementation independent of the graph runtime.
- Integrate through `NewtonCoupler`, preserving the standard Scene and
  Simulator callbacks, recorders, sensors, clocks, and visualization.
- Prohibit undocumented workarounds and scene-specific one-off implementations.

## Completed Rigid-only extraction milestone

Complete the RigidSolver graph-native frameworkization as one end-to-end
vertical slice.

The extraction milestone used `examples/newton_coupling/franka_cube.py` to
complete a Franka hold, grasp, and lift sequence through the graph runtime in
double precision while preserving native Rigid contact, friction, controllers,
and actuators. `examples/rigid/franka_cube.py` remains the native behavior and
performance reference.

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
- A built `RigidSolver` publishes its numerical data to `RigidSystem` once.
  `NewtonCoupler` invokes `SimEngine` from the standard Simulator step without
  running the native solver/coupler substep loop.
- The global PCG is a new graph-native composable implementation supporting BCOO, matrix-free participant operators,
  and participant preconditioners from its first version.
- Every framework component derives from `SimSystem`. Numerical participants register only the engine stages they
  implement; `SimSystem` inheritance alone does not imply the whole Newton interface.
- `SimEngine.init()` lowers timestep lifecycle functions directly onto the engine. `LinearPCG` independently
  lowers matrix-free operator and preconditioner contributions onto itself; there is no intermediate solve-plan object.
- Scalar generalized DOFs use the reference packing contract: three scalar DOFs per 3x3 block, one valid DOF
  count, and identity diagonal with zero gradient in the tail lanes. No padding mask is used.
- All built instances occupy one flattened global unknown space. Disconnected instances are ordinary block-diagonal
  components of the same linear system.
- There is one `SimSystem` architecture. The first public coupling path
  requires QCloth; the lower-level Rigid-only builder remains a development
  and parity surface.
- Scenes not selecting `NewtonCouplerOptions` retain their existing
  `Scene.step() -> Simulator -> solver/coupler` behavior unchanged.

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
- the existing scalar generalized-coordinate layout for one unbatched
  environment. Batched execution is deferred beyond the first version.

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
- GPU passes in `double`; unsupported CPU execution is rejected at build.
- One unbatched environment passes; unsupported batching is rejected at build.
- Every required native constraint/contact mode remains available.
- Matrix-free apply matches the assembled native Hessian.
- Native Cholesky apply matches the direct native solve.
- The untouched native baseline preserves its result and performance.

## Current milestone sequence

Only after the Rigid frameworkization is complete:

1. Add Cloth inertia and elasticity to the same framework, without self-contact. **Complete.**
2. Add the complete Cloth IPC self-contact and CCD pipeline. **Complete.**
3. Migrate the complete reduced-KKT rigid contact-proxy stack from the pinned reference. **Complete.**
4. Integrate the graph runtime through `NewtonCouplerOptions` and standard
   `Scene.step()`, with scripted Franka-Cloth grasp and Cloth-stack examples.
   **Complete.**
5. Finish full pinned-reference conformance and remove explicitly accepted
   temporary performance debt. **Current milestone.**

The first-version public examples are
`examples/newton_coupling/franka_cloth_grasp.py` and
`examples/newton_coupling/cloth_stack.py`. Both use `NewtonCouplerOptions` and
advance exclusively through `Scene.step()`.

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
[contact-parameter-manifest.md](contact-parameter-manifest.md).

### Contact performance acceptance

Numerical success in the crossed-cloth scene is necessary but not sufficient.
During the current migration stage, every scene-scale contact phase must use a
load-balanced GPU implementation and the most efficient applicable
warp/subgroup-level algorithm. Exact reference launch topology is optional only when
the alternative preserves semantics and evidence shows that it matches or
exceeds the applicable pinned-reference production path. A merely reasonable temporary
runtime implementation is not admissible.

No non-load-balanced implementation may enter the milestone runtime. For
irregular traversal, candidate compaction, output emission, reductions, and
sparse assembly, the implementation must use the most efficient applicable
warp/subgroup-level algorithm. A scalar or thread-local algorithm is rejected
when the reference provides a warp-frontier, warp-DFS, warp-batched, or warp-segmented
production path.

The production objective is stricter: match or exceed the highest-performance
pinned-reference path on representative contact workloads. Current-stage acceptance of a
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
- compile/runtime profiles and size-scaling data against the pinned reference
  reference.

The serialized `LBVH.query_ee_dual` prototype has been removed. Milestone
acceptance now requires the replacement `DualEEQueryState` path to retain
candidate parity, exact frontier-overflow replay, warp-batched writes, and
representative-workload scaling under profiling; implementation alone does not
close the gate.

### Reduced-KKT proxy migration

The reduced-KKT formulas and ordering are traced to a pinned native numerical
reference. The runtime performs no scene-specific algorithm experiment and
introduces no staged substitute. The production path includes:

- one explicit massless contact proxy for every delegated Genesis mechanism
  link;
- independent accepted/trial proxy SE(3) state and ordinary proxy screw
  trajectories;
- `constraint`, `tangent_map`, `normal_map`, `particular`, `slack`,
  `reaction`, path-limit, filter, merit, and restoration state with reference names
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

The native reference appends proxies to its growable maximal `RigidBodyDynamics`. Native
`RigidSolver` link storage cannot grow after scene build, so
`RigidContactProxySystem` owns the independent proxy SE(3) state while
preserving the reference field names and publishes that state to the global
contact managers. This storage-owner adaptation must not change the KKT maps,
pipeline order, contact routes, or proxy screw contract.

Contact ownership is fixed:

- Rigid-Rigid and Rigid-world: unchanged native contact outside the coupled
  runtime; unsupported by the public coupled first version.
- Cloth-Cloth: Consistent IPC.
- Rigid-Cloth: Consistent IPC through the explicit proxy surface.

Matrix representation is fixed:

- RR: matrix-free.
- Physical proxy-proxy, proxy-Cloth, and Cloth-Cloth contact: explicit BCOO.
- Reduced PCG matvec: expand through `P`, apply physical BCOO plus native
  matrix-free terms, then restrict through `P^T`.
- The reduced matrix is never formed explicitly.

## Current exclusions

- Explicit codimensional rod/particle PE/PP broad phase.
- Adhesion, variational adhesion, bonds, and contact topology mutation.
- MinCoo, MaskedPCG, and component-partitioned solving.
- CPU, `float`, differentiation, batched environments, and multiple substeps.
- FEM materials other than `FEM.QCloth` and active solvers other than Rigid
  plus FEM.

## Checkpoint requirement

Quadrants issue #750
tracks the general case of a checkpoint region containing child
`qd.graph.do_while` nodes. The Genesis pipeline does not require that graph
shape: capacity checks yield before iterative consumers, the explicit `SOLVE`
checkpoint ends after idempotent post-growth assembly, and PCG and line search
run as checkpoint-external child loops. Every inlined `qd.func` stage around
those loops is nevertheless enclosed by its own flat, no-yield checkpoint.
This is a correctness workaround: checkpoint auto-wrapping skips direct
kernel-AST tasks, but currently fails to propagate the resume gate across a
top-level inlined `qd.func`, allowing that function to execute once after an
earlier yield. `test_contact_checkpoint_yield_skips_pcg` pins the bug by
forcing a `SORT` yield and proving that a PCG iteration-count sentinel remains
untouched; the compiler defect is tracked by Quadrants issue #956.
Quadrants PR #944
implements the generic checkpoint-containing-child-loop shape, but is not a
Genesis runtime dependency.

The contact graph must yield only on a real capacity overflow, reallocate the
owning buffers, clear the triggering device scalar, and resume from the exact
reference phase. Do not add fixed-capacity failure behavior or move an overflow
check after work that consumes the capacity it protects.

## Open decisions

- Large-island policy for replacing the native dense preconditioner with block, subtree, or sparse alternatives.
