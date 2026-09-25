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
- Preserve the result and performance of the existing pure-Rigid native path.
- Keep Rigid-Rigid and Rigid-world contact in the Genesis native contact implementation.
- Use the QIPC/CGQ system model: systems, global layout, explicit BCOO plus matrix-free operators, PCG, and one nonlinear
  lifecycle.
- Treat `qipc` and `cuda-graph-qipc` as references only. Future code is not based on either repository.
- Use Quadrants `v1.3.1` and `double` only. `float` is outside the PoC.
- Environments are hard coupling boundaries. Connectivity is global inside each environment and never crosses
  environments.
- Use one numerical advance unit, called a timestep with duration `h`. The new framework has no Genesis
  `step/substep` hierarchy.
- Reuse only the Genesis Rigid numerical kernels/functions and the typed data they require. Do not inherit the existing
  Scene, Simulator, or coupler runtime lifecycle.
- A Genesis scene may be an authoring/build-time data source. It does not schedule the new Newton solver.
- Prohibit workarounds and one-off minimal implementations.

## Current first milestone

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

finalize velocity, configuration, and FK
```

The timestep pipeline uses `@qd.kernel(graph=True)`. Newton, PCG, and line search use `qd.graph.do_while` from the
first executable version.

The current PoC:

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
- Every numerical participant derives from `PhysicsSystem`. `SimEngine.initialize()` lowers the registered participants
  into a statically dispatched `SolvePlan`; Newton lifecycle and `LinearPCG` depend on that plan rather than concrete
  system types.
- Genesis scalar generalized DOFs use the CGQ scalar packing contract: three scalar DOFs per 3x3 block, one valid DOF
  count, and identity diagonal with zero gradient in the tail lanes. No padding mask is used.
- All built instances occupy one flattened global unknown space. Disconnected instances are ordinary block-diagonal
  components of the same linear system.
- The original `Scene.step() -> Simulator -> RigidSolver` lane remains permanently available without new-framework
  allocations or dispatch and is the performance baseline.

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

## Later milestones

Only after the Rigid frameworkization is complete:

1. Add Cloth inertia and elasticity to the same framework, without self-contact.
2. Add the complete Cloth IPC self-contact and CCD pipeline.
3. Add Rigid-Cloth strong coupling.

Future contact ownership is fixed:

- Rigid-Rigid and Rigid-world: Genesis native contact.
- Cloth-Cloth: IPC.
- Rigid-Cloth: global coupling system.

Future matrix representation is fixed:

- RR: matrix-free.
- RC/CR and CC: explicit BCOO.
- PCG matvec: matrix-free apply plus BCOO SpMV.

Rigid-Cloth contact alone uses a Gauss-Newton coordinate pullback. Exact total energy and gradient remain the line-search
merit. Phase one uses finite Rigid-Cloth contact and does not claim a Rigid-Cloth non-penetration guarantee.

## Current exclusions

- Cloth implementation.
- IPC contact and CCD.
- Rigid-Cloth coupling.
- Checkpoint/yield/resume.
- Existing Genesis Scene/Simulator/coupler runtime integration.
- `float`.

## Upstream checkpoint issue

[Quadrants #750](https://github.com/Genesis-Embodied-AI/quadrants/issues/750) tracks support for a checkpoint region that
contains a child `qd.graph.do_while`. The current milestone does not use checkpoint, so this issue does not block the
Rigid graph path.

Do not add a reduced-semantics checkpoint workaround. Revisit this only before dynamic contact-capacity handling.

## Open decisions

- Large-island policy for replacing the native dense preconditioner with block, subtree, or sparse alternatives.
