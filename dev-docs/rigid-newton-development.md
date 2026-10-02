# Rigid Newton Framework Development Guide

This document is the implementation handoff for the graph-native Newton
coupling runtime in the current checkout.

Read [rigid-newton-roadmap.md](rigid-newton-roadmap.md) first. The roadmap owns requirements and milestone order.
This guide records the current workspace, relevant source code, development procedure, and verification commands.
Read
[graph-native-simulation-system-design.md](graph-native-simulation-system-design.md)
for the SimSystem, SimEngine, lifecycle, data-scope, and Manager + Reporter
contracts.

## Supported baseline

- Repository root: the checkout containing this document.
- Quadrants dependency: the version pinned by `pyproject.toml`.
- Numerical precision: `double`.
- Runtime: GPU with the ndarray backend.

Optional Python, native, and compiler reference implementations may be checked
out anywhere. The runtime does not discover them, import them, or require a
particular directory layout. Treat all external implementations as read-only
references rather than package dependencies.

## Agent collaboration protocol

1. Make changes only in the current checkout. Treat comparison checkouts as read-only.
2. Read both Rigid Newton documents before changing code.
3. Check `git status --short` before editing. Preserve user changes.
4. Do not decide an item marked **Open** in the roadmap.
5. Keep the graph runtime behind `NewtonCoupler`; do not route it through the
   Legacy/SAP/IPC coupling implementations.
6. Do not create an eager or host-loop implementation.
7. Use checkpoint only for contact-capacity overflow and exact phase resume.
8. Do not create a participant-private PCG.
9. Do not copy numerical formulas into a second implementation. Refactor shared `qd.func` entry points instead.
10. Run the relevant GPU tests before handoff; record the device and software
    versions for every performance claim. Unsupported CPU execution must be
    rejected.
11. Reject every non-load-balanced scene-scale GPU implementation. This is a
    hard code-admission rule, not an optimization suggestion.
12. Traversal, compaction, reduction, segmented reduction, candidate emission,
    sparse assembly, and other irregular work must use the most efficient
    applicable warp/subgroup-level algorithm. Reject scalar or thread-local
    code when the pinned reference provides a warp-frontier, warp-DFS, warp-batched, or
    warp-segmented production path.
13. Do not admit one-task whole-scene traversal, unbounded per-lane work,
    per-item global reservation when warp batching applies, or scalar global
    reduction when warp/block reduction applies.
14. A simpler diagnostic implementation may exist only in isolated tests or
    offline comparison tools. It must not be registered by a builder or be
    reachable from a production or milestone runtime.
15. Numerical parity, a passing example, or a temporary-performance-debt entry
    never waives rules 11--14. Production work must match or exceed the
    highest-performance applicable pinned-reference path.

## Local environment

Commands below use a repository-local `.venv` for illustration. Any Python
environment satisfying `pyproject.toml` is valid.

Use the dependency versions declared by `pyproject.toml`. Select a
double-precision GPU backend supported by those dependencies. No specific GPU
model, workspace path, or sibling checkout is required.

Verify the active environment:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

Set UTF-8 for test runs so Genesis logging renders correctly:

```powershell
$env:PYTHONUTF8='1'
```

## Current architecture

The existing `RigidSolver` remains the owner of minimal-coordinate state and
the native numerical implementation. `RigidSystem` references its buffers and
shared `qd.func` stages; the dependency remains one-way from the graph runtime
to the Rigid numerical core. The first-version integration intentionally keeps
`genesis/engine/solvers/rigid/` unchanged.

Users select the graph runtime with `NewtonCouplerOptions` and advance it
through ordinary `Scene.step()` calls. `NewtonCoupler` validates the supported
configuration and constructs `SimEngine` lazily on the first step, after
post-build qpos, controller, and QCloth-constraint setup. Scene callbacks,
recorders, sensors, clocks, and visualization remain on the standard Scene
lifecycle.

The graph runtime is implemented under `genesis/engine/systems`:

- `sim_system.py`: system registration and dependency lookup;
- `sim_engine.py`: direct build-time lowering of registered timestep lifecycle functions;
- `global_linear_system.py`: global vectors and symmetric BCOO;
- `linear_pcg.py`: composable device-resident PCG and direct static lowering of operator/preconditioner contributions;
- `rigid_system.py`: Genesis Rigid numerical participant;
- `rigid_contact_proxy.py`: massless proxy SE(3) state and reduced-KKT workspace;
- `rigid_contact_proxy_kkt.py`: shared `A^-1`, `P`, `P^T`, slack, and FK-defect algebra;
- `rigid_joint_forest.py`: Genesis minimal-coordinate link expansion/restriction and reduced operator hooks;
- `rigid_contact_assemble.py`: proxy-proxy and proxy-FEM physical contact routes;

Read
[reduced-kkt-contact-proxy.md](reduced-kkt-contact-proxy.md)
before changing any
of the proxy, forest, contact-route, PCG, CCD, globalization, or commit-gate
code. `NewtonCoupler` selects this path for supported QCloth scenes; the
lower-level builder remains an internal composition and testing surface.

## First-version support matrix

Supported:

- GPU, double precision, and the ndarray backend;
- one unbatched environment and `SimOptions.substeps=1`;
- at least one `FEM.QCloth` entity, with optional Rigid entities;
- native Newton Rigid constraints without differentiation, no-slip
  post-processing, or hibernation;
- Consistent IPC contact, graph-parallel BVH/contact phases, global BCOO plus
  matrix-free Rigid terms, and `StandardPCGSolver`/`LinearPCG`;
- hard fixed QCloth vertex constraints;
- standard `Scene.step()` and full-scene `Scene.reset()`.

Unsupported:

- CPU, single precision, differentiation, batching, and multiple substeps;
- other active solvers or FEM materials;
- Rigid-Rigid and Rigid-world collision inside the coupled runtime;
- MinCoo, MaskedPCG, component-partitioned solving, alternative contact
  constitutions, adhesion, or topology mutation;
- partial-environment reset.

The engine is constructed lazily on the first step. A full reset discards the
adapter and rebuilds it lazily from authoritative solver state; it does not
reuse the previous engine instance.

## Genesis Rigid numerical source map

Primary numerical code:

- `genesis/engine/solvers/rigid/abd/`
  - forward kinematics;
  - articulated mass and smooth dynamics;
  - force accumulation;
  - damping, integration, and retraction.
- `genesis/engine/solvers/rigid/constraint/`
  - native row assembly;
  - active-set and friction-cone classification;
  - gradient and cost;
  - Hessian assembly;
  - Cholesky;
  - native line search;
  - island organization.
- `genesis/engine/solvers/rigid/collider/`
  - broad phase;
  - narrow phase;
  - contact generation and pruning.
- `genesis/utils/array_class.py`
  - `DynState`;
  - `DynInfo`;
  - `RigidInfo`;
  - `ConstraintState`;
  - collider state/info;
  - static Rigid configuration.

The current pure-Rigid numerical order is:

```text
kernel_step_1
  -> FK / velocity refresh when stale
  -> mass matrix / smooth dynamics
-> equality-row assembly
-> native collision detection
-> inequality-row assembly
-> native constraint solve
-> kernel_step_2
  -> constrained acceleration
  -> damping / integration / commit
  -> FK / velocity refresh
```

Scene and Simulator provide the public lifecycle shell. `NewtonCoupler` bridges
that shell to `SimEngine`; viewer, recorder, sensor, clock, and public entity
APIs continue to observe solver state through their existing interfaces.

## Graph-native implementation contract

The first executable solver is one `@qd.kernel(graph=True)` timestep pipeline.

Required device-resident control state includes:

- Newton condition and iteration count;
- PCG condition and iteration count;
- line-search condition and iteration count;
- line-search alpha;
- current and trial energy;
- Newton convergence data;
- PCG vectors and dot products.

The graph contains nested device loops:

```text
Newton graph_do_while
|-- preconditioner factorization
|-- PCG graph_do_while
`-- binary line-search graph_do_while
```

There is no eager reference implementation of the new solver. Tests may call individual numerical functions, but those
functions must be the same functions used by the graph path.

The completed Rigid and cloth-elasticity milestones used no `qd.checkpoint`.
The contact milestone uses `qd.checkpoint` for pair, contact-assembly, and
global-triplet capacity growth. Live counts remain device scalars; ordinary
frames remain one graph launch.

## Candidate-row and active-set lifecycle

The current Genesis behavior is the reference:

- geometric collision and candidate-row topology are built once per timestep;
- `M`, `J`, `D`, and `a_ref` remain fixed during the native Newton solve;
- unilateral active state, friction branch, and elliptic-cone zone may change at accepted Newton iterates;
- line-search candidates re-evaluate piecewise costs without rebuilding candidate topology.

This section records current behavior. If the new framework changes any part of this lifecycle, mark it **Open** and
obtain user approval first.

## Rigid gradient and operator

The Rigid unknown remains the Genesis minimal-coordinate trial acceleration `a_R`.

The exact native gradient includes inertia and every native constraint cost.

Let `p_R` be the Rigid slice of the global PCG vector. The matrix-free operator is:

$$
H_Rp_R
=
h^4
\left(
M_Rp_R
+J_s^TD_{\mathcal A}J_sp_R
+\sum_eJ_e^TK_eJ_ep_R
\right).
$$

Definitions:

- `M_R`: generalized mass matrix;
- `J_s`: scalar native-row Jacobian;
- `D_A`: active scalar-row curvature matrix;
- `J_e`: stacked elliptic-contact row Jacobian;
- `K_e`: elliptic-cone row-space Hessian.

`constraint/backward.py::func_matvec_Ap` is a useful scalar-row reference but is not the final operator because it does
not include the complete elliptic-cone term and uses backward-specific buffers.

Required tests:

- random-vector matrix-free apply versus the assembled native Hessian;
- empty-constraint mass-only apply;
- scalar unilateral rows;
- pyramidal friction;
- elliptic friction in every cone zone;
- multiple environments and islands.

These are unit tests inside the complete vertical slice, not separate milestones.

## Native preconditioner

Each Newton iteration factors the current Genesis native Rigid matrix:

$$
P_R
=
L_RL_R^T.
$$

Every PCG preconditioner call computes:

$$
z_R
=
L_R^{-T}L_R^{-1}r_R.
$$

The factor is fixed during one PCG solve. Reuse existing dense/tiled/skyline factorization and solve functions rather
than implementing another direct solver.

The preconditioner contains Genesis native equality, joint-limit, native-contact, friction-loss, and elliptic-cone
terms. Future Rigid-Cloth coupling terms do not enter the phase-one Rigid preconditioner.

## Global PCG

The global PCG owns `r`, `z`, `p`, `Ap`, scalar reductions, counters, and convergence state.

Its operator call is:

$$
Ap
=
H_{\mathrm{BCOO}}p
+H_R^{\mathrm{mf}}p_R.
$$

It solves:

$$
Hp
=
-g.
$$

`g` is the current exact global gradient. The Rigid-only BCOO may have no live entries, but the BCOO plus matrix-free
composition is the permanent global contract.

The native Cholesky preconditioner is exact for the Rigid-only block, so the Rigid block should converge in one PCG
iteration up to roundoff.

## Newton and line search

Newton, PCG, and line search use nested `qd.graph.do_while`.

Binary line search starts with:

$$
\alpha
=
1.
$$

Let `a_R^k` be the current accepted acceleration and `p_R` the Newton direction:

$$
a_R(\alpha)
=
a_R^k+\alpha p_R.
$$

Accept only if the exact merit decreases:

$$
\Phi(a_R(\alpha))
\le
\Phi(a_R^k).
$$

Otherwise:

$$
\alpha
\leftarrow
\frac12\alpha.
$$

Do not use Genesis derivative-based bracketing or an Armijo condition. Reject the Newton step when the binary
backtracking budget or minimum alpha is exhausted.

Rejected trials do not mutate accepted acceleration, active-set state, factorization, warm start, configuration, or
derived geometry.

## Timestep finalization

After Newton convergence:

1. retain the accepted generalized acceleration;
2. compute native constraint force;
3. update generalized velocity;
4. apply joint-specific integration/retraction;
5. commit generalized configuration;
6. refresh FK, geometry, and velocity.

The new framework advances one timestep `h`. Genesis `substep` terminology is used only when referring to old source
function names.

## Reference implementation map

External references are optional and may be located anywhere. Use these file
suffixes to locate the corresponding concepts inside whichever pinned
reference checkout is available.

Python numerical reference:

- `_src/solver/sim_engine.py`
- `_src/solver/linear_pcg.py`
- `_src/solver/global_linear_system.py`
- `_src/solver/sim_system.py`

Native numerical reference:

- `_src/native/solver/sim_system.h`
- `_src/native/solver/global_linear_system.h`
- `_src/native/solver/sim_engine_pipeline.cu`
- `_src/native/solver/subgraph/linear_pcg.h`

Compiler documentation:

- `docs/source/user_guide/graph.md`
- `docs/source/user_guide/tensor.md`

## Checkpoint status

Quadrants issue #750
tracks checkpoint regions containing child graph loops. Genesis keeps PCG and
line-search `qd.graph.do_while` loops outside explicit checkpoint bodies:
capacity checks yield before those consumers. Direct kernel-AST suffix tasks
are skipped correctly, but a top-level inlined `qd.func` after a yielding
checkpoint currently escapes the implicit resume gate and executes once.
`SimEngine` therefore wraps PCG initialization, each PCG iteration, PCG
finalization, line-search trial work, and line-search post work in separate
flat no-yield checkpoints. This keeps child loops outside checkpoint bodies
without trusting the broken implicit `qd.func` gating tracked by Quadrants
issue #956. The
contact pipeline requires checkpoint/yield/resume support, but not the
nested-checkpoint graph shape implemented by Quadrants PR #944.

Every yielded launch must identify one reference phase. The host may only grow the
owner's buffers, clear the corresponding overflow scalar, and resume from that
phase. A fixed-capacity workaround or a capacity check after its consumer is
prohibited.

Temporary performance deferral (accepted 2026-10-02): the explicit flat gates
required by Quadrants #956 increase the stable Franka-cloth median by 5.8% and
wall time per PCG work by 5.2% versus the nested-checkpoint baseline. Correct
yield semantics take priority for this first version. Once #956 is fixed,
remove the redundant flat gates and repeat the matched benchmark before
claiming the recovered performance.

## Verification

Run format, lint, and syntax checks with tools from the active environment.
Then run the targeted GPU suite:

```text
python -m pytest \
  tests/deformable/test_contact_system.py \
  tests/deformable/test_qcloth_newton.py \
  tests/deformable/test_rigid_contact_proxy_kkt.py \
  tests/rigid/test_dynamics.py \
  --backend gpu -n 0
```

Run the public examples through the repository's example harness:

```text
python -m pytest tests/test_examples.py -m examples \
  -k "cloth_stack or franka_cloth_grasp" -n 0
```

Performance reports must state the exact compiler commit, GPU, driver,
precision, timestep, warmup window, measurement window, Newton/PCG work, and
whether offline cache was enabled. Do not encode one developer's environment
or latest measurements as a permanent requirement in this guide.

## Handoff checklist

Before reporting completion:

- summarize changed files;
- report exact tests and backends;
- state whether CUDA graph was actually built or fell back;
- report Newton, PCG, and line-search iteration counts;
- report any behavior that differs from the native baseline;
- list every unresolved **Open** decision without choosing it.
