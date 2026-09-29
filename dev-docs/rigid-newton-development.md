# Rigid Newton Framework Development Guide

This document is the implementation handoff for agents working in `Genesis-RigidOnly`.

Read [rigid-newton-roadmap.md](rigid-newton-roadmap.md) first. The roadmap owns requirements and milestone order.
This guide records the current workspace, relevant source code, development procedure, and verification commands.
Read [qipc-simulation-system-design.md](qipc-simulation-system-design.md) for the SimSystem, SimEngine, lifecycle,
data-scope, and Manager + Reporter contracts.

## Repository state

- Worktree: `C:\Users\81946\Projects\GenesisWorldCouplingSystem\Genesis-RigidOnly`
- Branch: `newton/rigid-only-baseline`
- Genesis base: `09a8b65ab353369a2df778c20eaedee1c9b8a47c`
- Quadrants dependency: `1.3.1`
- Numerical precision: `double`

Reference checkouts beside this worktree:

- `..\quadrants`: local Quadrants checkpoint implementation rooted at
  `459a3eb57a36781e3abcd6272232c074a98f87a7`
- `..\qipc`: QIPC Quadrants migration, commit `b17773cf`
- `..\cuda-graph-qipc`: CGQ ground truth, commit
  `42e7d4cbbad08739107ad830a17918f5f0f209ff`

The QIPC and CGQ repositories are read-only references for this project. Do not add either as a dependency and do not
develop the new framework inside them.

## Agent collaboration protocol

1. Work only in `Genesis-RigidOnly`. Keep the original `Genesis` checkout unchanged.
2. Read both Rigid Newton documents before changing code.
3. Check `git status --short` before editing. Preserve user changes.
4. Do not decide an item marked **Open** in the roadmap.
5. Do not use the existing Legacy/SAP/IPC coupler lifecycle as an integration surface.
6. Do not create an eager or host-loop implementation.
7. Use checkpoint only for CGQ contact-capacity overflow and exact phase resume.
8. Do not create a participant-private PCG.
9. Do not copy numerical formulas into a second implementation. Refactor shared `qd.func` entry points instead.
10. Run the relevant RTX 5090 tests before handoff; CPU execution must be
    rejected.
11. Reject every non-load-balanced scene-scale GPU implementation. This is a
    hard code-admission rule, not an optimization suggestion.
12. Traversal, compaction, reduction, segmented reduction, candidate emission,
    sparse assembly, and other irregular work must use the most efficient
    applicable warp/subgroup-level algorithm. Reject scalar or thread-local
    code when CGQ provides a warp-frontier, warp-DFS, warp-batched, or
    warp-segmented production path.
13. Do not admit one-task whole-scene traversal, unbounded per-lane work,
    per-item global reservation when warp batching applies, or scalar global
    reduction when warp/block reduction applies.
14. A simpler diagnostic implementation may exist only in isolated tests or
    offline comparison tools. It must not be registered by a builder or be
    reachable from a production or milestone runtime.
15. Numerical parity, a passing example, or a temporary-performance-debt entry
    never waives rules 11--14. Production work must match or exceed the
    highest-performance applicable CGQ path.

## Local environment

The worktree has a local `.venv` using Python 3.13.

Required versions:

```text
Quadrants local checkpoint build / 459a3eb57
PyTorch 2.11.0+cu128
CUDA 12.8
RTX 5090 / SM 12.0
```

The default PyPI PyTorch wheel is CPU-only. Install the CUDA wheel explicitly:

```powershell
uv pip install --python .venv\Scripts\python.exe `
  --index-url https://download.pytorch.org/whl/cu128 `
  --reinstall torch==2.11.0
```

Verify:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

Set UTF-8 for test runs so Genesis logging renders correctly:

```powershell
$env:PYTHONUTF8='1'
```

## Current changes in the worktree

`genesis/engine/solvers/rigid/rigid_solver.py`

- removes Legacy/SAP/IPC type checks from the native numerical path;
- removes SAP split-step and IPC deferred-step branches;
- defines `step_rigid_core`, a direct native numerical entry;
- keeps `RigidSolver.substep` as a wrapper around `step_rigid_core`.

`genesis/engine/solvers/rigid/collider/collider.py`

- removes IPC-delegated collision-pair filtering from the native collider.

`tests/rigid/test_dynamics.py`

- compares `scene.step()` and direct `step_rigid_core` execution from the same initial state.

`pyproject.toml`

- pins `quadrants==1.3.1`.

The direct entry still uses a built `RigidSolver` as the container for typed state. It is a parity boundary for
mechanical extraction, not the final framework API.

The independent graph runtime is implemented under `genesis/engine/systems`:

- `sim_system.py`: system registration and dependency lookup;
- `sim_engine.py`: direct build-time lowering of registered timestep lifecycle functions;
- `global_linear_system.py`: global vectors and symmetric BCOO;
- `linear_pcg.py`: composable device-resident PCG and direct static lowering of operator/preconditioner contributions;
- `rigid_system.py`: Genesis Rigid numerical participant;
- `rigid_contact_proxy.py`: massless proxy SE(3) state and reduced-KKT workspace;
- `rigid_contact_proxy_kkt.py`: shared `A^-1`, `P`, `P^T`, slack, and FK-defect algebra;
- `rigid_joint_forest.py`: Genesis minimal-coordinate link expansion/restriction and reduced operator hooks;
- `rigid_contact_assemble.py`: proxy-proxy and proxy-FEM physical contact routes;
- `sim_engine.py`: the graph-native timestep driver and static pipeline.

Read [reduced-kkt-migration.md](reduced-kkt-migration.md) before changing any
of the proxy, forest, contact-route, PCG, CCD, globalization, or commit-gate
code. These files are under active migration and are not selected by
`build_scene_engine` until the complete pinned-CGQ production path is wired.

The existing Rigid kernels expose shared `qd.func` stages for the graph path while retaining their native kernel
wrappers. The dependency remains one-way from the new framework to the Rigid numerical core.

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

Scene, Simulator, couplers, viewer, recorder, checkpointing, and public entity accessors are not the new runtime
architecture.

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

Use the references for concepts and tests, not as dependencies.

QIPC Quadrants migration:

- `..\qipc\qipc\_src\solver\sim_engine.py`
- `..\qipc\qipc\_src\solver\linear_pcg.py`
- `..\qipc\qipc\_src\solver\global_linear_system.py`
- `..\qipc\qipc\_src\solver\sim_system.py`

CGQ:

- `..\cuda-graph-qipc\qipc\_src\native\solver\sim_system.h`
- `..\cuda-graph-qipc\qipc\_src\native\solver\global_linear_system.h`
- `..\cuda-graph-qipc\qipc\_src\native\solver\sim_engine_pipeline.cu`
- `..\cuda-graph-qipc\qipc\_src\native\solver\subgraph\linear_pcg.h`

Quadrants:

- `..\quadrants\docs\source\user_guide\graph.md`
- `..\quadrants\docs\source\user_guide\tensor.md`

## Checkpoint status

[Quadrants #750](https://github.com/Genesis-Embodied-AI/quadrants/issues/750)
tracked checkpoint regions containing child graph loops. The pinned local
Quadrants build supports that structure and is required for the contact
pipeline.

Every yielded launch must identify one CGQ phase. The host may only grow the
owner's buffers, clear the corresponding overflow scalar, and resume from that
phase. A flat-checkpoint or fixed-capacity workaround is prohibited.

## Verification

Run lint and syntax checks:

```powershell
uvx ruff check genesis\engine\solvers\rigid\rigid_solver.py `
  genesis\engine\solvers\rigid\collider\collider.py `
  tests\rigid\test_dynamics.py

uvx ruff format --check genesis\engine\solvers\rigid\rigid_solver.py `
  genesis\engine\solvers\rigid\collider\collider.py `
  tests\rigid\test_dynamics.py
```

Run direct-core parity on CPU:

```powershell
$env:PYTHONUTF8='1'
.venv\Scripts\python.exe -m pytest `
  tests/rigid/test_dynamics.py::test_rigid_core_step_matches_scene_step `
  -n 0 -s --backend cpu --dev --logical -p no:cacheprovider
```

Run direct-core parity and native contact on RTX 5090:

```powershell
$env:PYTHONUTF8='1'
.venv\Scripts\python.exe -m pytest `
  tests/rigid/test_dynamics.py::test_rigid_core_step_matches_scene_step `
  tests/rigid/test_collision.py::test_contact_forces `
  -n 0 --backend gpu --dev --logical -p no:cacheprovider
```

Current verified results:

- CPU direct-core parity: passed.
- CPU gravity, all-fixed, and plane-convex contact: passed.
- RTX 5090 direct-core parity: passed.
- RTX 5090 native contact force: passed.
- CPU and RTX 5090 graph-native sphere-plane contact: passed in double precision.
- CPU and RTX 5090 multi-participant solve: Rigid plus an independently registered quadratic physics system converged
  in the same PCG and shared line search.
- RTX 5090 elliptic contact with sliding, torsional, and rolling friction: passed; the last timestep used 11 Newton
  iterations and one PCG iteration per Newton solve.
- The RTX 5090 step used a CUDA graph with 118 nodes and no non-graph fallback.
- Rigid-only contact converged with one PCG iteration using the exact native Cholesky preconditioner.
- `examples/newton_coupling/franka_cube.py --runtime newton` completed hold, grasp, and lift on CPU and RTX 5090. The
  cube reached `z=0.1800`, matching the native example result to the displayed precision.
- On the RTX 5090 Franka example, synchronized steady-state timestep latency after multi-participant generalization
  measured 3.148 ms, versus 3.062 ms immediately before the generalization and 2.949 ms for the native path. Before
  optimization the graph path was 3.987 ms. Nsight Systems GPU-kernel time before generalization fell from 2.482 to
  2.003 ms/timestep; native is 1.836 ms/timestep.
- The optimization restores the native GPU cooperative contact-pruning gate and factors the accepted Hessian only
  when another Newton iteration will consume it. The accepted-state factor cost fell from 0.534 to 0.035 ms/timestep.
- The post-generalization initial graph-building step measured 99.67 s, versus 91.07 s immediately before
  generalization and 1.24 s for the already-compiled native path. Compilation remains the largest unresolved first-run
  cost.

## Handoff checklist

Before reporting completion:

- summarize changed files;
- report exact tests and backends;
- state whether CUDA graph was actually built or fell back;
- report Newton, PCG, and line-search iteration counts;
- report any behavior that differs from the native baseline;
- list every unresolved **Open** decision without choosing it.
