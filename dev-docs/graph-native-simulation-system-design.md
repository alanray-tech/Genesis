# Graph-Native Simulation System Design

Status: architecture contract for the current implementation.

This document contains architecture contracts and real source anchors only. Human-oriented examples live in the
[Notion article](https://app.notion.com/p/3e90c06194e18165b859dbbaaa64ed56).

## 1. Components

### SimSystem

A numerical `SimSystem` is a `@qd.data_oriented` data owner. It owns:

- device buffers;
- runtime scalars;
- static configuration;
- stable data accessors;
- named lifecycle functions implemented by that system.

The minimal host base may remain a plain abstract class when `@qd.data_oriented` must be applied to concrete classes.

#### Concrete data-oriented contract

Every concrete numerical system that participates in a GPU pipeline is decorated with `@qd.data_oriented`.

```python
@qd.data_oriented
class ClothSystem(SimSystem):
    def __init__(self, n_vertices: int):
        super().__init__()
        self.n_vertices = n_vertices
        self.positions = qd.ndarray(qd.f64, shape=(n_vertices, 3))
        self.predicted_positions = qd.ndarray(qd.f64, shape=(n_vertices, 3))

    def do_build(self) -> None:
        self.linear_system = self.require(GlobalLinearSystem)

    @qd.func(requires_top_level=True)
    def predict(self):
        for i_v in range(self.n_vertices):
            self.predicted_positions[i_v] = self.positions[i_v]
```

The object is both the host-side resource owner and the compile-time kernel argument tree. Its kernel-visible
attributes may contain:

- `qd.ndarray`, `qd.field`, `qd.Tensor`, and nested typed state;
- references to other stable `@qd.data_oriented` systems resolved during build;
- primitive values intentionally used as compile-time configuration;
- build-finalized lists of bound `@qd.func` functions used for static dispatch.

Host-only build data may live directly on a data-oriented object. Quadrants prunes members that compiled kernels do not
read, and explicitly supports NumPy arrays in fastcache hashing by dtype and rank. Device functions must still access
only values that Quadrants can lower.

The Python implementation uses no module-level object-id maps. `SimEngine` owns its registry directly during build;
after dependency resolution and static function binding it clears system back-references and releases the registry
before graph compilation. One-shot host arrays are uploaded directly by their owner.

#### GPU identity, pointer, and specialization rules

A data-oriented system instance is stable from graph compilation until graph invalidation:

- object identity is part of specialization;
- primitive attributes are compile-time values unless represented by device scalars;
- buffer addresses captured by a graph remain valid for that graph instance;
- lifecycle function lists are immutable after static lowering;
- adding or removing a system requires rebuilding `SimEngine`'s static function lists and graph.

Dynamic values that change every timestep use device-resident arrays. Examples include live triplet count, Newton and
PCG conditions, iteration counters, line-search alpha, energy, and convergence state. A changing value must not be a
Python primitive when changing it should avoid recompilation.

Capacity growth rebinds the owner system's buffer explicitly. The contact milestone uses the approved Quadrants
checkpoint protocol: the graph yields at the owning phase, the host grows that owner's buffers, and execution resumes
from the exact checkpoint with the rebound ndarrays. Normal frames do not leave the graph.

### SimEngine registry

`SimEngine` stores systems in a host-side `dict[type, SimSystem]` and exposes
the registry through explicit methods:

```python
class SimEngine:
    def add(self, system: SimSystem) -> None: ...
    def find(self, system_type: type[T]) -> T | None: ...
    def require(self, system_type: type[T]) -> T: ...
    def build_systems(self) -> None: ...
```

Required behavior:

- key systems by concrete type;
- reject duplicate concrete types;
- forbid additions after initialization starts;
- exclude invalid systems from lookup;
- perform no runtime lookup inside a graph kernel.

### SimEngine

`SimEngine` owns:

- the typed system dictionary;
- timestep-wide runtime state;
- lifecycle collection;
- static function lowering;
- the graph-mode timestep kernel;
- Newton, PCG, line-search, failure, and diagnostic state required by its algorithm.

In GraphMode, `SimEngine` constructs and launches `@qd.kernel(graph=True)`.

### Manager

A Manager owns one shared resource or index space:

- total extent;
- global offsets;
- allocation;
- capacity policy;
- indexing invariants;
- stable scoped accessors.

## 2. Dependency API

`SimSystem` forwards typed lookup to its engine collection:

```python
class SimSystem:
    @property
    def engine(self) -> SimEngine: ...

    def find(self, system_type: type[T]) -> T | None:
        return self.engine.find(system_type)

    def require(self, system_type: type[T]) -> T:
        return self.engine.require(system_type)

    def do_build(self) -> None:
        pass
```

Rules:

- use `require(T)` for mandatory dependencies;
- use `find(T)` for optional dependencies;
- resolve dependencies during `do_build()`;
- cache the returned typed system references;
- do not retrieve systems by string name;
- do not access the collection from device code.

## 3. Public integration boundary

`NewtonCouplerOptions` selects the runtime through the ordinary Scene API.
`Simulator` owns `NewtonCoupler`, which validates the first-version support
matrix and constructs `SimEngine` lazily on the first `Scene.step()`. This
allows post-build qpos, controller, and fixed-vertex setup before staging.

`build_scene_engine` and the systems package are internal composition and
testing surfaces. Users do not call `engine.step()` directly. Rigid state stays
owned by the existing `RigidSolver`; `RigidSystem` only references its buffers
and numerical functions.

## 4. Basic lifecycle

The lifecycle shared by every system is:

```text
construct
-> SimEngine.add_system and bind engine
-> build_systems / do_build dependency resolution
-> wire_data host ingress
-> init global offsets and buffers
-> compile/capture graph pipeline
-> Scene.step / NewtonCoupler.step / SimEngine.step
-> reset by discarding and lazily rebuilding the adapter
```

`do_build()` has one responsibility: establish `require/find` dependencies. Allocation, reporting, numerical phases,
diagnostics, and event behavior belong to their own lifecycle functions or actions.

The implementation uses the lifecycle names `build_systems()`, `do_build()`,
`wire_data()`, and `init()`.

## 5. Additional lifecycle

Additional lifecycle is provided by named method contracts implemented by the
relevant `SimSystem` objects and called explicitly by `SimEngine`.

There is no second system category for numerical participants. Every framework component is a `SimSystem`, and
`SimEngine` selects a system only for lifecycle stages whose named functions that system implements. Deriving from
`SimSystem` provides ownership, registration, and dependency lookup; it does not imply the complete Newton interface.

The Newton engine currently needs named operations including:

```text
predict
report_extent
assemble
energy
negate_dq / negate_dx
contribute_newton_max_disp
record_start_point
step_forward
update_velocity
copy_x_prev
```

`SimEngine` explicitly owns the registered systems and calls them in pipeline order. `FiniteElementMethod` owns
`FEMBDF1` plus its registered `FEMConstitution` systems. `StandardPCGSolver` owns `LinearPCG`. There is no
method-name discovery, bound-function-list lowering, virtual call, registry traversal, or dynamic function pointer
inside the graph.

## 6. GraphMode scheduling

The timestep pipeline is owned by `SimEngine`.

Required properties:

- the first executable implementation is graph-native;
- Newton, PCG, and line search use `qd.graph.do_while`;
- control flags and counters are device-resident;
- system/function selection completes before graph compilation;
- no eager or host-loop solver path is introduced;
- the contact milestone uses checkpoints only for capacity overflow and exact phase replay;

Current source anchors:

- `genesis/engine/systems/sim_engine.py`
- `genesis/engine/systems/standard_pcg_solver.py`
- `genesis/engine/systems/linear_pcg.py`
- `genesis/engine/systems/global_linear_system.py`
- `genesis/engine/systems/finite_element/finite_element_method.py`

The runtime is GPU-only and rejects CPU before graph construction. These files
are the authoritative implementation anchors for this contract.

### Host lowering sequence

The host completes all composition before the first timestep:

```text
add concrete systems
-> bind engine references
-> resolve require/find dependencies
-> wire concrete data and finalize global Manager ranges
-> allocate global buffers
-> compile/capture the graph kernel
```

The graph receives concrete system objects, finalized offsets, fixed capacities, and device-resident live extents.
Python dependency lookup and host range assignment never occur inside the graph.

### GPU timestep call sequence

The current Newton pipeline lowers to the following GPU phase order:

```text
RigidSystem.predict()
FiniteElementMethod.predict()

Newton graph_do_while:
    each emitter.report_extent()
    GlobalLinearSystem.derive_extents()
    GlobalLinearSystem.zero_rhs()
    GlobalLinearSystem.zero_triplet()
    RigidSystem.assemble()
    FiniteElementMethod.assemble()
    GlobalLinearSystem.body_sort_reduce()
    GlobalLinearSystem.traverse()
    StandardPCGSolver.solve()
    RigidSystem.negate_dq()
    FiniteElementMethod.negate_dx()
    contribute_newton_max_disp()
    record_start_point()
    evaluate baseline energy

    line-search graph_do_while:
        step_forward()
        evaluate trial energy
        accept or halve alpha

    update convergence and failure state

RigidSystem.step_forward()
FiniteElementMethod.update_velocity()
FiniteElementMethod.copy_x_prev()
```

The contact specialization inserts the reference contact phases without changing the
participant lifecycle:

```text
frame CP0:
    friction_snapshot
    predict and forward_global_vertices

Newton graph_do_while:
    COUNT: count active contact; yield on assembly-demand overflow
    FILTER: filter/assemble contact; yield on padding overflow
    SORT: sort/reduce, classify, derive extents; yield on triplet overflow
    SOLVE: zero and assemble Rigid/FEM/contact terms; build preconditioners
    PCG_INIT
    PCG graph_do_while:
        PCG_ITERATION
    PCG_FINISH
    SOLVE_POST: directions, convergence, baseline energy, BVH build
    QUERY: trajectory query; yield on pair overflow
    CCD
    LINE_SEARCH: initialize alpha
    line-search graph_do_while:
        LINE_SEARCH_TRIAL
    LINE_SEARCH_POST: accept state and update Newton condition
```

The PCG and line-search child loops remain outside explicit checkpoint bodies.
Their inlined stages use flat no-yield checkpoints because of the compiler
gating defect tracked by Quadrants issue #956.

The precise numerical parameters are fixed by
[contact-parameter-manifest.md](contact-parameter-manifest.md).

`StandardPCGSolver` owns `LinearPCG`; the engine explicitly supplies the Rigid matrix-free contribution and FEM
preconditioner. Every participating range must write its preconditioned residual.

### Top-level GPU phase rule

A lifecycle function containing top-level range loops, reductions, block synchronization, or multi-pass algorithms is
decorated with `@qd.func(requires_top_level=True)`. Such a function may be called:

- directly at the top level of a `@qd.kernel`;
- directly in a `qd.graph.do_while` body;
- through the explicit `SimEngine`, `FiniteElementMethod`, or `StandardPCGSolver` composition at one of those locations.

It must not be called inside a runtime `if`, runtime `for`, or ordinary `while`. Doing so demotes its phase loops below
top level and invalidates grid-wide phase ordering. Runtime gating belongs inside the generated tasks through device
flags, or in a graph conditional loop.

Independent phases may use `qd.graph.parallel_context()` only when their write sets are disjoint or their shared writes
use the required atomic reduction. Sequential ordering remains the default.

### GPU parallel-quality gate

This is a code-admission rule, not an optimization suggestion.

Production and milestone runtime paths must be load-balanced. Traversal,
compaction, reduction, segmented reduction, candidate emission, sparse
assembly, and other irregular scene-scale work must use the most efficient
applicable warp/subgroup-level organization. Code that leaves substantial GPU
lanes idle, assigns an unbounded workload to one lane, or uses per-item global
atomics where warp batching is applicable is forbidden.

Faithful migration does not require byte-for-byte reproduction of a reference
launch topology. A different implementation is acceptable only when it is
load-balanced and evidence shows that its warp utilization, atomic traffic,
memory traffic, occupancy, and scaling match or exceed the applicable pinned
native-reference path. Migration status is not a waiver for a merely
"reasonable" decomposition.

A simpler implementation may establish correctness only in an isolated test or
offline oracle. It cannot be connected to the builder, example, or runtime and
cannot satisfy any milestone gate.

Required properties:

- work that scales with vertices, primitives, BVH nodes, candidate pairs, or
  contact blocks is distributed across GPU threads, warps, or blocks;
- irregular traversal uses a frontier, work queue, batched query, or another
  bounded-work decomposition that prevents one lane from owning the whole scene;
- global atomics are amortized or shown by profiling not to serialize the
  workload;
- no host traversal, CPU fallback, or per-frame synchronization substitutes for
  missing GPU scheduling;
- asymptotic work and memory growth must not regress from the referenced
  production
  algorithm without prior approval;
- profiling must demonstrate scaling and useful GPU occupancy on representative
  cloth scenes before the implementation is accepted.
- every scheduling difference from the pinned reference must be recorded
  together with evidence that it meets or exceeds the applicable production
  path.

The following are explicitly forbidden from any builder-selected or runtime
production path:

- one serialized `for range(1)` task performing a complete BVH or contact
  traversal;
- a single global DFS stack for all EE node pairs;
- thread-per-query irregular traversal when a warp-frontier or warp-local DFS
  is the applicable production algorithm;
- pair-by-pair global output reservation when warp ballot/prefix compaction can
  reserve one batch;
- scalar global reductions when warp/block partial reductions are applicable;
- brute-force all-pairs contact used as a production broad phase;
- scene-specific capacities, thresholds, or branches added only to make the
  milestone example pass;
- retaining a reduced implementation while naming it after a more capable
  backend.

Temporary diagnostic code may use a simpler algorithm only in isolated tests or
offline comparison tools. It must not be registered as a `SimSystem`, selected
by `builders.py`, or reachable from the milestone runtime. The former
single-task `LBVH.query_ee_dual` has been removed. The runtime now uses
level-synchronous subgroup-compacted frontiers, a persistent warp-local DFS
work queue, and warp-batched pair emission; parity, overflow replay, scaling,
and profiling remain mandatory regression gates. Frontier, DFS, and warp
rollback launches are occupancy-sized static grids that manually grid-stride
device-live counts; scene size never enters the compiled shape. The static
thread/block extents describe launch topology only and are not simulation
sizes: frontier, task, edge, and pair counts remain zero-dimensional device
scalars read with `[()]`.

### Device-side loop control

Newton, PCG, and line search use scalar zero-dimensional `qd.i32` ndarrays as graph conditions. They have do-while
semantics and therefore execute the body once before checking the condition. The body must seed and clear its condition
explicitly.

On CUDA devices with native conditional-node support, the nested loops remain on the device. Other supported backends
may use the Quadrants runtime fallback while preserving one Python kernel invocation. Backend performance is not
assumed equivalent.

Scalar state changes use serialized one-iteration loops where required:

```python
for _ in range(1):
    self.alpha[0] = qd.f64(1.0)
    self.condition[()] = 1
```

Each serialized region can become a graph node. Related scalar updates should be grouped to avoid creating unnecessary
tiny kernels.

### Graph rebuild boundary

The following changes require a new static lowering or graph build:

- system addition, removal, or invalidation;
- lifecycle function-list changes;
- compile-time option changes;
- global topology or DOF layout changes;
- buffer reallocation outside an active checkpoint/resume protocol.

The following values remain runtime data and do not require graph rebuilding:

- current state and trial state;
- live counts within allocated capacity;
- active constraints and contacts;
- Newton, PCG, and line-search conditions;
- convergence metrics and diagnostic counters.

Contact pair, assembly, friction, and global-triplet buffers may grow at an
overflow checkpoint. The owner replaces the ndarray, clears only its overflow
flag, and resumes from the checkpoint whose graph launch context reads the new
pointer.

## 7. Data scope

Every system is the scope of the data it owns.

The owner defines:

- allocation and lifetime;
- units and shape;
- indexing and ranges;
- mutation policy;
- public host accessors;
- graph functions that read or write the data.

Cross-system access must follow:

```text
typed require/find
-> system-owned accessor
-> stable view or handle
```

Callers must not:

- read another system's private fields through `SimEngine`;
- duplicate another system's authoritative buffer;
- infer offsets owned by a Manager;
- depend on whether storage currently uses `qd.field`, `qd.ndarray`, `qd.Tensor`, or an external buffer.

## 8. Global range ownership

There is no generic `Reporter` registry. Concrete systems expose
domain-specific `report_*_extent()` and `receive_*_range()` methods, and the
composition root calls them explicitly before initialization.

Build semantics:

```text
offset = 0
for each participating system:
    count = system.report_extent()
    system.receive_range(offset, count)
    offset += count
allocate shared resource for total offset
```

This pattern applies to:

- global DOF ranges;
- vertex and surface ranges;
- BCOO triplet extents;
- static connectivity;
- contact capacities;
- diagnostic channels.

Range assignment is host/build-time work. Device-side `report_extent` methods
used for dynamic BCOO demand are a separate numerical phase. Graph kernels
consume finalized static offsets and device-resident live counts.

## 9. Extension properties

This organization supports extension because:

- a new system resolves typed dependencies instead of editing existing systems;
- function-name dispatch makes lifecycle calls visible;
- global managers receive ranges from explicit composition code;
- data ownership remains explicit;
- missing mandatory dependencies fail during build;
- selected functions are lowered before graph compilation;
- no runtime provider switch or generic mega-kernel is required;
- systems remain independently testable through their real APIs and GPU kernels.

## 10. Current source mapping

Optional Python-reference file suffixes:

- `_src/solver/sim_system.py`
- `_src/solver/sim_engine.py`
- `_src/solver/global_vertex_manager.py`
- `_src/solver/global_surface_manager.py`
- `_src/solver/global_linear_system.py`
- `_src/solver/linear_pcg.py`

Current implementation:

- `genesis/engine/systems/sim_system.py`
- `genesis/engine/systems/sim_engine.py`
- `genesis/engine/systems/global_linear_system.py`
- `genesis/engine/systems/global_body_manager.py`
- `genesis/engine/systems/global_vertex_manager.py`
- `genesis/engine/systems/global_surface_manager.py`
- `genesis/engine/systems/contact_system.py`
- `genesis/engine/systems/contact_constitution.py`
- `genesis/engine/systems/consistent_ipc_contact.py`
- `genesis/engine/systems/broad_phase_system.py`
- `genesis/engine/systems/lbvh_broad_phase.py`
- `genesis/engine/systems/standard_pcg_solver.py`
- `genesis/engine/systems/linear_pcg.py`
- `genesis/engine/systems/rigid_system.py`
- `genesis/engine/systems/finite_element/finite_element.py`
- `genesis/engine/systems/finite_element/finite_element_method.py`
- `genesis/engine/systems/finite_element/fem_bdf1.py`
- `genesis/engine/systems/finite_element/strain_limit_baraff_witkin_shell_2d.py`
- `genesis/engine/systems/finite_element/quadratic_bending.py`

Optional native-reference file suffixes:

- `_src/native/solver/sim_system.h`
- `_src/native/solver/sim_engine.h`
- `_src/native/solver/sim_engine_pipeline.cu`

External reference implementations are not dependencies or development bases,
and no checkout location is assumed.
