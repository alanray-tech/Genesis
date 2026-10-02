# Static Graph Composition Under a Strict No-Data-Oriented Rule

Status: architecture discussion.

## Decision statement

A strict prohibition on `@qd.data_oriented` is workable for a small, fixed
solver whose complete GPU call sequence can be written explicitly. It does not
preserve all of the requirements of the general multi-physics framework:

1. independently owned and extensible `SimSystem` implementations;
2. graph-level static dispatch with direct, inlineable `@qd.func` calls;
3. no central list of every system's data;
4. no repeated dispatch branch in every lifecycle phase;
5. adding a system without editing every existing pipeline phase;
6. no reflection, generated parameter pack, or compiler-managed object
   flattening.

At least one of these requirements must be relaxed. Any mechanism that
automatically discovers a system's data, binds its functions, generates kernel
arguments, and specializes the graph is semantically the same mechanism as
`data_oriented`, even if it uses another name.

This is not an argument against separating data storage from algorithms.
Storage-only data classes are useful. The issue is whether the compiler is
allowed to understand the association between one data object and its
statically selected functions.

## Verified CGQ scale

The current `cuda-graph-qipc` checkout contains several defensible counts:

- **44** registered frontend `Constitution` UID types in
  `qipc/constitution.py`; **43** remain after excluding the explicit
  test-only `BrokenTwinProbe`.
- **42** compiled concrete native constitution subclasses; **41** remain after
  excluding that test implementation.
- The native count comprises 23 FEM backends, 2 ABD backends, 8 inter-body
  joint backends, 4 contact backends, 2 extra-contact backends, and 3 B-spline
  backends.
- **13** additional plasticity or viscoelasticity modifier classes select or
  augment those backends without owning separate Constitution UIDs.

Therefore, "about 60 constitutive forms" is a reasonable description of the
user-facing architecture scale: 44 registered forms plus 13 modifiers gives
57. It must not be misreported as 60 independent compiled native
constitutions. The more conservative production-native count of 41 is already
large enough to expose the composition problem.

The inventory is grounded in:

- `cuda-graph-qipc/qipc/constitution.py`;
- the solver source list in `cuda-graph-qipc/CMakeLists.txt`;
- concrete subclasses under
  `cuda-graph-qipc/qipc/_src/native/solver/`;
- registrations under
  `cuda-graph-qipc/qipc/_src/native/solver/pybind/`.

Topology events, infrastructure systems, external-force helpers, and toy test
stubs are not counted as production constitutions. Presets, markers, MPM
frontends, and modifiers explain why frontend and native counts are not
one-to-one.

## What explicit Data plus functions provides

The proposed separation is valid for one system:

```python
@dataclass(frozen=True)
class RigidData:
    q: qd.types.NDArray[qd.f64, 2]
    dq: qd.types.NDArray[qd.f64, 2]


@qd.func
def rigid_predict(data: RigidData):
    ...


class RigidSolver:
    def __init__(self, data: RigidData):
        self.data = data
        self.predict_func = rigid_predict
```

The host class can expose a convenient object-oriented API. The data class can
remain a field-only container. A concrete graph can call the function with the
data explicitly:

```python
@qd.kernel(graph=True)
def rigid_step(rigid: RigidData):
    rigid_predict(rigid)
```

This scales while there is one known data type and one known call sequence.
It is the reason the pattern is adequate for a relatively small, fixed
RigidSolver.

## Failure at constitution scale

A generic FEM/contact pipeline has many lifecycle phases: extent reporting,
prediction, assembly, energy, Hessian action, preconditioning, line search,
accepted-step update, diagnostics, and subsystem-specific contact or topology
phases.

Without compiler-visible system objects, every phase must enumerate every
possible data/function pair:

```python
@qd.func(requires_top_level=True)
def predict_pipeline(...):
    if qd.static(has_constitution_a):
        predict_a(constitution_a_data)
    if qd.static(has_constitution_b):
        predict_b(constitution_b_data)
    # Repeated for every supported constitution.


@qd.func(requires_top_level=True)
def assemble_pipeline(...):
    if qd.static(has_constitution_a):
        assemble_a(constitution_a_data, linear_system_data)
    if qd.static(has_constitution_b):
        assemble_b(constitution_b_data, linear_system_data)
    # The same registry is handwritten again.
```

The pattern is repeated for each lifecycle phase. With 44 registered
constitution forms and only six common phases, the dispatch surface is already
up to 264 constitution/phase entries. Using the broader 57-form API surface
gives 342. Not every constitution implements every phase, but optional phases
increase the number of distinct matrices and do not remove the central
maintenance problem.

This design has concrete consequences:

- adding one constitution requires edits outside its own module;
- lifecycle omissions compile successfully and fail as missing physics;
- data and function registries can drift independently;
- each phase becomes a central knowledge owner for every constitution;
- tests must cover the Cartesian product of phase and constitution dispatch;
- graph and fastcache specialization keys must duplicate the same registry;
- merge conflicts increase as independent subsystem authors edit shared
  pipeline files.

## Why a central aggregate is not a solution

The following merely moves the same registry into a data declaration:

```python
@dataclass(frozen=True)
class EngineData:
    constitution_a: ConstitutionAData
    constitution_b: ConstitutionBData
    # One field for every current and future constitution.
```

It breaks subsystem encapsulation and the open/closed property. Adding a
constitution still edits a central class, changes the complete kernel ABI, and
exposes unrelated storage to every pipeline function.

## Why automatic generation is data-oriented under another name

A proposed generator could:

```text
iterate registered systems
-> inspect each DataClass
-> flatten its buffers into kernel arguments
-> bind lifecycle function pointers
-> generate a composition-specific kernel signature
-> include the object/function graph in the cache key
-> emit direct calls into the CUDA graph
```

This removes handwritten branches, but it is exactly the compile-time behavior
for which `data_oriented` exists. Calling it `PipelineSpec`,
`GeneratedArgumentPack`, or `bind_data_func` does not change the semantics.
It is still compiler-managed structural reflection and static method binding.

Generated Python/AST source is worse than using the existing mechanism:

- diagnostics point into generated code;
- source hashing and fastcache invalidation become more complex;
- type checking no longer sees the real call graph;
- debugger and profiler names lose their source-level owner;
- the generated API must reproduce member pruning, nested data handling,
  rebinding rules, and function-source invalidation already implemented by
  Quadrants.

## Function pointers do not remove the contradiction

Python function pointers can select an implementation before graph
compilation. CUDA graph execution cannot dynamically call those Python
objects. To retain static GPU dispatch, QD must eventually turn:

```text
(data object, Python function pointer)
```

into:

```text
flattened kernel arguments + a direct compiled call
```

Doing that automatically requires the compiler-visible binding described
above. Refusing that binding leaves only explicit per-pipeline calls.

Runtime device-function-pointer dispatch would avoid compile-time binding, but
it is the wrong target:

- it prevents normal inlining and cross-function optimization;
- it complicates CUDA graph and backend portability;
- it moves an invariant composition decision into every timestep;
- it does not solve heterogeneous function signatures or data ownership.

## The actual architectural choice

There are two coherent designs.

### Option A: compiler-visible system composition

Permit a narrowly defined compiler-aware numerical-system wrapper:

```python
@qd.data_oriented
class FEMConstitutionSystem(SimSystem):
    def __init__(self, data: FEMConstitutionData):
        self.data = data

    @qd.func(requires_top_level=True)
    def assemble(self, linear_system):
        ...
```

The storage may still live in a separate field-only data class. The wrapper
provides the compiler-visible relationship between owner, data, and lifecycle
functions. Composition is finalized before graph compilation, function lists
are statically lowered, and changing the composition rebuilds the graph.

Adding a constitution changes its own implementation plus one builder
registration. Existing lifecycle implementations do not enumerate it.

### Option B: explicit concrete pipelines

Prohibit all compiler-aware wrappers and write a concrete kernel for each
supported composition:

```python
@qd.kernel(graph=True)
def rigid_fem_contact_step(
    rigid: RigidData,
    fem: FEMData,
    contact: ContactData,
    linear_system: LinearSystemData,
):
    rigid_predict(rigid)
    fem_predict(fem)
    ...
```

This is simple and explicit, but it deliberately gives up generic lifecycle
dispatch. Adding a new participating system requires modifying or adding a
pipeline. At constitution scale, shared pipeline helpers still need explicit
calls for every active constitution unless separate composition-specific
kernels are maintained.

Option B is reasonable when pipelines are few and fixed. It is not equivalent
to an extensible SimSystem framework.

## Work that cannot be preserved simultaneously

Under a strict no-`data_oriented`, no-reflection, and no-generated-code rule,
the following combination is not implementable:

```text
arbitrary registered SimSystems
+ independently encapsulated heterogeneous data
+ graph-level static lifecycle dispatch
+ no central per-phase dispatch table
+ add-one-system without editing existing phases
```

This is the precise impossibility boundary. Static compilation must know the
complete data ABI and direct call graph somewhere. The choices are:

- write that knowledge manually into every concrete pipeline; or
- let a compiler-visible composition mechanism derive it.

There is no third representation that is both automatic and unaware of the
system/data structure.

## Recommendation

Do not generalize the constraints of the small RigidSolver to the full CGQ
constitution surface.

Retain separate field-only DataClasses, but allow `@qd.data_oriented` on the
numerical wrapper that binds:

- one owner identity;
- one data object;
- declared dependencies;
- statically selected lifecycle functions.

Keep the existing safeguards:

- composition closes before graph compilation;
- registry lookup never occurs in device code;
- lifecycle calls lower to direct graph tasks;
- adding/removing a system invalidates and rebuilds the graph;
- live sizes remain device scalars;
- fastcache keys include all data/function dependencies that affect codegen.

If the absolute prohibition remains, the architecture document must stop
claiming generic SimSystem lifecycle composition. It should instead describe a
set of explicit, manually maintained pipeline variants and accept the
constitution-by-phase dispatch cost.

## Review test

Before accepting either design, apply this change request:

> Add a new constitution with prediction, assembly, energy, and Hessian action
> without modifying any existing lifecycle phase.

Option A should require the new system and its builder registration only.
Option B cannot pass this test without code generation or edits to central
pipeline code. That difference, rather than decorator syntax, is the relevant
architectural decision.
