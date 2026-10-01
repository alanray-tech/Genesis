# Quadrants Compile-Speed Optimization Handoff

Status: 2026-10-01

This document is the working handoff for continuing Quadrants compiler
optimization with the Genesis QIPC pipeline as the large-graph workload. It is
not a historical summary only: the commands, measurement contract, current
repository state, remaining profiles, and next work order below are intended
to let another engineer resume without reconstructing the investigation from
chat logs.

The two user-facing goals are:

1. Reduce cold compilation during development.
2. Reduce process startup after the same graph has compiled successfully once.

Steady-state simulation speed and numerical behavior are guardrails. A faster
compile obtained by weakening cache invalidation, changing solver semantics, or
special-casing this scene is not acceptable.

## 1. Keep three different costs separate

There are three unrelated meanings of "compile" in this workspace:

- **Native Quadrants build**: compiling the Quadrants C++ extension itself
  after changing compiler source. `CMAKE_BUILD_PARALLEL_LEVEL=24` helps this.
- **Cold Quadrants program compilation**: Python materialization, Quadrants IR
  passes, offload, LLVM/PTX generation, CUDA graph construction, and first
  launch for the Genesis graph. `QD_NUM_THREADS` affects only portions after
  offload; it does not control the native build.
- **Cached restart**: a new Python process validates source/config/arguments,
  loads persisted compiler artifacts, rebuilds the CUDA executable graph, and
  completes its first step.

Never combine these into one number. In particular, an editable extension
installed with `editable.rebuild=true` may rebuild C++ during import and make a
cached application restart appear slow. Use an explicit native build and
`editable.rebuild=false` for application benchmarks.

## 2. Repository checkpoint

### Genesis workload

- Repository:
  `C:\Users\81946\Projects\GenesisWorldCouplingSystem\Genesis-RigidOnly`
- Branch: `newton/rigid-only-baseline`
- Checkpoint when this handoff was written: `07c039fb`
- Pull request:
  [alanray-tech/Genesis#6](https://github.com/alanray-tech/Genesis/pull/6)
- Primary workload:
  `examples/newton_coupling/franka_cloth_cube_teleop.py`
- Smaller workload:
  `examples/newton_coupling/franka_cloth.py`

The existing untracked `output/` directory contains local artifacts. Do not
delete it and do not commit it.

### Quadrants compiler

- Repository:
  `C:\Users\81946\Projects\GenesisWorldCouplingSystem\quadrants`
- Branch: `dev/compile-time-opt`
- Checkpoint: `bf3a00645617d77135abe937fa6ed1eb6ada018c`
- Remote branch: `fork/dev/compile-time-opt`
- Pull request:
  [alanray-tech/quadrants#2](https://github.com/alanray-tech/quadrants/pull/2)
- Upstream `origin/main` at handoff time: `7e3b27b89`

The branch is stacked on `fix/issue-750-nested-checkpoint`; the pull request is
not a clean patch directly against upstream main. Do not describe it as
upstream-merged. Prefer a new local branch from `bf3a00645` for the next
experiment, and split independently reviewable compiler changes before asking
upstream to review them.

Relevant commits, oldest first:

- `82cc06048`: graph do-while checkpoints.
- `459a3eb57`: checkpoint propagation through inlined functions.
- `e5811c2dd`: large-graph compile-time optimizations.
- `12119f03e`: occupancy-clamped CUDA grid-stride launches.
- `3b0f9c2ad`: graph-parallel regions inside checkpoints.
- `bf3a00645`: inline dynamic range bounds into CUDA workers.

Only `e5811c2dd` is principally a cold compiler optimization. The last three
commits primarily change runtime launch/graph behavior. They can change graph
task count, graph construction time, and first launch, so the exact current
revision still needs a fresh baseline.

## 3. Mandatory first action: fix the source/native revision mismatch

At handoff time, the Genesis virtual environment imports:

- Python source from the current local Quadrants checkout;
- the native extension from
  `Genesis-RigidOnly\.venv\Lib\site-packages\quadrants\_lib\core\...`;
- a native banner reporting commit `3b0f9c2a`, not current Quadrants HEAD
  `bf3a00645`.

The Quadrants repository's own `.venv` is older still and reports `63cf408f`.
Do not collect a new baseline from either mismatched environment.

Record the source state and loaded binary before rebuilding:

```powershell
cd C:\Users\81946\Projects\GenesisWorldCouplingSystem\quadrants
git rev-parse HEAD
git status --short
& ..\Genesis-RigidOnly\.venv\Scripts\python.exe -c `
  "import quadrants; from quadrants._lib import core; print('python=', quadrants.__file__); print('native=', core.__file__); print('native_commit=', core.get_commit_hash())"
```

Rebuild the extension into the Genesis environment:

```powershell
$env:CMAKE_BUILD_PARALLEL_LEVEL = "24"
& ..\Genesis-RigidOnly\.venv\Scripts\python.exe `
  -m pip install --no-build-isolation -e . `
  --config-settings=editable.rebuild=false
```

Then rerun the revision command. The native commit must match the source commit
used for the experiment. Also record `git status --short`: a native banner can
identify HEAD while omitting uncommitted source modifications, and a persistent
CMake configure can leave a stale commit stamp. A benchmark record therefore
needs all of:

- full Git HEAD;
- dirty/clean state and patch hash if dirty;
- Python package path;
- native extension path and reported commit;
- Quadrants version, Python, LLVM, CUDA, GPU, driver, and OS.

The persistent native build directory is configured as
`build/{wheel_tag}` in `quadrants/pyproject.toml`, so later native rebuilds
should be incremental. Keep automatic editable rebuild disabled while measuring
application startup.

## 4. Evidence already collected

The authoritative public record is
[Quadrants issue #945](https://github.com/Genesis-Embodied-AI/quadrants/issues/945).
The Genesis-side aggregate report is
[QIPC coupling performance report](performance-report/qipc-coupling-performance-report.md).

Historical benchmark environment:

- Windows 11 build 26200;
- RTX 5090;
- Python 3.13;
- LLVM 22.1;
- CUDA FP64;
- minimal-coordinate Franka plus a 32 x 25 cloth;
- one large `@qd.kernel(graph=True, fastcache=True)` pipeline.

### 4.1 Initial super-linear behavior

With advanced optimization enabled on the original revision:

- rigid-only backend compile: `73.55 s`;
- cloth-only backend compile: `10.73 s`;
- mixed backend compile: `175.64 s`.

The mixed graph cost more than twice the sum of its components. This located
the original problem in repeated work over monolithic pre-offload IR rather
than task-level code generation.

Increasing user-kernel compile workers did not help:

- `QD_NUM_THREADS=4`: `174.67 s`;
- `QD_NUM_THREADS=24`: `175.64 s`.

This does not imply that parallel compilation is useless. It proves only that
the measured dominant work occurred before the existing task-parallel boundary.
Do not repeat thread-count sweeps until a profile shows work has moved into
independent regions.

### 4.2 Strict pass-level results now represented by `e5811c2dd`

Same-revision measurements:

- Skip inapplicable pre-offload `merge_global_ptrs` for ndarray-only IR:
  `51.99 -> 38.57 s`, or `1.35x`.
- Avoid whole-tree SSA user searches while lowering frontend statements:
  `lower_ast 10.429 -> 0.229 s`, or `45.5x`.
- Reuse a statement-usage index in constant folding:
  `constant_fold 5.841 -> 0.527 s`, or `11.1x`.
- Combined lower/constant-fold changes:
  `compile_to_offloads 21.271 -> 5.312 s`;
  `Program::compile_kernel 24.660 -> 8.690 s`;
  backend graph `38.57 -> 22.99 s`.
- Skip unconditional DCE in `bit_loop_vectorize` when no quantized-array loop
  exists: approximately `0.93 s` removed from the measured no-op path.
- Honor `external_optimization_level` in CUDA:
  O3 `22.14 s` versus O1 `20.04 s`, a `9.5%` cold-compile reduction in that
  profile.

The O1 result is a development-mode candidate, not permission to change the
production default without current runtime evidence.

### 4.3 Aggregate values are context, not per-patch evidence

Across different revisions and configurations:

- production/advanced backend: `175.64 -> 28.44 s`;
- development/non-advanced backend: `54.68 -> 22.14 s`;
- materialization plus development backend: `65.09 -> 33.68 s`.

Do not quote these as strict individual-change speedups. Revisions differ.

### 4.4 Existing restart evidence

After clearing an accidentally persistent `QD_OFFLINE_CACHE=0`:

- first compile and cache population: `22.08 s`;
- unchanged restart graph load: `0.82 s`.

This is evidence that persistence works, not a complete process-startup
breakdown. The next benchmark must time imports, scene construction, source
validation, artifact load, graph instantiation, and first completed step
separately.

### 4.5 Last known remaining cold profile

This profile predates the exact `bf3a00645` rebuild and must be revalidated:

- Python `Kernel.materialize`: approximately `11.54 s`;
- `compile_to_offloads`: approximately `5.31 s`;
  - `full_simplify`: approximately `1.16 s`;
  - `cse_offloaded_tasks`: approximately `1.12 s`;
  - `scalarize`: approximately `0.75 s`;
  - `offload`: approximately `0.31 s`;
- CUDA `compile_module_to_ptx`: approximately `4.90 s`.

Only about `1 ms` of the measured `1.12 s` in
`cse_offloaded_tasks` was usage replacement. Profile candidate bucketing,
statement comparison, address analysis, and fixpoint traversal before changing
that pass.

Current Quadrants already avoids a monolithic pre-offload CFG optimization and
runs the relevant CFG work per offloaded task. Inspect
`quadrants/transforms/cfg_optimization.cpp`; do not reimplement the already
landed CFG partitioning work.

## 5. Current Genesis compiler modes

`genesis/__init__.py` currently initializes Quadrants with:

- `force_scalarize_matrix=True`;
- `advanced_optimization=True`;
- `cfg_optimization=False`;
- `external_optimization_level=3` through its Quadrants default;
- `offline_cache=True` through its Quadrants default;
- `src_ll_cache=True` through its Quadrants default;
- four compile workers unless `QD_NUM_THREADS` is set.

The teleoperation example then sets:

- `advanced_optimization=False` and `external_optimization_level=1` by
  default;
- `advanced_optimization=True` and `external_optimization_level=3` with
  `--advanced-optimization`;
- an explicit `--external-optimization-level 0..3` overrides only the LLVM
  level when a comparison needs another combination.

These three options are independent:

- `cfg_optimization` controls the expensive CFG forwarding/DSE pass.
- `advanced_optimization` controls the Quadrants fixed-point simplify path.
- `external_optimization_level` selects LLVM/code-generation O0 through O3.

Do not repeat the stale explanation that treats `cfg_optimization` and
`advanced_optimization` as the same switch. Also record every option because it
participates in cache selection; alternating development and production modes
normally produces distinct cache entries.

Use these named modes in new results:

- **Production**: advanced optimization on, external O3.
- **Current development baseline**: advanced optimization off, external O3.
- **Candidate development**: advanced optimization off, external O1.

Keep `cfg_optimization=False` for all three unless the experiment explicitly
tests that option.

## 6. Cache and restart architecture

Quadrants has four relevant cache layers:

1. An in-process materialized-kernel cache.
2. The C++ persisted compiled-artifact/offline cache.
3. Python fastcache (`src_ll_cache`) for source/config/argument validation and
   pruning metadata.
4. NVIDIA's independent driver cache.

Important controls:

- `offline_cache=True` persists C++ compilation artifacts.
- `offline_cache_file_path` chooses the cache root.
- `src_ll_cache=True` enables source-level fastcache.
- `print_non_pure=True` reports kernels that are not fastcache-eligible.
- `QD_OFFLINE_CACHE=0` disables Quadrants persistence.
- `CUDA_CACHE_DISABLE=1` disables the separate NVIDIA driver cache.

`offline_cache=False` does not disable Python fastcache bookkeeping. A strict
cold compiler run must use a fresh dedicated cache path and disable both
offline cache and `src_ll_cache`. Genesis does not currently expose the latter
as a public `gs.init` argument. A benchmark harness may set
`qd.lang.impl.get_runtime().src_ll_cache = False` immediately after `gs.init`
and before any kernel materializes. Keep that benchmark-only control out of
solver code.

Never delete a user's shared cache. Allocate a unique directory under
`output/quadrants-compile/` for each experiment.

Audit inherited environment state before every reported run:

```powershell
Get-ChildItem Env:QD_*
Get-ChildItem Env:CUDA_CACHE_DISABLE -ErrorAction SilentlyContinue
```

Select an isolated cache without touching the default:

```powershell
$runId = Get-Date -Format "yyyyMMdd-HHmmss"
$env:QD_OFFLINE_CACHE_FILE_PATH = Join-Path `
  $PWD "output\quadrants-compile\$runId"
$env:QD_NUM_THREADS = "4"
```

Set `QD_NUM_THREADS` explicitly even when testing the four-thread default.
Record and remove experiment-only environment variables after the run so a
later interactive launch cannot silently inherit cold-cache or O1 settings.

### 6.1 Fastcache path

For a `fastcache=True` kernel, `Kernel._try_load_fastcache` currently:

1. hashes source, compiler configuration, version, and device capabilities for
   an L1 key;
2. loads pruning paths from L1;
3. hashes only the argument paths selected by pruning to produce an L2 key;
4. validates the Python-side L2 record;
5. calls `Program::load_fast_cache(...)` to load the C++ artifact;
6. restores graph/checkpoint metadata and skips materialization pass 0 on a
   complete hit.

Useful observations on a kernel's `_primal` object:

```python
kernel._primal.src_ll_cache_observations
kernel._primal.fe_ll_cache_observations
```

Source-level fields:

- `cache_key_generated`;
- `cache_validated`;
- `cache_loaded`;
- `cache_stored`.

The frontend/offline field is:

- `cache_hit`.

Interpret restart failures in this order:

- `cache_key_generated=False`: L1 was absent, source hashing/pruning could not
  produce a usable key, the arguments contain an unsupported field-like value,
  or the kernel is not fastcache-pure.
- key generated but `cache_validated=False`: L2 miss or source/config/argument
  key churn.
- validated but `cache_loaded=False`: the Python record exists but the C++
  artifact did not load; inspect version, compiler config, device capability,
  artifact existence, and native/source revision.
- loaded but startup is still slow: time source hashing, narrow argument
  hashing, JSON I/O, artifact deserialization, CUDA module load, graph
  instantiation, imports, and scene authoring separately.

The Genesis SimEngine initialization, contact initialization, and step kernels
are all already declared `fastcache=True`. Inspect all three, not just
`_step_kernel`:

- `_initialize_global_resources`;
- `_init_contact_kernel`;
- `_step_kernel`.

### 6.2 Cache correctness is a hard gate

`contact-performance-debt.md` records an unresolved Quadrants dependency bug:
an arithmetic expression such as
`qd.static(range(self.n_levels_host - 1))` did not add the nested property to
the specialization key. A graph compiled for one forest depth could be reused
for another and omit mapped-forest work.

Genesis now keeps live topology extents in zero-dimensional device scalars and
emits graph topology from allocation capacity, but Quadrants still needs a
regression test that used properties inside static arithmetic expressions
invalidate correctly. Do not optimize source hashing by weakening conservative
dependency discovery.

Other cache hazards:

- `qd.field` arguments are not fastcache-eligible; Genesis' ndarray path is.
- Python floats used as template values can generate one specialization per
  value. Use `raise_on_templated_floats=True` diagnostically.
- Dynamic simulation sizes must remain device scalars. Do not turn a live size
  into a host static property to improve a benchmark.
- A compiler-config change intentionally creates another cache key.
- Python source changes and native C++ changes have different invalidation
  paths. The native version/hash must be correct before trusting a hit.

## 7. Required benchmark harness

Stop relying on one-off monkey patches embedded only in a terminal session.
Before the next optimization, add a standalone harness, suggested location:

`dev-docs/performance-report/benchmark_quadrants_compile.py`

It should reuse the same scene-construction function as
`franka_cloth_cube_teleop.py`; it must not maintain a simplified copy that can
drift from the real workload.

The harness should emit machine-readable JSON containing:

- source and native Quadrants revisions;
- dirty state;
- Genesis revision;
- hardware/software versions;
- complete compiler configuration;
- cache directory and cache/driver state;
- import time;
- `gs.init` time;
- scene authoring/build time;
- SimEngine construction/contact-initialization time;
- Python `Kernel.materialize` time by kernel;
- first `Kernel.launch_kernel` time by kernel;
- C++ scoped-profiler tree;
- CUDA graph construction/instantiation time;
- first launch plus synchronization time;
- total process-to-first-completed-step;
- every fastcache observation field;
- graph task count if available;
- Newton, PCG, line-search, contact-count, and overflow telemetry;
- 200-step warm runtime median and percentiles for guard runs.

Use `time.perf_counter()` around Python phases and synchronize before closing a
GPU interval. Quadrants' host scoped profiler is always available:

```python
qd.profiler.clear_scoped_profiler_info()
# Compile or load and execute the selected phase.
qd.sync()
qd.profiler.print_scoped_profiler_info()
```

Wrapping `quadrants.lang.kernel.Kernel.materialize` and
`Kernel.launch_kernel` is acceptable inside the benchmark harness. Filter and
report by `self.func.__qualname__` so initialization kernels and `_step_kernel`
remain distinguishable. Do not ship global monkey patches in Genesis runtime
code.

### 7.1 Primary workload

Use headless, fixed arguments:

```powershell
cd C:\Users\81946\Projects\GenesisWorldCouplingSystem\Genesis-RigidOnly
& .\.venv\Scripts\python.exe `
  .\examples\newton_coupling\franka_cloth_cube_teleop.py `
  --no-gui --steps 1 --ee-query dual
```

Add `--advanced-optimization` only for production-mode measurements. Keep
asset, precision, contact parameters, query path, capacities, and scene
construction identical between runs.

The one-step example is useful for manual confirmation but is not yet a valid
phase-separated benchmark. Use the new harness for reported numbers.

### 7.2 Cold compiler protocol

For every cold sample:

- create a fresh dedicated cache directory;
- set `offline_cache=False`;
- set `src_ll_cache=False`;
- use a new Python process;
- leave the NVIDIA driver cache in its normal state and record that state;
- use `CUDA_CACHE_DISABLE=1` only for a separately named "complete first-ever"
  experiment;
- run at least three independent samples and report the median plus each raw
  value.

The cold breakdown must separately report:

- Python materialization;
- Quadrants IR/offload;
- LLVM/PTX;
- CUDA graph construction;
- first launch and synchronization.

Do not call a warm-cache result a cold compile merely because it is the first
step in the current process.

### 7.3 First population and restart protocol

Use a new dedicated directory with:

- `offline_cache=True`;
- `src_ll_cache=True`;
- process A compiling and storing the graph;
- process B starting unchanged and loading it;
- optional processes C and D to show warm-run variance.

Keep the exact same source, native extension, compiler config, device, and
arguments. Record process-to-first-completed-step as well as the internal
breakdown.

Also run a separate offline-only experiment with `src_ll_cache=False` in both
population and restart processes. It distinguishes:

- ordinary frontend work followed by a C++ offline-cache hit;
- full fastcache, which can skip materialization pass 0 and restore compiled
  data directly.

Do not alternate production O3 and development O1 in one cache directory and
expect a hit; they are intentionally different keys.

### 7.4 Runtime guard

After every accepted compiler optimization:

- run the same workload for 200 completed steps after compilation;
- report synchronized step-time median and high percentile;
- compare Newton, PCG, line-search, contact, CCD, and overflow telemetry;
- run the same mode/config on baseline and candidate;
- reject unexplained numerical or graph-structure changes.

Compile-speed work must be general. Insights from this scene may identify a
compiler pattern, but code may not recognize Franka, cloth, a particular
capacity, or a particular scene topology.

## 8. Next investigation order

### Step 0: establish a valid binary

Rebuild `bf3a00645` into the Genesis environment and verify both source and
native revisions. Do not profile before this passes.

### Step 1: land the durable harness

The harness and its JSON output schema are prerequisites for optimization.
Validate that:

- cold mode cannot accidentally read either Quadrants cache;
- restart mode reuses one explicit directory;
- the report includes cache observations;
- GPU timings synchronize correctly;
- production, current-development, and candidate-development modes are named
  explicitly.

### Step 2: rebaseline exact current HEAD

Collect:

- cold production O3;
- cold current-development O3;
- cold candidate-development O1;
- cache population plus full-fastcache restart;
- cache population plus offline-only restart;
- a 200-step runtime guard for O3 and O1.

The old profile is a prioritization hint only. Use the new exact profile to
choose work.

### Step 3: instrument cached restart

Add scoped, optional timing around:

- `Kernel._try_load_fastcache`;
- source/config hashing;
- L1 file read and validation;
- narrow argument hashing;
- L2 file read and validation;
- `Program::load_fast_cache`;
- artifact deserialization/module load;
- CUDA graph instantiation;
- first launch and synchronization.

The timing must be opt-in and nearly free when disabled. It should make a long
silent phase visibly different from a solver hang.

Optimize the largest measured restart stage, not the stage that is easiest to
edit. Likely questions, to be answered by data:

- Is the source walker rereading or rehashing the same functions?
- How many functions/files/bytes are included in L1 validation?
- Is the narrow argument walk actually narrow for the SimEngine object?
- Is JSON file-per-key I/O or `_touch()` significant on Windows?
- Is C++ artifact deserialization/module load dominant?
- Does CUDA executable-graph instantiation dominate after a complete cache
  hit?
- Is most process wall time outside Quadrants in imports or scene authoring?

### Step 4: reduce Python cold materialization

Current implementation in `python/quadrants/lang/kernel.py` calls
`get_tree_and_ctx` and constructs an `ASTGenerator` in pass 0 and pass 1. Pass
0 discovers used properties and propagates pruning information; pass 1 emits
the real kernel. A successful full fastcache hit can skip pass 0.

Profile before redesigning. Candidate general directions:

- cache/reuse parsed source and immutable AST transforms between the two passes;
- avoid repeated source-file reads and source-info construction;
- persist safe pruning/dependency products independently from backend artifacts;
- measure pruning fixpoint propagation and data-oriented path folding;
- cache immutable function-level transformations by a content/config key;
- avoid transformations whose required syntax or argument categories are
  absent.

Any reuse must account for data-oriented property dependencies, nested called
functions, static control flow, compiler config, default dtypes, and device
capabilities. The earlier static-arithmetic invalidation bug is a mandatory
regression case.

### Step 5: reduce remaining Quadrants IR work

Use the existing `QD_AUTO_PROF` scoped tree and profile:

- `full_simplify`;
- `cse_offloaded_tasks`;
- `scalarize`;
- `offload`;
- repeated `die()`/cleanup;
- any remaining per-replacement whole-tree scans.

General directions already supported by evidence:

- cheap pass-applicability checks;
- change tracking before scheduling cleanup;
- reusable use-def information or batched replacements;
- analysis invalidation instead of unconditional recomputation;
- candidate bucketing/address-analysis work in offloaded CSE;
- independent region/task processing before the current offload boundary.

Do not assume more `QD_NUM_THREADS` will fix a serial pre-offload pass. Expose
independent work first, then measure parallel scaling and memory use.

### Step 6: LLVM/PTX development mode

`external_optimization_level=1` already produced a modest cold improvement.
Re-evaluate it on exact current HEAD and measure 200-step runtime. If useful,
expose it as an explicit development choice while preserving production O3.

LLVM/PTX was approximately `4.90 s` after the major IR fixes, so it is no
longer sufficient to explain the original multi-minute compile. Do not spend
the next iteration there unless the new profile makes it the largest target.

Recent upstream history reverted several per-task or cross-process cache
experiments, including per-task CUDA module assembly and cross-process
per-task artifacts. Read the original and revert discussions around upstream
PRs `#875`, `#893`, and reverts `#914` through `#918` before reviving those
designs.

### Step 7: compile progress diagnostics

Add an optional progress/timing mode that reports major phases with elapsed
time:

- source validation/materialization pass 0;
- materialization pass 1;
- pre-offload IR;
- per-task compilation progress;
- LLVM/PTX;
- cache store/load;
- CUDA graph construction;
- first launch.

Do not enable noisy per-pass logging by default. The goal is to distinguish a
real hang from a long compiler phase without materially changing timings.

## 9. Source map

Start with these files:

- `quadrants/python/quadrants/lang/kernel.py`
  - `_try_load_fastcache`;
  - `materialize`;
  - `launch_kernel`.
- `quadrants/python/quadrants/lang/_fast_caching/src_hasher.py`
- `quadrants/python/quadrants/lang/_fast_caching/config_hasher.py`
- `quadrants/python/quadrants/lang/_fast_caching/args_hasher.py`
- `quadrants/python/quadrants/lang/_fast_caching/python_side_cache.py`
- `quadrants/quadrants/program/program.cpp`
- `quadrants/quadrants/compilation_manager/kernel_compilation_manager.cpp`
- `quadrants/quadrants/transforms/compile_to_offloads.cpp`
- `quadrants/quadrants/transforms/simplify.cpp`
- `quadrants/quadrants/transforms/whole_kernel_cse.cpp`
- `quadrants/quadrants/transforms/scalarize.cpp`
- `quadrants/quadrants/transforms/offload.cpp`
- `quadrants/quadrants/runtime/cuda/jit_cuda.cpp`
- `quadrants/quadrants/runtime/cuda/graph_manager.cpp`
- `quadrants/quadrants/system/profiler.cpp`

Genesis integration points:

- `Genesis-RigidOnly/genesis/__init__.py`
- `Genesis-RigidOnly/genesis/engine/systems/sim_engine.py`
- `Genesis-RigidOnly/examples/newton_coupling/franka_cloth_cube_teleop.py`

## 10. Validation

For a Python-only Quadrants change:

```powershell
black -l 120 <changed-files>
ruff check <changed-files>
& ..\Genesis-RigidOnly\.venv\Scripts\python.exe `
  -m pytest <target-test> -v -p no:cacheprovider
```

For a native change, explicitly rebuild the extension before testing. Useful
cache tests include:

- `tests/python/test_runtime.py`;
- `tests/python/test_offline_cache.py`;
- `tests/python/quadrants/lang/fast_caching/test_src_ll_cache.py`;
- `tests/python/quadrants/lang/fast_caching/test_src_hasher.py`;
- `tests/python/quadrants/lang/fast_caching/test_config_hasher.py`;
- `tests/python/quadrants/lang/fast_caching/test_args_hasher.py`.

Add targeted regressions before a broad suite:

- process A population/process B full-fastcache load;
- source change invalidation;
- compiler-config change invalidation;
- native-version change invalidation;
- used data-oriented property change invalidation;
- used property nested in static arithmetic/range invalidation;
- unchanged dynamic ndarray contents not causing recompilation;
- templated-float diagnostics;
- graph/checkpoint metadata restored identically on a hit.

Run `git diff --check`, targeted tests, the cold/restart benchmark, and the
200-step runtime guard before calling a layer complete. Commit one measured
optimization layer at a time with before/after evidence.

## 11. Pitfalls to avoid

- Do not benchmark with the GUI.
- Do not modify the Genesis solver, integration scheme, assets, contact
  parameters, or scene to make compilation appear faster.
- Do not replace dynamic device sizes with host-static values.
- Do not leave `QD_OFFLINE_CACHE=0` in a persistent shell.
- Do not delete shared Quadrants or NVIDIA caches.
- Do not compare runs using different cache directories or cache states.
- Do not compare different compiler configs as if they were the same baseline.
- Do not trust a Python-source checkout paired with an older native extension.
- Do not infer a compiler bottleneck from total first-step wall time.
- Do not treat cache reuse as cold compiler acceleration.
- Do not increase compile threads without locating parallelizable work.
- Do not weaken fastcache invalidation to improve restart timing.
- Do not revive recently reverted cache/module designs without reading their
  failure rationale.
- Do not leave benchmark-only monkey patches in production runtime.

## 12. Completion criteria

An optimization layer is complete only when:

1. A phase-separated baseline identifies the bottleneck.
2. The change is general compiler/runtime infrastructure, not scene-specific.
3. A same-revision, same-config, same-cache-state profile shows that bottleneck
   reduced or removed.
4. Raw measurements and aggregate speedup are recorded.
5. Cache correctness and targeted compiler tests pass.
6. The 200-step workload preserves numerical behavior and acceptable runtime.
7. The layer is committed independently before work begins on the next layer.

The immediate next deliverable is not another speculative compiler patch. It
is an exact `bf3a00645` native rebuild plus the durable benchmark harness and
fresh production/development/restart baseline.

## 13. Continuation result: 2026-10-01

The immediate deliverable above is complete. The durable harness is
`dev-docs/performance-report/benchmark_quadrants_compile.py`; the real teleop
and harness share `build_franka_cloth_cube_scene` and
`build_franka_cloth_cube_engine`, so the benchmark no longer carries a
simplified scene copy.

Final local checkpoints:

- Genesis: `ce8f27d6`, branch `newton/rigid-only-baseline`.
- Quadrants rollback checkpoint: `0d3bf03db`.
- Quadrants accepted head: `30d61d5f3`, branch
  `dev/compile-restart-opt`.
- The native extension loaded by the final verification reports the same full
  `30d61d5f3` revision and the Quadrants worktree is clean.

### 13.1 Exact baseline

The first exact `bf3a00645` development-O3 cold sample reported:

- process to first completed step: `164.875 s`;
- scene author/build: `20.637 s`;
- engine build/init: `25.460 s`;
- first step compile/launch/sync: `107.602 s`;
- `SimEngine._step_kernel` Python materialization: `31.008 s`;
  - pass 0 AST transform: `15.856 s`;
  - pass 1 AST transform: `15.119 s`;
- `Program::compile_kernel`: `51.478 s`;
- `compile_to_offloads`: `39.119 s`;
- `cse_offloaded_tasks`: `11.180 s`;
- `compile_module_to_ptx`: `24.141 s`.

The source/native mismatch warning was therefore material: the exact current
revision was much slower than the older profile used for prioritization.

An unchanged restart from a textual-LLVM artifact isolated:

- step artifact validation/load: `16.827 s`;
- cached step graph launch: `3.308 s`;
- first step compile/load/launch/sync: `20.188 s`.

Python L1/L2 validation was not responsible. Across all 20 fastcache kernels,
1,614 source slices took approximately `0.18 s`, L1 validation approximately
`0.25 s`, and L2 validation approximately `0.25 s`. The missing time was C++
textual LLVM parsing.

### 13.2 Accepted Quadrants changes

`bd14aeaef` stores LLVM modules in bitcode with a versioned payload prefix and
retains a textual-IR fallback for old artifacts.

- The large step artifact load changed from `16.827 s` to
  `0.758--0.878 s`.
- Before the next layer, cached first-step time changed from `20.188 s` to
  `3.87--4.20 s`.
- Existing textual artifacts loaded successfully through the fallback.

`6dddce8f0` uses bitcode rather than textual LLVM as the PTX cache
fingerprint. It also includes `external_optimization_level` in the PTX key;
previously O1 and O3 artifacts could address the same PTX entry.

- Cold `compile_module_to_ptx`: `24.141 -> 21.941 s` (`9.1%` lower).
- Cached PTX/module stage: approximately `2.5 -> 0.814 s`.
- Cached step launch: `3.066--3.280 -> 1.540 s`.

`30be6b503` caches immutable inspect/textwrap results on each `FuncBase`, and
`30d61d5f3` skips dataclass AST flattening when `struct_locals` is empty. A
cProfile run showed 8,742 inline function transformations in each step-kernel
pass.

- Step Python materialization: `31.008 -> 23.160 s` (`25.3%` lower).
- Pass 0: `15.856 -> 12.272 s`.
- Pass 1: `15.119 -> 10.866 s`.

The combined development-O3 cold first step changed from `107.602 -> 95.193 s`
(`11.5%` lower). The final clean full-fastcache restart at `30d61d5f3`
reported:

- step artifact load: `0.840 s`;
- step launch: `1.417 s`;
- first step load/launch/sync: `2.302 s`;
- process to first completed step: `22.932 s`.

The internal cached first-step path is therefore `8.77x` lower than the
textual-artifact baseline (`20.188 -> 2.302 s`). Do not attribute the entire
full-process change to Quadrants: Windows import and scene-build times varied
substantially between samples.

### 13.3 Development O1 result

On the same workload, O1 reduced the pre-bitcode-fingerprint first step from
`107.602 -> 101.411 s` (`5.8%`). The 200-step guard showed no material runtime
regression:

- O1 median/mean/p95: `21.748 / 22.524 / 23.381 ms`;
- O3 median/mean/p95: `21.795 / 22.308 / 23.206 ms`;
- neither mode reported overflow.

The contact trajectory is dynamic, so iteration maxima were not used as
equal-work proof. The close timing distributions and clean overflow telemetry
support O1 as the interactive default while production remains advanced/O3.

### 13.4 Rejected experiments

Two attempts to reduce offloaded CSE candidate comparisons were committed,
measured, and explicitly reverted:

- storage-identity bucketing: `cse_offloaded_tasks 11.18 -> 11.23 s`;
- canonical-index bucketing: `11.18 -> 10.88 s`, within run variance and too
  small for the added analysis surface.

Do not repeat coarse CSE bucketing. Remaining time is dominated by comparisons
within the same storage and by fixpoint/pass work, not by cross-array
candidates.

### 13.5 Validation and remaining profile

Validation completed:

- offline-cache CPU/CUDA round trips: `2 passed`;
- offload and optimization suite: `35 passed`;
- kernel implementation, dataclass, and AST suite:
  `363 passed, 6 xfailed`;
- final black, clang-format, whitespace, Ruff, and Pylint hooks: passed;
- first-step telemetry remained `newton=2`, `max_pcg=5`, `total_pcg=8`,
  `ccd_alpha=1.0`, with every overflow flag zero.

The next cold targets are still large:

- Python step materialization: approximately `23.16 s`;
- `Program::compile_kernel`: approximately `48.45 s`;
- `compile_to_offloads`: approximately `36.72 s`;
- `cse_offloaded_tasks`: approximately `10.22 s`;
- `compile_module_to_ptx`: approximately `21.42 s`.

For cached restart, Quadrants' first-step path is now approximately `2.3 s`.
The larger and highly variable full-process costs are Quadrants/Torch native
imports and Genesis scene authoring. Measure those separately before proposing
another cache change.
