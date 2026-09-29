# CGQ Contact Parameter Manifest

Status: authoritative migration manifest.

Ground truth is `cuda-graph-qipc` `main@42e7d4cbbad08739107ad830a17918f5f0f209ff`.
Genesis must copy these names, units, defaults, validation rules, and update
cadence. Values in this file are not tuning suggestions.

Primary sources:

- `qipc/scene/config.py`
- `qipc/contact.py`
- `qipc/solver/solver.py::_wire_contact`
- `qipc/_src/native/solver/contact_system.cu`
- `docs/contact.md`
- `docs/simulation_pipeline.md`

## Scene contact configuration

- `contact/enable`: integer boolean, default `1`.
- `contact/d_hat`: float64 length, default `0.01`.
- `contact/max_step_in_d_hat`: float64 ratio, default `-1.0`; `-1` disables the cap.
- `contact/ccd_bound`: string, default `"directional"`; accepted values are
  `"additive"` and `"directional"`.
- `contact/adaptive_kappa_mode`: string, default `"per-body"`; accepted values are
  `"off"`, `"global"`, `"per-vertex"`, and `"per-body"`.
- `contact/adaptive_kappa_tick`: string, default `"newton"`; accepted values are
  `"frame"` and `"newton"`.
- `Solver.enable_adaptive_kappa` controller defaults: `gap_ratio=0.01`,
  `grow=2.0`, `max_scale=128.0`, `calm_time=0.3`, `relax_time=0.4`, and
  `hysteresis=2.0`.
- `contact/init_collision_pair_capacity`: integer, default `1000`, minimum `1`.
- `contact/ccd_partition`: integer boolean, default `1`.
- `contact/ccd_partition_sv_max_iter`: integer, default `64`, minimum `1`.
- `contact/intersection_check`: integer boolean, default `0`.
- `contact/intersection_check_capacity`: integer, default `1024`, minimum `1`.
- `contact/constitution`: string, default `"auto"`. The accepted CGQ values are
  `"auto"`, `"consistent_ipc"`, `"gipc"`, `"adhesive_ipc"`, and
  `"variational_adhesive_ipc"`. This milestone resolves `"auto"` to
  `ConsistentIPCContactConstitution`; the other constitutions are not implemented.
- `friction/eps_v`: float64 velocity, default `1e-2`, strictly positive.
- `extras/capacity_grow_factor`: float64, default `1.2`, minimum `1.0`.
- `extras/capacity_shrink_threshold`: float64, default `0.8`, range `(0, 1]`.
- `topo/grow_factor`: float64, default `1.5`, minimum `1.0`.
- `bvh/type`: string, default `"info_lbvh_batched_dop14"`.
- `bvh/ee_query`: string, default `"dual"`.
- `bvh/dual/frontier_levels`: integer, default `0`, range `[0, 18]`; zero selects auto depth.
- `bvh/dual/target_waves`: float64, default `24.0`.
- `bvh/dual/max_levels`: integer, default `18`.
- `rigid_proxy/globalization`: string, default `"merit"`; `"watchdog"` is a
  non-production diagnostic control.
- `rigid_proxy/restoration`: integer boolean, default `1`; zero is a
  non-production diagnostic control.
- `rigid_proxy/test_merit_energy_bias`: float64, default `0.0`, finite and
  non-negative; nonzero is test-only fault injection.
- `rigid_forest/fused`: integer boolean, default `1`; zero retains only the
  non-production generic correction path for controlled comparison.

## Contact table

`ContactTabular` creates the default contact element and row at construction.

- `friction_rate`: float64 coefficient, default `0.05`.
- `resistance`: float64 barrier stiffness, default `1e4`.
- `enable`: boolean, default `True`.
- `enable_ee`: boolean; when omitted it equals `enable`.

This milestone implements the default and ordinary pair rows only. Adhesion,
variational adhesion, bond, and release columns remain excluded rather than being
silently accepted.

## Derived and kernel constants

- `dt_sq = dt * dt`; `ContactSystem.set_dt_sq` updates it from the live timestep.
- `ccd_eta`: float64, `0.2`.
- CCD maximum iterations: `50000`.
- Consistent-IPC Gauss-Newton clamp threshold: `gassThreshold = 1e-6`.
- Unrecorded minimum-gap sentinel: `MIN_GAP_RATIO_UNRECORDED = 1e300`.
- EE mollifier coefficient:
  `eps_x = 1e-3 * ||e_0||^2 * ||e_1||^2`, computed from rest edges.
- Contact pair capacities, frozen-friction capacities, assembly demand, padding,
  active counts, and unique counts are device-resident dynamic scalars.
- Overflow growth uses
  `ceil(required * extras/capacity_grow_factor)` and never scene-specific constants.

## Runtime cadence

- `friction_snapshot` executes once at frame checkpoint CP0 and freezes lagged
  friction pairs for the timestep.
- Adaptive kappa defaults to per-body growth at Newton-iteration boundaries;
  frame CP0 retains hold/relax bookkeeping.
- Contact assembly demand is counted before filter/assemble.
- Pair, assembly, and global-triplet overflow yield to the host. The owning system
  reallocates, clears only its overflow flag, and resumes from the exact checkpoint.
- Normal frames perform no host readback inside the graph.

## Milestone scene authoring

Only geometry and initial pose are example-specific:

- one analytical ground halfplane;
- one vertical XZ cloth;
- one vertical YZ cloth;
- collision-free staggered heights so frame 0 has no cloth-cloth intersection and
  all vertices begin outside the halfplane thickness shell.

Solver/contact values in the example use this manifest unchanged.
