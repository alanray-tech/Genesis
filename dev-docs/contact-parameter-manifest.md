# Contact Parameter Manifest

Status: internal runtime manifest.

These names, units, defaults, validation rules, and update cadence define the
internal composition interface. Values in this file are not tuning
suggestions. `NewtonCouplerOptions` exposes only the supported first-version
physical parameters; the remaining keys are internal or diagnostic.

Primary sources:

- `genesis/engine/systems/contact.py`
- `genesis/engine/systems/builders.py`
- `genesis/engine/systems/contact_system.py`
- `genesis/engine/systems/consistent_ipc_contact.py`
- `genesis/engine/systems/sim_engine.py`

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
- `contact/constitution`: string, default `"auto"`. The accepted manifest values are
  `"auto"`, `"consistent_ipc"`, `"gipc"`, `"adhesive_ipc"`, and
  `"variational_adhesive_ipc"`. This milestone resolves `"auto"` to
  `ConsistentIPCContactConstitution`; the other constitutions are not implemented.
- `friction/eps_v`: float64 velocity, default `1e-2`, strictly positive.
- `extras/capacity_grow_factor`: float64, default `1.2`, minimum `1.0`.
- `extras/capacity_shrink_threshold`: float64, default `0.8`, range `(0, 1]`.
- `topo/grow_factor`: float64, default `1.5`, minimum `1.0`.
- `bvh/type`: string, default `"info_lbvh_batched_dop14"`.
- `bvh/pt_query`: string, default `"warp"`.
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
- `rigid_forest/fused`: integer boolean, default `1`; zero selects the
  compact level-order debug fallback.
- `extras/rigid_forest/genesis_legacy`: diagnostic integer boolean, default
  `0`; one forces the original scan-all level implementation as a
  non-production performance baseline and overrides `rigid_forest/fused`.
- `extras/pipeline/genesis_serial`: diagnostic integer boolean, default `0`;
  one retains the pre-overlap serial ordering of independent contact/BVH/CCD
  branches as a non-production A/B baseline. It changes scheduling only, not
  contact formulas, live extents, or solver selection.
- `extras/rigid_contact/genesis_collision`: diagnostic integer boolean,
  default `0`; one retains native collision alongside graph contact for A/B
  comparison.
- `extras/sort_reduce/genesis_legacy`: diagnostic integer boolean, default
  `0`.
- `extras/bvh/genesis_legacy_fp64_bounds`: diagnostic integer boolean, default
  `0`.
- `extras/bvh/genesis_legacy_refit`: diagnostic integer boolean, default `0`.

`contact/ccd_partition` and `contact/ccd_partition_sv_max_iter` are reserved
manifest values; the first version uses one global screw-CCD alpha and does
not consume them.

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

## Example authoring

Cloth-only scenes derive one analytical ground halfplane from
`FEMOptions.floor_height`. Mixed Rigid-QCloth scenes author fixed Rigid support
geometry instead, preventing the analytical plane from also intersecting the
robot. Every scene must start collision-free and outside all contact-thickness
shells.
