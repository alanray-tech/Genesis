# CGQ MinCoo Rigid Dynamics Backend

The Newton runtime has two build-time rigid dynamics backends:

- `genesis` keeps the Genesis `RigidSolver` prediction, gradient, Hessian, and
  state commit. This is the default and preserves existing behavior.
- `cgq_mincoo` keeps the Genesis scene, entity, state, and control APIs but
  replaces rigid dynamics with the CGQ hard minimal-coordinate formulation.

Select the backend through the system config:

```python
engine = build_scene_engine(
    scene,
    contact_config={
        "rigid/dynamics_backend": "cgq_mincoo",
        "linear_system/solver": "partition_pcg",
    },
)
```

The pure rigid builder accepts the same key:

```python
engine = build_rigid_engine(
    scene.rigid_solver,
    config={
        "rigid/dynamics_backend": "cgq_mincoo",
        "linear_system/solver": "partition_pcg",
    },
)
```

`partition_pcg` is the default; use `linear_pcg` for a matched global-PCG
fallback. See [Component-partitioned PCG](partition-pcg.md).

`cgq_mincoo` is a graph-native implementation and does not import or link the
`cuda-graph-qipc` repository. It ports the following CGQ contracts:

- Per-link BDF1 translational kinetic potential.
- Four-substep RK4 torque-free rotational prediction.
- Cancellation-safe rotational energy and full-Newton rotational Hessian,
  with whole-block Gauss-Newton fallback when the block is not positive
  definite.
- Physical-link gradient and Hessian projection through
  `RigidJointForestSystem`.
- CGQ scalar-joint position, velocity, and force control; force clamping; and
  quadratic joint-limit penalty. The default limit stiffness is configured by
  the global `rigid/joint_limit_kappa` and is `1e6`. Limits use Genesis
  absolute `qpos` coordinates; control targets use `qpos - qpos0`.
- Screw-path line search and CGQ velocity commit.

The forest supports fixed or free roots and scalar revolute or prismatic
edges. Genesis native rigid collision is intentionally unavailable in
`cgq_mincoo`; coupled contact must use the rigid contact proxy path.
Spherical and other multi-DOF edges, equality constraints, passive joint
stiffness/damping/friction loss, armature, and externally applied link wrenches
are not part of the current port. Use the `genesis` backend when a comparison
depends on those features. Control potentials currently apply only to scalar
forest edges, not six-DOF free roots, and assume Genesis' standard reducible PD
actuator layout; custom `act_bias` actuator formulas are unsupported.

For matched A/B measurements, use:

```text
python examples/newton_coupling/franka_cloth_cgq_benchmark.py \
  --rigid-backend genesis ...

python examples/newton_coupling/franka_cloth_cgq_benchmark.py \
  --rigid-backend cgq_mincoo ...
```

The benchmark JSON records the selected backend as `rigid_backend`. Discard
the cold graph-compilation frames with `--warmup`; they are not steady-state
simulation measurements.

For interactive Franka cloth teleoperation with the MinCoo backend, run:

```text
python examples/newton_coupling/cgq_franka_cloth_cube_teleop.py
```

This entry point shares the scene, key bindings, headless flags, and
performance overlay with `franka_cloth_cube_teleop.py`; only the rigid dynamics
backend selection differs.
