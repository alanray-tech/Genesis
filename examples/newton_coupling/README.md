# Newton Coupling Examples

These examples use the experimental graph-native system runtime under `genesis.engine.systems`. The runtime consumes
an already-built Genesis Rigid model and advances one numerical timestep without the existing Simulator or coupler
lifecycle.

The proof of concept requires double precision.

```powershell
python examples/newton_coupling/franka_cube.py --runtime newton --gpu -v
python examples/newton_coupling/franka_cube.py --runtime native --gpu
```

`franka_cube.py` exercises articulated dynamics, position control, native rigid contact, friction, and grasping. The
two runtime modes use the same authored scene so their state and steady-state performance can be compared. With `-v`,
the new runtime refreshes the viewer after each numerical timestep and uses the viewer's real-time pacing.
