"""The scene of the rigid parity case 3, shared by both sides: CGQ aeb58956's examples/rigid_franka_cloth_grasp.py run
with --press-table (y up, dt = 0.01, one substep per frame), a fixed-base Franka pressing its fingertips below the top of a
table, through a cloth draped over a small riser on the table.

The rigid geometry is the mesh of the rigid contact proxies Genesis builds for the scene (franka_genesis.py --dump), which
CGQ loads: the links of the Franka as the collision meshes of an MJCF that keeps the bodies, joints, inertials and
actuators of the Franka of Genesis, the table and the riser as fixed meshes in the world. The table and the riser are
fixed rigid links on the side of Genesis and fixed affine bodies on the side of CGQ, as in CGQ's example.
"""

import numpy as np

DT = 0.01
GRAVITY = (0.0, -9.8, 0.0)
D_HAT = 1e-3
VELOCITY_TOL = 5e-2
N_FRAMES = 100

GROUND_H = 0.4
BASE_POS = (0.0, 0.41, 0.0)
# The rotation of the base of the Franka, -90 degrees about x, which turns its z up into the y up of the world
BASE_QUAT = (np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0)
BASE_EULER_DEG = (-90.0, 0.0, 0.0)
HOME_QPOS = (0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7854, 0.04, 0.04)
FINGER_OPEN = 0.04
ARM_FORCE_RANGE = 400.0

TABLE_TOP = 0.64
TABLE_EXTENT_X = 0.36
TABLE_EXTENT_Z = 0.66
CLEARANCE = 0.002
RISER_SIZE = 0.05
RISER_OFFSET_X = 0.07
RISER_TOP = TABLE_TOP + CLEARANCE + RISER_SIZE

CLOTH_X = 0.45
CLOTH_SIZE = 0.24
CLOTH_N = 24
CLOTH_THICKNESS = 1e-3
CLOTH_SPAWN_Y = RISER_TOP + 0.005
YOUNGS_MODULUS = 2e4
SHEAR_MODULUS = 2e3
MASS_DENSITY = 200.0
BENDING_YOUNGS_MODULUS = 3e3
FRICTION = 1.0
RESISTANCE = 1e4

# The fingertips are commanded this far below the top of the table
PRESS_DEPTH = 0.02

TABLE_HEIGHT = TABLE_TOP - GROUND_H - CLEARANCE
TABLE_SIZE = (TABLE_EXTENT_X, TABLE_HEIGHT, TABLE_EXTENT_Z)
TABLE_POS = (CLOTH_X, GROUND_H + CLEARANCE + 0.5 * TABLE_HEIGHT, 0.0)
RISER_POS = (CLOTH_X + RISER_OFFSET_X, RISER_TOP - 0.5 * RISER_SIZE, 0.0)


def cloth_mesh():
    """The flat square grid of the cloth in the xz-plane above the riser, as CGQ's example builds it."""
    i, j = np.meshgrid(np.arange(CLOTH_N + 1), np.arange(CLOTH_N + 1))
    verts = np.stack(
        [
            CLOTH_X + (i.ravel() / CLOTH_N - 0.5) * CLOTH_SIZE,
            np.full(i.size, CLOTH_SPAWN_Y),
            (j.ravel() / CLOTH_N - 0.5) * CLOTH_SIZE,
        ],
        axis=1,
    )
    v00 = (np.arange(CLOTH_N)[None, :] + (CLOTH_N + 1) * np.arange(CLOTH_N)[:, None]).ravel()
    v10, v01 = v00 + 1, v00 + CLOTH_N + 1
    v11 = v01 + 1
    tris = np.stack([np.stack([v00, v10, v01], axis=1), np.stack([v10, v11, v01], axis=1)], axis=1).reshape(-1, 3)
    return verts, tris.astype(np.int32)
