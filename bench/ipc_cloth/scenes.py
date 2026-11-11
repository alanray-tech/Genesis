"""The cloth scenes of the parity of Genesis with CGQ, shared by both sides.

Every scene runs in CGQ's frame, y up, gravity (0, -9.8, 0), dt = 0.01, one substep per frame. The membrane is CGQ's
Cloth (Baraff-Witkin with strain limiting), E = 2e4, G = 2e3, rho = 200, one-sided thickness 1e-3 (cloth_drape.py).
"""

from dataclasses import dataclass

import numpy as np

DT = 0.01
GRAVITY = (0.0, -9.8, 0.0)
VELOCITY_TOL = 1e-6
YOUNGS_MODULUS = 2e4
SHEAR_MODULUS = 2e3
MASS_DENSITY = 200.0
THICKNESS = 1e-3
BENDING_YOUNGS_MODULUS = 1e6
D_HAT = 1e-2
GROUND_POS = (0.0, 0.0, 0.0)
GROUND_NORMAL = (0.0, 1.0, 0.0)


@dataclass
class ClothScene:
    name: str
    verts: np.ndarray
    tris: np.ndarray
    pinned: np.ndarray
    bending: str
    has_ground: bool
    n_frames: int


def grid(n: int, size: float):
    """Flat n x n grid in the xz plane at y = 0, as CGQ's cloth_drape.py builds it."""
    verts = np.array(
        [[(i / n - 0.5) * size, 0.0, (j / n - 0.5) * size] for j in range(n + 1) for i in range(n + 1)],
        dtype=np.float64,
    )
    tris = []
    for j in range(n):
        for i in range(n):
            v00 = j * (n + 1) + i
            tris += [[v00, v00 + 1, v00 + n + 1], [v00 + 1, v00 + n + 2, v00 + n + 1]]
    return verts, np.array(tris, dtype=np.int32)


def make_scenes(n: int) -> list[ClothScene]:
    verts, tris = grid(n, 1.0)
    drape = verts + np.array([0.0, 2.0, 0.0])
    flat = verts + np.array([0.0, 0.3, 0.0])
    angle = np.deg2rad(30.0)
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]])
    tilted = verts @ rotation.T + np.array([0.0, 0.4, 0.0])
    pinned = np.zeros(len(verts), dtype=bool)
    pinned[[0, n]] = True
    no_pin = np.zeros(len(verts), dtype=bool)
    return [
        ClothScene("drape", drape, tris, pinned, "hinge", False, 150),
        ClothScene("ground_flat", flat, tris, no_pin, "quadratic", True, 120),
        ClothScene("ground_tilted", tilted, tris, no_pin, "quadratic", True, 120),
    ]
