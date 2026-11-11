"""The cloth scenes of the parity of Genesis with CGQ, shared by both sides.

Every scene runs in CGQ's frame, y up, gravity (0, -9.8, 0), dt = 0.01, one substep per frame. The membrane is CGQ's
Cloth (Baraff-Witkin with strain limiting), E = 2e4, G = 2e3, rho = 200, one-sided thickness 1e-3 (cloth_drape.py).
The vertices of the cloths of a scene follow one another in the order of the scene, on both sides.

The multi-cloth scenes are those of Genesis-ipc-ref-pr6/examples/newton_coupling/multilayer_cloth_benchmark.py
(crossed_drop, multilayer) at a coarser resolution for the CPU runs, `resolution` scaling their subdivisions, and a
sheet folded in two, whose upper layer falls onto its lower one once that one lands (self_fold).
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
class ClothSpec:
    verts: np.ndarray
    tris: np.ndarray
    pinned: np.ndarray


@dataclass
class ClothScene:
    name: str
    cloths: list[ClothSpec]
    bending: str
    has_ground: bool
    n_frames: int

    @property
    def verts(self) -> np.ndarray:
        return np.concatenate([cloth.verts for cloth in self.cloths])


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


def sheet(n_u: int, n_v: int, size_u: float, size_v: float, center, axis_u, axis_v) -> ClothSpec:
    """A rectangular sheet of n_u x n_v squares spanning axis_u and axis_v around its center, as make_cloth_grid of the
    reference benchmark lays out its vertices and triangles."""
    center, axis_u, axis_v = (np.asarray(a, dtype=np.float64) for a in (center, axis_u, axis_v))
    verts = np.array(
        [
            center + (column / n_u - 0.5) * size_u * axis_u + (row / n_v - 0.5) * size_v * axis_v
            for row in range(n_v + 1)
            for column in range(n_u + 1)
        ]
    )
    tris = []
    for row in range(n_v):
        for column in range(n_u):
            lower_left = row * (n_u + 1) + column
            upper_left = lower_left + n_u + 1
            tris += [[lower_left, lower_left + 1, upper_left], [lower_left + 1, upper_left + 1, upper_left]]
    return ClothSpec(verts, np.array(tris, dtype=np.int32), np.zeros(len(verts), dtype=bool))


def folded_sheet(n_length: int, n_width: int, flat_length: float, radius: float, width: float, height: float):
    """A sheet folded in two around the z axis: a lower layer at `height`, a half circle of `radius`, and an upper layer
    back above it, `n_length` squares along its length and `n_width` across."""
    length = 2.0 * flat_length + np.pi * radius
    path = []
    for s in np.linspace(0.0, length, n_length + 1):
        if s <= flat_length:
            path.append((s - flat_length, height))
        elif s <= flat_length + np.pi * radius:
            angle = (s - flat_length) / radius - 0.5 * np.pi
            path.append((radius * np.cos(angle), height + radius + radius * np.sin(angle)))
        else:
            path.append((flat_length + np.pi * radius - s, height + 2.0 * radius))
    verts = np.array(
        [[x, y, (row / n_width - 0.5) * width] for row in range(n_width + 1) for x, y in path], dtype=np.float64
    )
    flat = sheet(n_length, n_width, 1.0, 1.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    return ClothSpec(verts, flat.tris, flat.pinned)


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
        ClothScene("drape", [ClothSpec(drape, tris, pinned)], "hinge", False, 150),
        ClothScene("ground_flat", [ClothSpec(flat, tris, no_pin)], "quadratic", True, 120),
        ClothScene("ground_tilted", [ClothSpec(tilted, tris, no_pin)], "quadratic", True, 120),
    ]


def make_contact_scenes(resolution: int) -> list[ClothScene]:
    """The scenes of the contact between cloths, `resolution` subdivisions for the smallest side of a sheet."""
    r = resolution
    crossed = [
        sheet(r, r, 0.38, 0.38, (0.0, 0.23, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        sheet(r, r, 0.38, 0.38, (0.0, 0.73, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ]
    multilayer = [
        sheet(4 * r // 2, 4 * r // 2, 0.5, 0.5, (0.0, 0.10, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        sheet(3 * r // 2, 3 * r // 2, 0.3, 0.3, (0.0, 0.14, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        sheet(r, r, 0.2, 0.2, (0.0, 0.16, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        sheet(r // 2, r // 2, 0.1, 0.1, (0.0, 0.18, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ]
    self_fold = [folded_sheet(4 * r, r, 0.3, 0.03, 0.3, 0.1)]
    return [
        ClothScene("crossed_drop", crossed, "quadratic", True, 200),
        ClothScene("multilayer", multilayer, "quadratic", True, 150),
        ClothScene("self_fold", self_fold, "quadratic", True, 200),
    ]
