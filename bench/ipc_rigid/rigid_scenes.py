"""The rigid scenes of the parity of Genesis with CGQ, shared by both sides, in the frame and with the cloth of
bench/ipc_cloth/scenes.py (y up, dt = 0.01, one substep per frame, CGQ's Cloth).

The rigid geometry of a scene is the mesh of the rigid contact proxies Genesis builds for it, which the Genesis side
dumps once (genesis_side.py --dump-mesh) and the CGQ side loads, so that both sides see the same vertices and triangles.
The contact vertices are the vertices of the cloths, then those of the rigid meshes, on both sides once CGQ's global
vertices are mapped by their positions at the first frame. A free box carries the mass properties of a solid box of its
density, both sides taking them from `BoxSpec.mass_properties`.

cloth_on_box: a sheet dropped onto a box fixed in the world, without ground (parity case 1: no rigid dofs).
box_on_cloth: a free box of three different sides, tilted about two axes, dropped onto a sheet lying on the ground, so
    that it lands on a corner and rotates (parity case 2: the dofs of a free joint).
box_spin: the box of box_on_cloth spinning in the air at the angular velocity it reaches in its impact, a sheet lying on
    the ground far from it, without contact (the rotation of a free body alone).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "ipc_cloth"))
from scenes import (  # noqa: E402
    BENDING_YOUNGS_MODULUS,
    D_HAT,
    DT,
    GRAVITY,
    GROUND_NORMAL,
    GROUND_POS,
    MASS_DENSITY,
    SHEAR_MODULUS,
    THICKNESS,
    VELOCITY_TOL,
    YOUNGS_MODULUS,
    ClothSpec,
    grid,
)

__all__ = [
    "BENDING_YOUNGS_MODULUS",
    "D_HAT",
    "DT",
    "GRAVITY",
    "GROUND_NORMAL",
    "GROUND_POS",
    "MASS_DENSITY",
    "SHEAR_MODULUS",
    "THICKNESS",
    "VELOCITY_TOL",
    "YOUNGS_MODULUS",
    "BoxSpec",
    "ClothSpec",
    "RigidScene",
    "make_scenes",
]


def quat_to_R(quat) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass
class BoxSpec:
    size: tuple
    pos: tuple
    is_fixed: bool
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    density: float = 1000.0
    # The initial angular velocity in the world
    omega: tuple = (0.0, 0.0, 0.0)

    def mass_properties(self):
        """The mass, the center of mass and the inertia about it in the world at the first frame of a solid box."""
        a, b, c = self.size
        mass = self.density * a * b * c
        inertia_local = mass / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
        rotation = quat_to_R(self.quat)
        return mass, np.array(self.pos, dtype=np.float64), rotation @ inertia_local @ rotation.T


@dataclass
class RigidScene:
    name: str
    cloths: list[ClothSpec]
    boxes: list[BoxSpec]
    bending: str
    has_ground: bool
    n_frames: int

    @property
    def verts(self) -> np.ndarray:
        return np.concatenate([cloth.verts for cloth in self.cloths])


def make_scenes(n: int) -> list[RigidScene]:
    verts, tris = grid(n, 0.6)
    no_pin = np.zeros(len(verts), dtype=bool)
    half_angle = 0.5 * np.deg2rad(25.0)
    axis = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    tilt = (np.cos(half_angle), *(np.sin(half_angle) * axis))
    return [
        RigidScene(
            "cloth_on_box",
            [ClothSpec(verts + np.array([0.0, 0.35, 0.0]), tris, no_pin)],
            [BoxSpec((0.3, 0.3, 0.3), (0.0, 0.15, 0.0), True)],
            "quadratic",
            False,
            120,
        ),
        RigidScene(
            "box_on_cloth",
            [ClothSpec(verts + np.array([0.0, 0.02, 0.0]), tris, no_pin)],
            [BoxSpec((0.2, 0.12, 0.16), (0.0, 0.22, 0.0), False, tilt, 500.0)],
            "quadratic",
            True,
            150,
        ),
        RigidScene(
            "box_spin",
            [ClothSpec(verts + np.array([1.5, 0.02, 0.0]), tris, no_pin)],
            [BoxSpec((0.2, 0.12, 0.16), (0.0, 1.0, 0.0), False, tilt, 500.0, (-9.0, -0.3, -8.7))],
            "quadratic",
            True,
            30,
        ),
    ]
