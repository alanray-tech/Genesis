"""The rigid scenes of the parity of Genesis with CGQ, shared by both sides, in the frame and with the cloth of
bench/ipc_cloth/scenes.py (y up, dt = 0.01, one substep per frame, CGQ's Cloth).

The rigid geometry of a scene is the mesh of the rigid contact proxies Genesis builds for it, which the Genesis side
dumps once (genesis_side.py --dump-mesh) and the CGQ side loads, so that both sides see the same vertices and triangles.
The contact vertices are the vertices of the cloths, then those of the rigid meshes, on both sides once CGQ's global
vertices are mapped by their positions at the first frame.

cloth_on_box: a sheet dropped onto a box fixed in the world, without ground (parity case 1: no rigid dofs).
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
    "MASS_DENSITY",
    "SHEAR_MODULUS",
    "THICKNESS",
    "VELOCITY_TOL",
    "YOUNGS_MODULUS",
    "ClothSpec",
    "RigidScene",
    "make_scenes",
]


@dataclass
class BoxSpec:
    size: tuple
    pos: tuple
    is_fixed: bool


@dataclass
class RigidScene:
    name: str
    cloths: list[ClothSpec]
    boxes: list[BoxSpec]
    bending: str
    n_frames: int

    @property
    def verts(self) -> np.ndarray:
        return np.concatenate([cloth.verts for cloth in self.cloths])


def make_scenes(n: int) -> list[RigidScene]:
    verts, tris = grid(n, 0.6)
    no_pin = np.zeros(len(verts), dtype=bool)
    return [
        RigidScene(
            "cloth_on_box",
            [ClothSpec(verts + np.array([0.0, 0.35, 0.0]), tris, no_pin)],
            [BoxSpec((0.3, 0.3, 0.3), (0.0, 0.15, 0.0), True)],
            "quadratic",
            120,
        )
    ]
