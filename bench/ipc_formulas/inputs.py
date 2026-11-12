"""Input rows of the IPC formula parity, for every case of cases.py: random, degenerate and real sets.

    python bench/ipc_formulas/inputs.py [--scenes cloth_stack kimono] [--seed 0]

Writes <data>/inputs.npz (cases.DATA) with one array per case and set, named '<case>/<set>'. The sets are:
- random: random geometry in the unit box, barrier parameters spread over the whole active band;
- degenerate: exact dyadic coordinates (axis-aligned, coplanar, parallel, collinear, coincident, on the boundaries of
  the distance-type regions), band extremes (r below the Hessian clamp, r near 1, squared distance near xi^2), near
  parallel edges in the mollifier region, the three friction regimes, CCD pairs approaching, grazing, inside their
  thickness or at rest, screw motions with rotation;
- real_<scene>: the candidate and active stencils of T0's captured frames (bench/ipc_bvh/data/<scene>), their
  trajectories to the step end, their per-vertex d_hat and thickness; real_<scene>_mollified: every active edge-edge
  pair of those frames below the mollifier threshold;
- parallel, mollifier_sweep: edge-edge barrier rows of exactly parallel edges, and of I1 / eps_x swept over 1e-6..1;
- regression: edge-triangle rows of the coplanar side test (an edge parallel to a side, collinear overlaps, a 1e-8 m
  triangle).
The barrier parameters (d_hat, xi) of a row are set from the true distance of its closest features, so that the pair
lies inside the active band.
"""

import argparse

import numpy as np

from cases import CASES, CCD_ETA, DATA, HERE, MOLLIFIER_RATIO, STENCIL_VERTS

T0_DATA = HERE.parent / "ipc_bvh" / "data"


def dot(a, b):
    return np.einsum("...i,...i->...", a, b)


def norm(a):
    return np.sqrt(dot(a, a))


def unit(a):
    return a / norm(a)[..., None]


def log_uniform(rng, low, high, size):
    return np.exp(rng.uniform(np.log(low), np.log(high), size))


def rodrigues(rotvec, v):
    """Rotate the rows of v by the rotation vectors (exact formula, numpy)."""
    theta = norm(rotvec)
    with np.errstate(invalid="ignore", divide="ignore"):
        axis = np.where(theta[:, None] > 0.0, rotvec / theta[:, None], 0.0)
    cos, sin = np.cos(theta)[:, None], np.sin(theta)[:, None]
    return v * cos + np.cross(axis, v) * sin + axis * dot(axis, v)[:, None] * (1.0 - cos)


# --- True distances of the closest features (Ericson), to place the rows in the active band ------------------------


def closest_pt(p, a, b, c):
    ab, ac, ap, bp, cp = b - a, c - a, p - a, p - b, p - c
    d1, d2, d3, d4, d5, d6 = dot(ab, ap), dot(ac, ap), dot(ab, bp), dot(ac, bp), dot(ab, cp), dot(ac, cp)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    with np.errstate(all="ignore"):
        denom = va + vb + vc
        candidates = [
            ((d1 <= 0) & (d2 <= 0), a),
            ((d3 >= 0) & (d4 <= d3), b),
            ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + (d1 / (d1 - d3))[:, None] * ab),
            ((d6 >= 0) & (d5 <= d6), c),
            ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + (d2 / (d2 - d6))[:, None] * ac),
            (
                (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
                b + ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None] * (c - b),
            ),
        ]
        closest = a + ab * (vb / denom)[:, None] + ac * (vc / denom)[:, None]
    for condition, point in reversed(candidates):
        closest = np.where(condition[:, None], point, closest)
    return norm(p - closest)


def closest_ee(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f, c, b = dot(d1, d1), dot(d2, d2), dot(d2, r), dot(d1, r), dot(d1, d2)
    with np.errstate(all="ignore"):
        denom = a * e - b * b
        s = np.where(denom > 0, np.clip((b * f - c * e) / denom, 0, 1), 0.0)
        t = (b * s + f) / e
        s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s))
        t = np.clip(t, 0, 1)
    return norm(p1 + s[:, None] * d1 - p2 - t[:, None] * d2)


def closest_pe(p, e0, e1):
    edge = e1 - e0
    with np.errstate(all="ignore"):
        t = np.clip(dot(p - e0, edge) / dot(edge, edge), 0, 1)
    return norm(p - e0 - t[:, None] * edge)


def stencil_distance(kind, x):
    if kind == "pt":
        return closest_pt(x[:, 0], x[:, 1], x[:, 2], x[:, 3])
    if kind == "ee":
        return closest_ee(x[:, 0], x[:, 1], x[:, 2], x[:, 3])
    if kind == "pe":
        return closest_pe(x[:, 0], x[:, 1], x[:, 2])
    return norm(x[:, 0] - x[:, 1])


def band_parameters(rng, distance2, ratio=None, xi_fraction=None):
    """Thickness and activation distance placing each squared distance at the ratio r = d_tilde / (d_hat^2 + 2 d_hat xi)."""
    n = len(distance2)
    if xi_fraction is None:
        xi_fraction = np.where(rng.random(n) < 0.3, 0.0, rng.uniform(0.0, 0.95, n))
    if ratio is None:
        ratio = log_uniform(rng, 1e-3, 0.95, n)
    xi = xi_fraction * np.sqrt(distance2)
    d_tilde = distance2 - xi * xi
    d_hat = -xi + np.sqrt(xi * xi + d_tilde / ratio)
    return d_hat, xi


def extreme_band(rng, distance2):
    """Band parameters at the extremes: below the Hessian clamp, near the activation bound, near the thickness."""
    n = len(distance2)
    pick = rng.integers(0, 3, n)
    ratio = np.where(
        pick == 0, log_uniform(rng, 1e-10, 1e-5, n), np.where(pick == 1, 1.0 - log_uniform(rng, 1e-9, 1e-2, n), 0.3)
    )
    xi_fraction = np.where(
        pick == 2, np.sqrt(1.0 - log_uniform(rng, 1e-6, 1e-2, n)), np.where(rng.random(n) < 0.5, 0.0, 0.5)
    )
    return band_parameters(rng, distance2, ratio, xi_fraction)


def kappas(rng, n):
    return log_uniform(rng, 1e-1, 1e5, n)


def flat(x):
    return x.reshape(len(x), -1)


def near_parallel_for_friction(x):
    """Edge-edge pairs the friction pair filter of production leaves out (contact_kernels.cu)."""
    edge_a, edge_b = x[:, 1] - x[:, 0], x[:, 3] - x[:, 2]
    cross = np.cross(edge_a, edge_b)
    return dot(cross, cross) < 1e-3 * dot(edge_a, edge_a) * dot(edge_b, edge_b)


# --- Geometry ------------------------------------------------------------------------------------------------------


def random_stencils(rng, kind, n):
    n_v = sum(STENCIL_VERTS[kind])
    x = rng.uniform(-1.0, 1.0, (n, n_v, 3))
    if kind in ("pt", "pe"):
        x[:, 0] *= 1.5
    return x


def axis_frames():
    """The 48 signed axis permutations, exact."""
    frames = []
    for perm in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
        for signs in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3):
            frames.append(np.eye(3)[perm] * signs[:, None])
    return np.array(frames)


def exact_transform(rng, x):
    """Apply a random signed axis permutation and a dyadic translation, which keep coordinates exact."""
    frames = axis_frames()[rng.integers(0, 48, len(x))]
    shift = rng.integers(-8, 8, (len(x), 1, 3)) / 4.0
    return np.einsum("nij,nvj->nvi", frames, x) + shift


def degenerate_stencils(rng, kind, include_touching):
    """Exact dyadic configurations on the boundaries of the distance-type regions."""
    grid = np.arange(-2, 6) / 4.0
    heights = np.array([1.0, 2.0, 4.0, 8.0]) / 16.0
    if include_touching:
        heights = np.concatenate([[0.0], heights])
    rows = []
    if kind == "pt":
        tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        for x in grid:
            for y in grid:
                for z in heights:
                    for order in ([0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1]):
                        rows.append(np.vstack([[x, y, z], tri[order]]))
        if include_touching:
            # Zero-area triangles: collinear and coincident vertices
            for z in heights:
                rows.append(np.array([[0.25, 0.5, z], [0, 0, 0], [1, 0, 0], [2, 0, 0]]))
                rows.append(np.array([[0.25, 0.5, z], [0, 0, 0], [0, 0, 0], [1, 0, 0]]))
    elif kind == "ee":
        segment = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        directions = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, -1, 0], [-1, 0, 0]], dtype=float)
        for x in grid:
            for y in grid[::2]:
                for z in heights:
                    for d in directions:
                        start = np.array([x, y, z])
                        rows.append(np.vstack([segment, [start, start + 0.5 * d]]))
        if include_touching:
            # Collapsed edges
            for x in grid:
                rows.append(np.array([[x, 0, 0], [x, 0, 0], [0, 0, 0.25], [1, 0, 0.25]]))
                rows.append(np.array([[0, 0, 0], [1, 0, 0], [x, 0.25, 0], [x, 0.25, 0]]))
    elif kind == "pe":
        for x in grid:
            for y in grid[::2]:
                for z in heights:
                    rows.append(np.array([[x, y, z], [0, 0, 0], [1, 0, 0]]))
        if include_touching:
            for x in grid:
                rows.append(np.array([[x, 0.25, 0], [0.5, 0, 0], [0.5, 0, 0]]))
    else:
        for x in grid:
            for z in heights:
                rows.append(np.array([[x, 0.25, z], [0.0, 0.0, 0.0]]))
    x = np.array(rows)
    return np.concatenate([x, exact_transform(rng, x)])


def near_parallel_edges(rng, n):
    """Edge pairs at small angles and offsets, spanning the near-parallel guard and the mollifier region."""
    angle = 10.0 ** rng.uniform(-9, -1, n)
    angle[: n // 10] = 0.0
    ea0 = rng.uniform(-1, 1, (n, 3))
    direction = unit(rng.normal(size=(n, 3)))
    length_a = rng.uniform(0.2, 1.0, n)
    ea1 = ea0 + direction * length_a[:, None]
    normal = unit(np.cross(direction, rng.normal(size=(n, 3))))
    side = unit(np.cross(direction, normal))
    direction_b = direction * np.cos(angle)[:, None] + side * np.sin(angle)[:, None]
    offset = 10.0 ** rng.uniform(-3, -1, n)
    # Edge B overlaps edge A (interior-interior), sticks out of it (point-edge), or lies beyond it (point-point)
    along = rng.choice([0.2, 0.7, 1.3], n) * length_a
    eb0 = ea0 + direction * along[:, None] + normal * offset[:, None]
    eb1 = eb0 + direction_b * rng.uniform(0.2, 1.0, n)[:, None]
    return np.stack([ea0, ea1, eb0, eb1], axis=1)


# --- Sets ----------------------------------------------------------------------------------------------------------


class Builder:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.arrays = {}

    def add(self, case, name, rows):
        rows = np.ascontiguousarray(rows, dtype=np.float64)
        assert rows.ndim == 2 and rows.shape[1] == CASES[case].n_inputs, (case, rows.shape)
        if len(rows):
            key = f"{case}/{name}"
            self.arrays[key] = np.concatenate([self.arrays[key], rows]) if key in self.arrays else rows

    def distances(self, name, stencils):
        for kind, x in stencils.items():
            if kind == "pp":
                continue
            self.add(f"{kind}_distance", name, flat(x))
            self.add(f"closest_{kind}", name, flat(x))
            if kind == "ee":
                self.add("ee_distance_type", name, flat(x))
                self.add("mollifier", name, flat(x))

    def barriers(self, name, stencils, rest=None, extreme=False):
        rng = self.rng
        for kind, x in stencils.items():
            distance2 = stencil_distance(kind, x) ** 2
            keep = distance2 > 1e-20
            x, distance2 = x[keep], distance2[keep]
            d_hat, xi = extreme_band(rng, distance2) if extreme else band_parameters(rng, distance2)
            params = np.column_stack([d_hat, xi, kappas(rng, len(x))])
            if kind == "ee":
                rest_x = x if rest is None else rest[keep]
                self.add("barrier_ee", name, np.column_stack([flat(x), flat(rest_x), params]))
            else:
                self.add(f"barrier_{kind}", name, np.column_stack([flat(x), params]))

    def pairs(self, name, n):
        rng = self.rng
        d_hats = log_uniform(rng, 1e-4, 1e-1, (n, 2))
        thick = np.where(rng.random((n, 4)) < 0.3, 0.0, log_uniform(rng, 1e-5, 1e-2, (n, 4)))
        weights = log_uniform(rng, 1e-6, 1e-1, (n, 2))
        for kind in ("pt", "ee", "pe", "pp"):
            self.add(f"pair_{kind}", name, np.column_stack([d_hats, thick, weights]))

    def frictions(self, name, stencils):
        rng = self.rng
        for kind, (lagged, current) in stencils.items():
            n = len(lagged)
            if kind == "pt":
                normal = np.cross(lagged[:, 2] - lagged[:, 1], lagged[:, 3] - lagged[:, 1])
                distance2 = dot(lagged[:, 0] - lagged[:, 1], normal) ** 2 / dot(normal, normal)
            elif kind == "ee":
                normal = np.cross(lagged[:, 1] - lagged[:, 0], lagged[:, 3] - lagged[:, 2])
                distance2 = dot(lagged[:, 2] - lagged[:, 0], normal) ** 2 / dot(normal, normal)
            elif kind == "pe":
                cross = np.cross(lagged[:, 1] - lagged[:, 0], lagged[:, 2] - lagged[:, 0])
                distance2 = dot(cross, cross) / dot(lagged[:, 2] - lagged[:, 1], lagged[:, 2] - lagged[:, 1])
            else:
                distance2 = dot(lagged[:, 0] - lagged[:, 1], lagged[:, 0] - lagged[:, 1])
            keep = distance2 > 1e-20
            if kind == "ee":
                keep &= ~near_parallel_for_friction(lagged)
            lagged, distance2 = lagged[keep], distance2[keep]
            current = None if current is None else current[keep]
            n = len(lagged)
            d_hat, xi = band_parameters(rng, distance2)
            eps_vh = log_uniform(rng, 1e-5, 1e-1, n)
            if current is None:
                # Displacements around the smoothing length: sliding, sticking, at rest, rigid translation
                scale = log_uniform(rng, 1e-3, 1e3, n) * eps_vh
                current = lagged + rng.normal(size=lagged.shape) * scale[:, None, None]
                regime = rng.integers(0, 6, n)
                current[regime == 0] = lagged[regime == 0]
                translation = rng.normal(size=(n, 1, 3)) * scale[:, None, None]
                current[regime == 1] = (lagged + translation)[regime == 1]
            params = np.column_stack([d_hat, kappas(rng, n), rng.uniform(0.1, 1.0, n), eps_vh, xi])
            self.add(f"friction_{kind}", name, np.column_stack([flat(current), flat(lagged), params]))
            self.add(f"frame_{kind}", name, flat(lagged))

    def halfplanes(self, name, n, exact=False):
        rng = self.rng
        if exact:
            normal = axis_frames()[rng.integers(0, 48, n)][:, 2]
            plane_point = rng.integers(-4, 4, (n, 3)) / 4.0
            x = plane_point + normal * (rng.integers(1, 16, n) / 32.0)[:, None] + rng.integers(-4, 4, (n, 3)) / 8.0
        else:
            normal = unit(rng.normal(size=(n, 3)))
            plane_point = rng.uniform(-1, 1, (n, 3))
            x = rng.uniform(-1, 1, (n, 3))
        if not exact:
            # Reflect the points below the plane above it
            distance = dot(normal, x - plane_point)
            x = x - (distance - np.abs(distance))[:, None] * normal + (1e-3 * (distance == 0))[:, None] * normal
        distance = dot(normal, x - plane_point)
        keep = distance > 0
        x, normal, plane_point, distance = x[keep], normal[keep], plane_point[keep], distance[keep]
        n = len(x)
        d_hat, xi = band_parameters(rng, distance**2)
        self.add("barrier_ph", name, np.column_stack([x, plane_point, normal, d_hat, xi, kappas(rng, n)]))
        eps_vh = log_uniform(rng, 1e-5, 1e-1, n)
        lagged = x
        scale = log_uniform(rng, 1e-3, 1e3, n) * eps_vh
        current = lagged + rng.normal(size=(n, 3)) * scale[:, None]
        at_rest = rng.random(n) < 0.15
        current[at_rest] = lagged[at_rest]
        self.add(
            "friction_ph",
            name,
            np.column_stack(
                [current, lagged, plane_point, normal, d_hat, kappas(rng, n), rng.uniform(0.1, 1, n), eps_vh, xi]
            ),
        )
        self.add("frame_ph", name, normal)
        # Linear CCD: towards, away from and along the plane, some already inside the thickness
        displacement = rng.normal(size=(n, 3)) * distance[:, None] * rng.uniform(0.1, 3.0, n)[:, None]
        if exact:
            # Exactly tangent displacements, the normal being an axis
            along = rng.random(n) < 0.1
            displacement[along] -= normal[along] * dot(displacement[along], normal[along])[:, None]
        thickness = distance * np.where(rng.random(n) < 0.1, 1.5, rng.uniform(0.0, 0.9, n))
        eta = np.where(rng.random(n) < 0.5, CCD_ETA, rng.uniform(0.1, 0.9, n))
        self.add("ccd_ph", name, np.column_stack([x, displacement, plane_point, normal, eta, thickness]))
        # Broad-phase cull: trajectories crossing the band, staying beyond it, or within the inflation of it
        inflation = np.where(rng.random(n) < 0.5, 0.0, distance * rng.uniform(0.0, 0.5, n))
        cull_d_hat = distance * log_uniform(rng, 0.1, 2.0, n)
        self.add(
            "candidate_ph",
            name,
            np.column_stack([x, x + displacement, plane_point, normal, inflation, cull_d_hat, thickness]),
        )
        # Screw CCD of a point on a rotating body
        rotation = unit(rng.normal(size=(n, 3))) * log_uniform(rng, 1e-4, 3.0, n)[:, None]
        pivot = x + rng.normal(size=(n, 3)) * 0.3
        pivot_displacement = -normal * distance[:, None] * rng.uniform(0.0, 3.0, n)[:, None]
        chord = pivot + pivot_displacement + rodrigues(rotation, x - pivot) - x
        thickness = distance * np.where(rng.random(n) < 0.05, 1.5, rng.uniform(0.0, 0.9, n))
        self.add(
            "ccd_ph_screw",
            name,
            np.column_stack([x, chord, rotation, pivot, pivot_displacement, plane_point, normal, eta, thickness]),
        )

    def ccds(self, name, stencils, displacements=None, thickness=None, eta=None):
        rng = self.rng
        for kind, x in stencils.items():
            n_a, n_b = STENCIL_VERTS[kind]
            n = len(x)
            distance = stencil_distance(kind, x)
            if displacements is None:
                centroid_gap = x[:, n_a:].mean(axis=1) - x[:, :n_a].mean(axis=1)
                approach = rng.uniform(0.0, 1.5, (n, 1, 1))
                d = rng.normal(size=x.shape) * distance[:, None, None] * rng.uniform(0.0, 1.0, (n, 1, 1))
                d[:, :n_a] += approach * centroid_gap[:, None] * 0.5
                d[:, n_a:] -= approach * centroid_gap[:, None] * 0.5
                d[rng.random(n) < 0.05] = 0.0
            else:
                d = displacements[kind]
            xi = (
                distance * np.where(rng.random(n) < 0.05, 1.5, np.where(rng.random(n) < 0.3, 0.0, rng.uniform(0, 0.9, n)))
                if thickness is None
                else thickness[kind]
            )
            eta_k = np.where(rng.random(n) < 0.5, CCD_ETA, rng.uniform(0.1, 0.9, n)) if eta is None else eta
            rows = np.column_stack([flat(x), flat(d), eta_k, xi])
            self.add(f"ccd_additive_{kind}", name, rows)
            self.add(f"ccd_directional_{kind}", name, rows)

    def screws(self, name, stencils):
        rng = self.rng
        for kind, x in stencils.items():
            n_a, n_b = STENCIL_VERTS[kind]
            n = len(x)
            distance = stencil_distance(kind, x)
            motions = []
            chords = np.empty_like(x)
            for side, (lo, hi) in enumerate(((0, n_a), (n_a, n_a + n_b))):
                pick = rng.integers(0, 4, n)
                angle = np.where(pick == 0, 0.0, np.where(pick == 1, 1e-16, log_uniform(rng, 1e-3, 3.0, n)))
                rotation = unit(rng.normal(size=(n, 3))) * angle[:, None]
                pivot = x[:, lo:hi].mean(axis=1) + rng.normal(size=(n, 3)) * 0.2
                towards = x[:, n_a:].mean(axis=1) - x[:, :n_a].mean(axis=1)
                pivot_displacement = (1 - 2 * side) * towards * rng.uniform(0.0, 0.8, (n, 1)) + rng.normal(
                    size=(n, 3)
                ) * 0.05
                motions.append((rotation, pivot, pivot_displacement))
            # Same rigid motion on both sides for some rows
            is_same = rng.random(n) < 0.1
            for k in range(3):
                motions[1][k][is_same] = motions[0][k][is_same]
            for side, (lo, hi) in enumerate(((0, n_a), (n_a, n_a + n_b))):
                rotation, pivot, pivot_displacement = motions[side]
                for i_v in range(lo, hi):
                    rotated = pivot + pivot_displacement + rodrigues(rotation, x[:, i_v] - pivot)
                    linear = x[:, i_v] + pivot_displacement
                    is_linear = (dot(rotation, rotation) > 0)[:, None]
                    chords[:, i_v] = np.where(is_linear, rotated, linear) - x[:, i_v]
            motion = np.column_stack([np.column_stack(m) for m in motions])
            xi = distance * np.where(rng.random(n) < 0.05, 1.5, np.where(rng.random(n) < 0.3, 0.0, rng.uniform(0, 0.9, n)))
            eta = np.where(rng.random(n) < 0.5, CCD_ETA, rng.uniform(0.1, 0.9, n))
            base = np.column_stack([flat(x), flat(chords), motion, eta, xi, is_same])
            self.add(f"ccd_screw_{kind}", name, base)
            if kind in ("pt", "ee"):
                t_start = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.0, 0.9, n))
                t_end = np.where(rng.random(n) < 0.5, 1.0, rng.uniform(t_start, 1.0))
                max_iters = np.where(rng.random(n) < 0.5, 16, 50000)
                self.add(
                    f"ccd_screw_interval_{kind}",
                    name,
                    np.column_stack([base, t_start, t_end, distance, max_iters]),
                )

    def candidates(self, name, stencils, ends, inflations, speeds, thicknesses, d_hat, directional):
        for kind, x in stencils.items():
            self.add(
                f"candidate_{kind}",
                name,
                np.column_stack(
                    [flat(x), flat(ends[kind]), inflations[kind], speeds[kind], thicknesses[kind], d_hat[kind],
                     directional[kind]]
                ),
            )

    def random_candidates(self, name, stencils, exact=False):
        rng = self.rng
        ends, inflations, speeds, thicknesses, d_hat, directional = {}, {}, {}, {}, {}, {}
        for kind, x in stencils.items():
            n, n_v = x.shape[:2]
            distance = stencil_distance(kind, x)
            scale = distance[:, None, None] * rng.uniform(0.0, 1.0, (n, 1, 1))
            moves = rng.normal(size=x.shape) * scale if not exact else rng.integers(-2, 3, x.shape) / 8.0
            ends[kind] = x + moves
            inflations[kind] = np.where(rng.random((n, n_v)) < 0.7, 0.0, rng.uniform(0, 0.05, (n, n_v)))
            speeds[kind] = np.where(rng.random((n, n_v)) < 0.8, 0.0, rng.uniform(0, 0.5, (n, n_v)))
            thicknesses[kind] = np.where(rng.random((n, n_v)) < 0.3, 0.0, rng.uniform(0, 0.02, (n, n_v)))
            d_hat[kind] = distance * log_uniform(rng, 0.05, 2.0, n)
            directional[kind] = (rng.random(n) < 0.8).astype(float)
        self.candidates(name, stencils, ends, inflations, speeds, thicknesses, d_hat, directional)

    def intersections(self, name, n, exact=False):
        rng = self.rng
        if exact:
            tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            grid = np.arange(-2, 7) / 4.0
            rows = []
            for x0 in grid[::2]:
                for y0 in grid[::2]:
                    for x1 in grid[1::2]:
                        for z0, z1 in ((0.0, 0.0), (-0.25, 0.25), (0.0, 0.5), (0.25, 0.5)):
                            rows.append(np.vstack([tri, [[x0, y0, z0], [x1, 1.0 - x1, z1]]]))
            x = np.array(rows)
            x = np.concatenate([x, exact_transform(rng, x)])
        else:
            tri = rng.uniform(-1, 1, (n, 3, 3))
            inside = tri.mean(axis=1) + rng.normal(size=(n, 3)) * 0.3
            direction = rng.normal(size=(n, 3))
            e0 = inside - direction * rng.uniform(0, 1, (n, 1))
            e1 = inside + direction * rng.uniform(0, 1, (n, 1))
            x = np.concatenate([tri, e0[:, None], e1[:, None]], axis=1)
        self.add("intersect", name, flat(x))

    def scalars(self, name, n, extreme):
        rng = self.rng
        distance2 = log_uniform(rng, 1e-8, 1.0, n)
        d_hat, xi = extreme_band(rng, distance2) if extreme else band_parameters(rng, distance2)
        if extreme:
            xi_fraction = np.sqrt(1.0 - log_uniform(rng, 1e-12, 1e-6, n))
            d_hat2, xi2 = band_parameters(rng, distance2, xi_fraction=xi_fraction)
            replace = rng.random(n) < 0.3
            d_hat, xi = np.where(replace, d_hat2, d_hat), np.where(replace, xi2, xi)
        kappa = kappas(rng, n)
        self.add("barrier", name, np.column_stack([distance2, d_hat, xi, kappa]))
        threshold = log_uniform(rng, 1e-6, 1.0, n)
        cross_norm2 = threshold * np.where(rng.random(n) < 0.1, 0.0, rng.uniform(0, 1, n))
        self.add("mollified_barrier", name, np.column_stack([distance2, cross_norm2, threshold, d_hat, xi, kappa]))


def build_random(builder, n):
    rng = builder.rng
    stencils = {kind: random_stencils(rng, kind, n) for kind in STENCIL_VERTS}
    builder.distances("random", stencils)
    builder.barriers("random", stencils)
    builder.pairs("random", n)
    builder.scalars("random", n, extreme=False)
    lagged = {kind: random_stencils(rng, kind, n) for kind in STENCIL_VERTS}
    builder.frictions("random", {kind: (x, None) for kind, x in lagged.items()})
    builder.halfplanes("random", n)
    builder.ccds("random", stencils)
    builder.screws("random", stencils)
    builder.random_candidates("random", stencils)
    builder.intersections("random", n)


def build_degenerate(builder):
    rng = builder.rng
    touching = {kind: degenerate_stencils(rng, kind, include_touching=True) for kind in STENCIL_VERTS}
    separated = {kind: degenerate_stencils(rng, kind, include_touching=False) for kind in STENCIL_VERTS}
    parallel = near_parallel_edges(rng, 2000)
    builder.distances("degenerate", touching)
    builder.distances("degenerate", {"ee": parallel})
    builder.barriers("degenerate", separated, extreme=True)
    builder.barriers("degenerate", {"ee": parallel})
    # Rest edges longer than the current ones, which puts more pairs below the mollifier threshold
    builder.barriers("degenerate", {"ee": parallel}, rest=parallel * 1.5)
    builder.scalars("degenerate", 4000, extreme=True)
    builder.frictions("degenerate", {kind: (x, None) for kind, x in separated.items()})
    builder.frictions("degenerate", {"ee": (parallel, None)})
    builder.halfplanes("degenerate", 2000, exact=True)
    builder.ccds("degenerate", separated)
    builder.ccds("degenerate", {"ee": parallel})
    # Grazing: every vertex slides along the same direction, tangentially, at a small gap above the thickness
    for kind, x in separated.items():
        n = len(x)
        distance = stencil_distance(kind, x)
        slide = np.broadcast_to(rng.normal(size=(n, 1, 3)), x.shape).copy()
        builder.ccds(
            "degenerate",
            {kind: x},
            displacements={kind: slide + rng.normal(size=x.shape) * 1e-3},
            thickness={kind: distance * (1.0 - log_uniform(rng, 1e-4, 1e-1, n))},
        )
    builder.screws("degenerate", separated)
    builder.random_candidates("degenerate", separated, exact=True)
    builder.intersections("degenerate", 0, exact=True)


def build_mollified(builder, n):
    """Edge pairs of the mollified edge-edge barrier: exactly parallel, then I1 / eps_x swept over 1e-6..1."""
    rng = builder.rng
    # Exactly parallel edges in exact dyadic coordinates: overlapping, sticking out of each other and apart, at every
    # offset of the grid, then the same pairs rotated (parallel to the rounding of the rotation)
    rows = []
    for start in np.arange(-3, 6) / 4.0:
        for length in (0.25, 0.5, 1.0, -0.5):
            for offset_y, offset_z in ((1 / 16, 0.0), (0.0, 1 / 8), (1 / 32, 1 / 16), (1 / 4, 1 / 4)):
                eb0 = np.array([start, offset_y, offset_z])
                rows.append(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], eb0, eb0 + [length, 0.0, 0.0]]))
    x = np.array(rows)
    x = np.concatenate([x, exact_transform(rng, x)])
    rotation = unit(rng.normal(size=(len(x), 3))) * rng.uniform(0.1, 3.0, (len(x), 1))
    rotated = np.stack([rodrigues(rotation, x[:, k]) for k in range(4)], axis=1)
    builder.barriers("parallel", {"ee": np.concatenate([x, rotated])})
    # I1 / eps_x log-uniform over 1e-6..1 for rest edges equal to the current ones, where I1 / eps_x is
    # sin(angle)^2 / MOLLIFIER_RATIO
    ratio = log_uniform(rng, 1e-6, 1.0, n)
    angle = np.arcsin(np.sqrt(MOLLIFIER_RATIO * ratio))
    ea0 = rng.uniform(-1, 1, (n, 3))
    direction = unit(rng.normal(size=(n, 3)))
    length_a = rng.uniform(0.2, 1.0, n)
    ea1 = ea0 + direction * length_a[:, None]
    normal = unit(np.cross(direction, rng.normal(size=(n, 3))))
    side = unit(np.cross(direction, normal))
    direction_b = direction * np.cos(angle)[:, None] + side * np.sin(angle)[:, None]
    along = rng.choice([0.2, 0.7, 1.3], n) * length_a
    eb0 = ea0 + direction * along[:, None] + normal * (10.0 ** rng.uniform(-3, -1, n))[:, None]
    eb1 = eb0 + direction_b * rng.uniform(0.2, 1.0, n)[:, None]
    builder.barriers("mollifier_sweep", {"ee": np.stack([ea0, ea1, eb0, eb1], axis=1)})


def build_intersection_regressions(builder):
    """Edge-triangle rows of the coplanar side test, as the CGQ test of the fix has them (no random draw)."""
    cloth_triangle = [
        [-0.03750000000250894, 0.010796496484808772, -5.6792833359554066e-11],
        [3.702922774809752e-18, 0.01079649648479572, -6.91262025920539e-19],
        [-0.03749999994495807, 0.010796496484828983, 0.03749999994495807],
    ]
    unit_triangle = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    small_triangle = [[0.0, 0.0, 0.0], [1e-8, 0.0, 0.0], [0.0, 1e-8, 0.0]]
    rows = [
        # box_on_cloth (CPU, frame 8): edge [145, 161] parallel to the side 144-160 of triangle [143, 144, 160], 2.65 cm
        # away, tilted out of the plane by 6.3e-13 inside the coplanar band, then the same edge tilted by 1.6e-12
        cloth_triangle
        + [[0.03750000000250894, 0.010796496484808772, 5.679283242788697e-11],
           [5.6792832912178434e-11, 0.010796496484808772, 0.037500000002508936]],
        cloth_triangle
        + [[0.03750000000250894, 0.010796496484825094, 5.6792832427886965e-11],
           [5.679283291217842e-11, 0.010796496484773505, 0.037500000002508936]],
        # Exactly parallel to a side outside the triangle, in its plane and tilted inside the coplanar band, then
        # collinear overlaps of the side t0-t1
        unit_triangle + [[1.0, 0.25, 0.0], [0.25, 1.0, 0.0]],
        unit_triangle + [[1.0, 0.25, -1e-12], [0.25, 1.0, 1e-12]],
        unit_triangle + [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
        unit_triangle + [[-0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
        # A triangle of 1e-8 m sides, crossed and missed
        small_triangle + [[2.5e-9, 2.5e-9, -1e-8], [2.5e-9, 2.5e-9, 1e-8]],
        small_triangle + [[1e-8, 1e-8, -1e-8], [1e-8, 1e-8, 1e-8]],
    ]
    builder.add("intersect", "regression", flat(np.array(rows)))


def gather_feature(stencils, flags, kind):
    """The point-edge or point-point sub-stencil of each PT or EE stencil, from its distance flag."""
    rows_pe, rows_pp = [], []
    for x, flag in zip(stencils, flags):
        active = [i for i in range(4) if flag >> i & 1]
        if len(active) == 3:
            if kind == "ee" and (flag & 0x3) == 0x3:
                point = [i for i in (2, 3) if flag >> i & 1][0]
                rows_pe.append(x[[point, 0, 1]])
            elif kind == "ee":
                point = [i for i in (0, 1) if flag >> i & 1][0]
                rows_pe.append(x[[point, 2, 3]])
            else:
                rows_pe.append(x[active])
        elif len(active) == 2:
            rows_pp.append(x[active])
    return np.array(rows_pe).reshape(-1, 3, 3), np.array(rows_pp).reshape(-1, 2, 3)


def build_real(builder, scene, cap):
    rng = builder.rng
    name = f"real_{scene}"
    with np.load(T0_DATA / scene / "frames.npz") as f, np.load(T0_DATA / scene / "gt.npz") as g:
        frames = {k: f[k] for k in f.files}
        gt = {k: g[k] for k in g.files}
    d_hats, thick = frames["d_hats"], frames["thicknesses"]
    kappa = float(np.median(frames["contact_kappa_table"][frames["contact_kappa_table"] > 0]))
    rest = frames["positions"][0]

    def frame_rows(key):
        offsets = gt[f"{key}_offsets"]
        return [(i_f, gt[key][offsets[i_f] : offsets[i_f + 1]]) for i_f in range(len(offsets) - 1)]

    def sample(n_rows):
        return np.sort(rng.choice(n_rows, size=min(cap, n_rows), replace=False))

    for kind in ("pt", "ee"):
        stencil_rows = frame_rows(f"{kind}_stencil")
        flag_rows = dict(frame_rows(f"{kind}_flag"))
        active_rows = dict(frame_rows(f"{kind}_active"))
        stencils, ends, flags, actives, frame_ids = [], [], [], [], []
        for i_f, stencil in stencil_rows:
            stencils.append(frames["positions"][i_f][stencil])
            ends.append(frames["trajectory_end_positions"][i_f][stencil])
            flags.append(flag_rows[i_f])
            actives.append(active_rows[i_f].astype(bool))
            frame_ids.append(np.full(len(stencil), i_f))
        all_stencil = np.concatenate([s for _, s in stencil_rows])
        x, end, flag, active = map(np.concatenate, (stencils, ends, flags, actives))
        frame_id = np.concatenate(frame_ids)
        if kind == "pt":
            d_hat = (d_hats[all_stencil[:, 0]] + d_hats[all_stencil[:, 1]]) / 2.0
            xi = thick[all_stencil[:, 0]] + thick[all_stencil[:, 1:]].max(axis=1)
            thick_rows = thick[all_stencil]
        else:
            d_hat = (d_hats[all_stencil[:, 0]] + d_hats[all_stencil[:, 2]]) / 2.0
            xi = thick[all_stencil[:, :2]].max(axis=1) + thick[all_stencil[:, 2:]].max(axis=1)
            thick_rows = thick[all_stencil]

        pick = sample(len(x))
        builder.distances(name, {kind: x[pick]})
        pick = sample(len(x))
        builder.ccds(
            name,
            {kind: x[pick]},
            displacements={kind: end[pick] - x[pick]},
            thickness={kind: xi[pick]},
            eta=np.full(len(pick), CCD_ETA),
        )
        pick = sample(len(x))
        inflation = frames["path_inflation"][frame_id[pick, None], all_stencil[pick]]
        speed = frames["path_speed"][frame_id[pick, None], all_stencil[pick]]
        builder.candidates(
            name,
            {kind: x[pick]},
            {kind: end[pick]},
            {kind: inflation},
            {kind: speed},
            {kind: thick_rows[pick]},
            {kind: d_hat[pick]},
            {kind: (rng.random(len(pick)) < 0.8).astype(float)},
        )

        # Active pairs: barrier, friction from the lagged step-start positions to the trajectory end
        act = np.flatnonzero(active)
        act = act[sample(len(act))]
        params = np.column_stack([d_hat[act], xi[act], np.full(len(act), kappa)])
        if kind == "pt":
            builder.add("barrier_pt", name, np.column_stack([flat(x[act]), params]))
        else:
            builder.add("barrier_ee", name, np.column_stack([flat(x[act]), flat(rest[all_stencil[act]]), params]))
            # Every active pair below the mollifier threshold, the first 'cap' of them, without drawing from the
            # generator so that the sets after this one stay the same
            rest_ee = rest[all_stencil]
            threshold = MOLLIFIER_RATIO * dot(rest_ee[:, 1] - rest_ee[:, 0], rest_ee[:, 1] - rest_ee[:, 0]) * dot(
                rest_ee[:, 3] - rest_ee[:, 2], rest_ee[:, 3] - rest_ee[:, 2]
            )
            cross = np.cross(x[:, 1] - x[:, 0], x[:, 3] - x[:, 2])
            mollified = np.flatnonzero(active & (dot(cross, cross) < threshold))[:cap]
            builder.add(
                "barrier_ee",
                f"{name}_mollified",
                np.column_stack(
                    [flat(x[mollified]), flat(rest_ee[mollified]), d_hat[mollified], xi[mollified],
                     np.full(len(mollified), kappa)]
                ),
            )
        full = act[flag[act] == 0xF]
        if kind == "ee":
            full = full[~near_parallel_for_friction(x[full])]
        friction_params = np.column_stack(
            [d_hat[full], np.full(len(full), kappa), np.full(len(full), 0.5), np.full(len(full), 1e-4), xi[full]]
        )
        builder.add(f"friction_{kind}", name, np.column_stack([flat(end[full]), flat(x[full]), friction_params]))
        builder.add(f"frame_{kind}", name, flat(x[full]))
        lagged_pe, lagged_pp = gather_feature(x[act], flag[act], kind)
        current_pe, current_pp = gather_feature(end[act], flag[act], kind)
        is_pe = np.array([bin(int(f)).count("1") == 3 for f in flag[act]], dtype=bool)
        is_pp = np.array([bin(int(f)).count("1") == 2 for f in flag[act]], dtype=bool)
        for sub_kind, lagged, current, mask in (
            ("pe", lagged_pe, current_pe, is_pe),
            ("pp", lagged_pp, current_pp, is_pp),
        ):
            if not len(lagged):
                continue
            sub_params = np.column_stack([d_hat[act][mask], xi[act][mask], np.full(mask.sum(), kappa)])
            builder.add(f"barrier_{sub_kind}", name, np.column_stack([flat(lagged), sub_params]))
            if sub_kind == "pe":
                builder.add("pe_distance", name, flat(lagged))
            friction_sub = np.column_stack(
                [sub_params[:, 0], sub_params[:, 2], np.full(mask.sum(), 0.5), np.full(mask.sum(), 1e-4), sub_params[:, 1]]
            )
            builder.add(f"friction_{sub_kind}", name, np.column_stack([flat(current), flat(lagged), friction_sub]))
            builder.add(f"frame_{sub_kind}", name, flat(lagged))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="*", default=["cloth_stack", "kimono"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-random", type=int, default=4000)
    parser.add_argument("--cap", type=int, default=30000)
    args = parser.parse_args()
    builder = Builder(args.seed)
    build_random(builder, args.n_random)
    build_degenerate(builder)
    for scene in args.scenes:
        build_real(builder, scene, args.cap)
    build_mollified(builder, args.n_random)
    build_intersection_regressions(builder)
    missing = {case for case in CASES} - {key.split("/")[0] for key in builder.arrays}
    assert not missing, missing
    path = DATA / "inputs.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **builder.arrays)
    for key, rows in builder.arrays.items():
        print(f"{key}: {rows.shape[0]} rows")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
