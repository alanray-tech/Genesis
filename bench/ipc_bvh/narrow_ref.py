"""Reference port of CGQ's narrow phase: distance classification and activation of PT, EE and PH candidates.

Ported from cuda-graph-qipc `contact_function/distance_flag.h` (pt_distance_flag, ee_distance_flag, the flagged
squared distances), `pair_d_hat.h`, `codim_thickness.h` and the activation test of the export / filter kernels
(`!(d2 >= (d_hat + xi)^2 || d2 <= xi^2)`). Every operation is one torch float64 op, rounded on its own, which is the
IEEE round-to-nearest arithmetic CGQ's non-contractible classifiers (`nofma_*`) use, so the distance flags are
bitwise those of CGQ. The squared distances CGQ computes with Eigen may be contracted into FMAs, so they agree to
rounding, and the activation of a pair could only differ for a squared distance within ulps of a threshold.

Candidates are global vertex stencils: PT (point, t0, t1, t2), EE (ea0, ea1, eb0, eb1) in the orientation CGQ emits,
PH (vertex, plane).
"""

from __future__ import annotations

import torch


def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]) + a[:, 2] * b[:, 2]


def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1], a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2], a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]),
        dim=1,
    )


def det3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return dot(a, cross(b, c))


def pt_distance_flag(p, t0, t1, t2) -> torch.Tensor:
    n_vec = cross(t1 - t0, t2 - t0)
    params = []
    for origin, target in ((t0, t1), (t1, t2), (t2, t0)):
        basis0 = target - origin
        basis1 = cross(basis0, n_vec)
        basis2 = p - origin
        det = det3(basis0, basis1, n_vec)
        params.append((det3(basis2, basis1, n_vec) / det, det3(basis0, basis2, n_vec) / det))
    (p0x, p0y), (p1x, p1y), (p2x, p2y) = params
    flag = torch.full_like(p0x, 0xF, dtype=torch.int32)
    # Assigned from the last case of the classifier to the first, so that the first matching case wins
    flag = torch.where((p2x <= 0.0) & (p1x >= 1.0), 0x1 | 0x8, flag)
    flag = torch.where((p1x <= 0.0) & (p0x >= 1.0), 0x1 | 0x4, flag)
    flag = torch.where((p0x <= 0.0) & (p2x >= 1.0), 0x1 | 0x2, flag)
    flag = torch.where((p2x > 0.0) & (p2x < 1.0) & (p2y >= 0.0), 0x1 | 0x8 | 0x2, flag)
    flag = torch.where((p1x > 0.0) & (p1x < 1.0) & (p1y >= 0.0), 0x1 | 0x4 | 0x8, flag)
    flag = torch.where((p0x > 0.0) & (p0x < 1.0) & (p0y >= 0.0), 0x1 | 0x2 | 0x4, flag)
    return flag


def ee_distance_flag(ea0, ea1, eb0, eb1) -> torch.Tensor:
    u, v, w = ea1 - ea0, eb1 - eb0, ea0 - eb0
    a, b, c, d, e = dot(u, u), dot(u, v), dot(v, v), dot(u, w), dot(v, w)
    big_d = a * c - b * b
    s_n = b * e - c * d
    is_near_parallel = (a > 0.0) & (c > 0.0) & (big_d <= 1.0e-6 * (a * c))
    is_low = is_near_parallel | (s_n <= 0.0)
    is_high = ~is_low & (s_n >= big_d)
    t_n = torch.where(is_low, e, torch.where(is_high, e + b, a * e - b * d))
    t_d = torch.where(is_low | is_high, c, big_d)
    default = torch.where(is_low, 0x1 | 0x4 | 0x8, torch.where(is_high, 0x2 | 0x4 | 0x8, 0xF)).to(torch.int32)
    s_lo, s_hi = -d, -d + b
    flag_lo = torch.where(s_lo <= 0.0, 0x1 | 0x4, torch.where(s_lo >= a, 0x2 | 0x4, 0x1 | 0x2 | 0x4))
    flag_hi = torch.where(s_hi <= 0.0, 0x1 | 0x8, torch.where(s_hi >= a, 0x2 | 0x8, 0x1 | 0x2 | 0x8))
    return torch.where(t_n <= 0.0, flag_lo, torch.where(t_n >= t_d, flag_hi, default)).to(torch.int32)


def sq_norm(a: torch.Tensor) -> torch.Tensor:
    return dot(a, a)


def pp_distance2(a, b) -> torch.Tensor:
    return sq_norm(a - b)


def pe_distance2(p, e0, e1) -> torch.Tensor:
    return sq_norm(cross(e0 - p, e1 - p)) / sq_norm(e1 - e0)


def pt_distance2(p, t0, t1, t2) -> torch.Tensor:
    b = cross(t1 - t0, t2 - t0)
    a_t_b = dot(p - t0, b)
    return a_t_b * a_t_b / sq_norm(b)


def ee_distance2(ea0, ea1, eb0, eb1) -> torch.Tensor:
    b = cross(ea1 - ea0, eb1 - eb0)
    a_t_b = dot(eb0 - ea0, b)
    return a_t_b * a_t_b / sq_norm(b)


def set_bits(flag: torch.Tensor) -> torch.Tensor:
    """The indices of the set bits of each 4-bit flag, lowest first, padded with 0."""
    bits = (flag[:, None] >> torch.arange(4, device=flag.device)) & 1
    order = torch.argsort(1 - bits, dim=1, stable=True)
    return order


def flagged_distance2(flag: torch.Tensor, verts: torch.Tensor, is_ee: bool) -> torch.Tensor:
    """verts [n, 4, 3]: the squared distance of the closest feature pair the flag selects."""
    n_bits = ((flag >> 0) & 1) + ((flag >> 1) & 1) + ((flag >> 2) & 1) + ((flag >> 3) & 1)
    rows = torch.arange(len(flag), device=flag.device)
    offsets = set_bits(flag)
    if is_ee:
        # PE inside an EE stencil: the point is the endpoint whose partner bit is unset, the edge the other edge
        missing = torch.argmin(((flag[:, None] >> torch.arange(4, device=flag.device)) & 1), dim=1)
        point = torch.where(missing == 0, 1, torch.where(missing == 1, 0, torch.where(missing == 2, 3, 2)))
        edge0 = torch.where(missing < 2, 2, 0)
        pe_idx = torch.stack((point, edge0, edge0 + 1), dim=1)
    else:
        pe_idx = offsets[:, :3]
    d2_pp = pp_distance2(verts[rows, offsets[:, 0]], verts[rows, offsets[:, 1]])
    d2_pe = pe_distance2(verts[rows, pe_idx[:, 0]], verts[rows, pe_idx[:, 1]], verts[rows, pe_idx[:, 2]])
    distance2 = ee_distance2 if is_ee else pt_distance2
    d2_full = distance2(verts[:, 0], verts[:, 1], verts[:, 2], verts[:, 3])
    return torch.where(n_bits == 2, d2_pp, torch.where(n_bits == 3, d2_pe, d2_full))


def classify(channel: str, stencil: torch.Tensor, positions: torch.Tensor, d_hats: torch.Tensor,
             thicknesses: torch.Tensor, planes_pos: torch.Tensor | None = None,
             planes_normal: torch.Tensor | None = None, hp_off: int = 0):
    """Return (flag, d2, active) of every candidate of a channel, as CGQ's narrow phase computes them."""
    if channel == "ph":
        v0, plane = stencil[:, 0], stencil[:, 1] - hp_off
        dist = dot(planes_normal[plane], positions[v0] - planes_pos[plane])
        d2 = dist * dist
        d_hat = d_hats[v0]
        xi = thicknesses[v0]
        flag = torch.zeros_like(v0, dtype=torch.int32)
    else:
        verts = positions[stencil]
        if channel == "pt":
            flag = pt_distance_flag(verts[:, 0], verts[:, 1], verts[:, 2], verts[:, 3])
            xi = thicknesses[stencil[:, 0]] + torch.fmax(
                thicknesses[stencil[:, 1]], torch.fmax(thicknesses[stencil[:, 2]], thicknesses[stencil[:, 3]])
            )
        else:
            flag = ee_distance_flag(verts[:, 0], verts[:, 1], verts[:, 2], verts[:, 3])
            xi = torch.fmax(thicknesses[stencil[:, 0]], thicknesses[stencil[:, 1]]) + torch.fmax(
                thicknesses[stencil[:, 2]], thicknesses[stencil[:, 3]]
            )
        d2 = flagged_distance2(flag, verts, is_ee=channel == "ee")
        # pair_d_hat_pt / _ee: the mean of the d_hat of the two primitives, uniform within each primitive
        d_hat = (d_hats[stencil[:, 0]] + d_hats[stencil[:, 2 if channel == "ee" else 1]]) / 2.0
    d_hat_xi = d_hat + xi
    active = ~((d2 >= d_hat_xi * d_hat_xi) | (d2 <= xi * xi))
    return flag, d2, active
