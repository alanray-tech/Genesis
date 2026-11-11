"""The bvh.py broad phase and the Quadrants narrow phase of the benchmark (see upstream_side.py)."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch

import quadrants as qd

import genesis as gs
from genesis.engine.bvh import (
    BoxQuery,
    BVHLeaves,
    BVHQueryResults,
    BVHTreeState,
    build_bvh,
    func_bvh_query_leaves,
    get_bvh_data,
)
from genesis.utils.misc import qd_to_numpy

from narrow_ref import classify

# halfplane_query_kernel widens its cull by GlobalContactContext::pair_reach_scale, 1 unless a constitution reaches
# past d_hat (adhesion), which none of the benchmark scenes uses
PAIR_REACH_SCALE = 1.0
CHANNELS = ("pt", "ee", "ph")


@qd.func
def func_f32_down(x):
    """The fp32 value at most x: round to nearest, then step down by at least one ulp."""
    y = qd.cast(x, qd.f32)
    return y - (qd.abs(y) * 1.2e-7 + 1e-30)


@qd.func
def func_f32_up(x):
    y = qd.cast(x, qd.f32)
    return y + (qd.abs(y) * 1.2e-7 + 1e-30)


@qd.func
def func_dot(a, b):
    return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]


@qd.func
def func_cross(a, b):
    return qd.Vector(
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], dt=qd.f64
    )


@qd.func
def func_det3(a, b, c):
    return func_dot(a, func_cross(b, c))


@qd.func
def func_pt_flag(p, t0, t1, t2) -> qd.i32:
    """pt_distance_flag of CGQ (contact_function/distance_flag.h)."""
    n_vec = func_cross(t1 - t0, t2 - t0)
    basis0 = t1 - t0
    basis1 = func_cross(basis0, n_vec)
    basis2 = p - t0
    det = func_det3(basis0, basis1, n_vec)
    p0x = func_det3(basis2, basis1, n_vec) / det
    p0y = func_det3(basis0, basis2, n_vec) / det
    basis0 = t2 - t1
    basis1 = func_cross(basis0, n_vec)
    basis2 = p - t1
    det = func_det3(basis0, basis1, n_vec)
    p1x = func_det3(basis2, basis1, n_vec) / det
    p1y = func_det3(basis0, basis2, n_vec) / det
    basis0 = t0 - t2
    basis1 = func_cross(basis0, n_vec)
    basis2 = p - t2
    det = func_det3(basis0, basis1, n_vec)
    p2x = func_det3(basis2, basis1, n_vec) / det
    p2y = func_det3(basis0, basis2, n_vec) / det
    flag = 0xF
    if p0x > 0.0 and p0x < 1.0 and p0y >= 0.0:
        flag = 0x1 | 0x2 | 0x4
    elif p1x > 0.0 and p1x < 1.0 and p1y >= 0.0:
        flag = 0x1 | 0x4 | 0x8
    elif p2x > 0.0 and p2x < 1.0 and p2y >= 0.0:
        flag = 0x1 | 0x8 | 0x2
    elif p0x <= 0.0 and p2x >= 1.0:
        flag = 0x1 | 0x2
    elif p1x <= 0.0 and p0x >= 1.0:
        flag = 0x1 | 0x4
    elif p2x <= 0.0 and p1x >= 1.0:
        flag = 0x1 | 0x8
    return flag


@qd.func
def func_ee_flag(ea0, ea1, eb0, eb1) -> qd.i32:
    """ee_distance_flag of CGQ (contact_function/distance_flag.h)."""
    u = ea1 - ea0
    v = eb1 - eb0
    w = ea0 - eb0
    a = func_dot(u, u)
    b = func_dot(u, v)
    c = func_dot(v, v)
    d = func_dot(u, w)
    e = func_dot(v, w)
    big_d = a * c - b * b
    t_d = big_d
    s_n = b * e - c * d
    t_n = 0.0
    default = 0xF
    is_near_parallel = a > 0.0 and c > 0.0 and big_d <= 1.0e-6 * (a * c)
    if is_near_parallel or s_n <= 0.0:
        t_n = e
        t_d = c
        default = 0x1 | 0x4 | 0x8
    elif s_n >= big_d:
        t_n = e + b
        t_d = c
        default = 0x2 | 0x4 | 0x8
    else:
        t_n = a * e - b * d
    flag = default
    if t_n <= 0.0:
        if -d <= 0.0:
            flag = 0x1 | 0x4
        elif -d >= a:
            flag = 0x2 | 0x4
        else:
            flag = 0x1 | 0x2 | 0x4
    elif t_n >= t_d:
        s_hi = -d + b
        if s_hi <= 0.0:
            flag = 0x1 | 0x8
        elif s_hi >= a:
            flag = 0x2 | 0x8
        else:
            flag = 0x1 | 0x2 | 0x8
    return flag


@qd.func
def func_lowest_bit(flag: qd.i32) -> qd.i32:
    i_bit = 3
    if flag & 0x1:
        i_bit = 0
    elif flag & 0x2:
        i_bit = 1
    elif flag & 0x4:
        i_bit = 2
    return i_bit


@qd.func
def func_select(i: qd.i32, x0, x1, x2, x3):
    x = x0
    if i == 1:
        x = x1
    elif i == 2:
        x = x2
    elif i == 3:
        x = x3
    return x


@qd.func
def func_sq_norm(a):
    return func_dot(a, a)


@qd.func
def func_flagged_distance2(flag: qd.i32, x0, x1, x2, x3, is_ee: qd.template()):
    """The squared distance of the feature pair of a flag (pt_ / ee_flagged_distance2 of CGQ)."""
    o0 = func_lowest_bit(flag)
    rest = flag & (flag - 1)
    o1 = func_lowest_bit(rest)
    o2 = func_lowest_bit(rest & (rest - 1))
    n_bits = (flag & 1) + ((flag >> 1) & 1) + ((flag >> 2) & 1) + ((flag >> 3) & 1)
    if qd.static(is_ee):
        if n_bits == 3:
            # PE in an EE stencil: the point is the endpoint whose partner bit is unset, the edge the other edge
            missing = func_lowest_bit(~flag & 0xF)
            o1 = 0
            if missing < 2:
                o1 = 2
            o2 = o1 + 1
            o0 = 1 - missing
            if missing == 2:
                o0 = 3
            elif missing == 3:
                o0 = 2
    d2 = 0.0
    if n_bits == 2:
        d2 = func_sq_norm(func_select(o0, x0, x1, x2, x3) - func_select(o1, x0, x1, x2, x3))
    elif n_bits == 3:
        p = func_select(o0, x0, x1, x2, x3)
        e0 = func_select(o1, x0, x1, x2, x3)
        e1 = func_select(o2, x0, x1, x2, x3)
        d2 = func_sq_norm(func_cross(e0 - p, e1 - p)) / func_sq_norm(e1 - e0)
    else:
        if qd.static(is_ee):
            b = func_cross(x1 - x0, x3 - x2)
            a_t_b = func_dot(x2 - x0, b)
            d2 = a_t_b * a_t_b / func_sq_norm(b)
        else:
            b = func_cross(x2 - x1, x3 - x1)
            a_t_b = func_dot(x0 - x1, b)
            d2 = a_t_b * a_t_b / func_sq_norm(b)
    return d2


@qd.func
def func_clamp01(x):
    return qd.min(qd.max(x, 0.0), 1.0)


@qd.func
def func_closest_pt(p, t0, t1, t2):
    """Closest point of a triangle to a point and its distance (Ericson, ClosestPtPointTriangle, as CGQ's
    directional_ccd.h)."""
    ab = t1 - t0
    ac = t2 - t0
    ap = p - t0
    d1 = func_dot(ab, ap)
    d2 = func_dot(ac, ap)
    bp = p - t1
    d3 = func_dot(ab, bp)
    d4 = func_dot(ac, bp)
    vc = d1 * d4 - d3 * d2
    cp = p - t2
    d5 = func_dot(ab, cp)
    d6 = func_dot(ac, cp)
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    cb = t0
    if d1 <= 0.0 and d2 <= 0.0:
        cb = t0
    elif d3 >= 0.0 and d4 <= d3:
        cb = t1
    elif vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        cb = t0 + (d1 / (d1 - d3)) * ab
    elif d6 >= 0.0 and d5 <= d6:
        cb = t2
    elif vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        cb = t0 + (d2 / (d2 - d6)) * ac
    elif va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        cb = t1 + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (t2 - t1)
    else:
        denom = 1.0 / (va + vb + vc)
        cb = t0 + ab * (vb * denom) + ac * (vc * denom)
    return cb, qd.sqrt(func_sq_norm(p - cb))


@qd.func
def func_closest_ee(p1, q1, p2, q2):
    """Closest points of two segments and their distance (Ericson, ClosestPtSegmentSegment, as CGQ's
    directional_ccd.h)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = func_sq_norm(d1)
    e = func_sq_norm(d2)
    f = func_dot(d2, r)
    c = func_dot(d1, r)
    b = func_dot(d1, d2)
    s = 0.0
    t = 0.0
    tiny = 1e-300
    if a <= tiny and e <= tiny:
        s = 0.0
        t = 0.0
    elif a <= tiny:
        t = func_clamp01(f / e)
    elif e <= tiny:
        s = func_clamp01(-c / a)
    else:
        denom = a * e - b * b
        if denom > 0.0:
            s = func_clamp01((b * f - c * e) / denom)
        t = (b * s + f) / e
        if t < 0.0:
            t = 0.0
            s = func_clamp01(-c / a)
        elif t > 1.0:
            t = 1.0
            s = func_clamp01((b - c) / a)
    ca = p1 + s * d1
    cb = p2 + t * d2
    return ca, cb, qd.sqrt(func_sq_norm(ca - cb))


@qd.data_oriented
class Mesh:
    """The topology and contact data of a captured scene, and the vertex state of the frame being processed."""

    def __init__(self, frames: dict) -> None:
        self.n_verts = int(frames["n_verts"])
        self.n_bodies = len(frames["self_collision"])
        self.n_ce = int(frames["n_contact_elements"])
        self.hp_off = int(frames["hp_off"])
        self.n_hp = len(frames["halfplane_normals"])
        faces, edges = frames["surf_faces"], frames["surf_edges"]
        self.faces = qd.Vector.field(3, qd.i32, shape=len(faces))
        self.faces.from_numpy(faces.astype(np.int32))
        self.edges = qd.Vector.field(2, qd.i32, shape=len(edges))
        self.edges.from_numpy(edges.astype(np.int32))
        self.surf_verts = qd.field(qd.i32, shape=len(frames["surf_verts"]))
        self.surf_verts.from_numpy(frames["surf_verts"].astype(np.int32))
        self.verts_body = qd.field(qd.i32, shape=self.n_verts)
        self.verts_body.from_numpy(frames["body_id"].astype(np.int32))
        contact_ids = np.zeros(self.n_verts, dtype=np.int32)
        contact_ids[: len(frames["vert_contact_element_ids"])] = frames["vert_contact_element_ids"]
        self.verts_contact_id = qd.field(qd.i32, shape=self.n_verts)
        self.verts_contact_id.from_numpy(contact_ids)
        self.verts_d_hat = qd.field(qd.f64, shape=self.n_verts)
        self.verts_d_hat.from_numpy(frames["d_hats"])
        self.verts_thickness = qd.field(qd.f64, shape=self.n_verts)
        self.verts_thickness.from_numpy(frames["thicknesses"])
        self.bodies_self_collision = qd.field(qd.i32, shape=self.n_bodies)
        self.bodies_self_collision.from_numpy(frames["self_collision"].astype(np.int32))
        self.ignore_ranges = qd.field(qd.i32, shape=self.n_bodies + 1)
        self.ignore_ranges.from_numpy(frames["body_ignore_ranges"].astype(np.int32))
        ignore_ids = frames["body_ignore_ids"].astype(np.int32)
        self.ignore_ids = qd.field(qd.i32, shape=max(1, len(ignore_ids)))
        self.ignore_ids.from_numpy(np.concatenate([ignore_ids, np.zeros(1 - min(1, len(ignore_ids)), np.int32)]))
        n_mask = max(1, self.n_ce * self.n_ce)
        self.mask_pt = qd.field(qd.i32, shape=n_mask)
        self.mask_pt.from_numpy(np.resize(frames["contact_mask_table"].astype(np.int32).reshape(-1), n_mask))
        self.mask_ee = qd.field(qd.i32, shape=n_mask)
        self.mask_ee.from_numpy(np.resize(frames["contact_mask_table_ee"].astype(np.int32).reshape(-1), n_mask))
        self.hp_normal = qd.Vector.field(3, qd.f64, shape=max(1, self.n_hp))
        self.hp_contact_id = qd.field(qd.i32, shape=max(1, self.n_hp))
        if self.n_hp > 0:
            self.hp_normal.from_numpy(frames["halfplane_normals"])
            self.hp_contact_id.from_numpy(frames["halfplane_contact_element_ids"].astype(np.int32))
        self.pos = qd.Vector.field(3, qd.f64, shape=self.n_verts)
        self.end = qd.Vector.field(3, qd.f64, shape=self.n_verts)
        self.inflation = qd.field(qd.f64, shape=self.n_verts)
        self.path_speed = qd.field(qd.f64, shape=self.n_verts)
        self.path_kind = qd.field(qd.i32, shape=self.n_verts)

    @qd.func
    def travel(self, i_v: qd.i32):
        """candidate_vertex_travel of CGQ: the bound on how far a vertex moves over the step."""
        travel = self.path_speed[i_v]
        if not travel > 0.0:
            travel = qd.sqrt(func_sq_norm(self.end[i_v] - self.pos[i_v])) + self.inflation[i_v]
        return travel

    @qd.func
    def is_directional_kept(self, proj_a, close_a, infl_a, proj_b, close_b, infl_b, band) -> bool:
        """The verdict of candidate_directional_reach of CGQ: Keep, or a drop (Drop and CoMotion alike)."""
        gap = proj_b - proj_a - infl_a - infl_b
        per_side = gap - qd.max(close_a, 0.0) - qd.max(close_b, 0.0)
        is_kept = False
        if per_side < band:
            is_kept = gap - qd.max(close_a + close_b, 0.0) < band
        return is_kept

    @qd.func
    def is_pt_within_reach(self, i_v: qd.i32, tri) -> bool:
        """pt_candidate_within_reach of CGQ (contact_function/candidate_reach.h), reach scale 1, directional."""
        x0, x1, x2, x3 = self.pos[i_v], self.pos[tri[0]], self.pos[tri[1]], self.pos[tri[2]]
        flag = func_pt_flag(x0, x1, x2, x3)
        d2 = func_flagged_distance2(flag, x0, x1, x2, x3, False)
        band = (self.verts_d_hat[i_v] + self.verts_d_hat[tri[0]]) / 2.0 + (
            self.verts_thickness[i_v]
            + qd.max(self.verts_thickness[tri[0]], qd.max(self.verts_thickness[tri[1]], self.verts_thickness[tri[2]]))
        )
        travel = self.travel(i_v) + qd.max(self.travel(tri[0]), qd.max(self.travel(tri[1]), self.travel(tri[2])))
        reach = band + travel
        is_kept = not (d2 >= reach * reach)
        if is_kept:
            cb, dist = func_closest_pt(x0, x1, x2, x3)
            is_any_articulated = self.path_kind[i_v] != 0
            for k in qd.static(range(3)):
                if self.path_kind[tri[k]] != 0:
                    is_any_articulated = True
            if dist > 0.0 and not is_any_articulated:
                n = (cb - x0) / dist
                proj_b = func_dot(n, x1)
                close_b = -func_dot(n, self.end[tri[0]] - x1)
                infl_b = qd.max(0.0, self.inflation[tri[0]])
                for k in qd.static(range(1, 3)):
                    x_k = self.pos[tri[k]]
                    proj_b = qd.min(proj_b, func_dot(n, x_k))
                    close_b = qd.max(close_b, -func_dot(n, self.end[tri[k]] - x_k))
                    infl_b = qd.max(infl_b, self.inflation[tri[k]])
                is_kept = self.is_directional_kept(
                    func_dot(n, x0),
                    func_dot(n, self.end[i_v] - x0),
                    qd.max(0.0, self.inflation[i_v]),
                    proj_b,
                    close_b,
                    infl_b,
                    band,
                )
        return is_kept

    @qd.func
    def is_ee_within_reach(self, ea, eb) -> bool:
        """ee_candidate_within_reach of CGQ (contact_function/candidate_reach.h), reach scale 1, directional."""
        x0, x1, x2, x3 = self.pos[ea[0]], self.pos[ea[1]], self.pos[eb[0]], self.pos[eb[1]]
        flag = func_ee_flag(x0, x1, x2, x3)
        d2 = func_flagged_distance2(flag, x0, x1, x2, x3, True)
        band = (self.verts_d_hat[ea[0]] + self.verts_d_hat[eb[0]]) / 2.0 + (
            qd.max(self.verts_thickness[ea[0]], self.verts_thickness[ea[1]])
            + qd.max(self.verts_thickness[eb[0]], self.verts_thickness[eb[1]])
        )
        travel = qd.max(self.travel(ea[0]), self.travel(ea[1])) + qd.max(self.travel(eb[0]), self.travel(eb[1]))
        reach = band + travel
        is_kept = not (d2 >= reach * reach)
        if is_kept:
            ca, cb, dist = func_closest_ee(x0, x1, x2, x3)
            is_any_articulated = False
            for k in qd.static(range(2)):
                if self.path_kind[ea[k]] != 0 or self.path_kind[eb[k]] != 0:
                    is_any_articulated = True
            if dist > 0.0 and not is_any_articulated:
                n = (cb - ca) / dist
                proj_a = qd.max(func_dot(n, x0), func_dot(n, x1))
                close_a = qd.max(func_dot(n, self.end[ea[0]] - x0), func_dot(n, self.end[ea[1]] - x1))
                proj_b = qd.min(func_dot(n, x2), func_dot(n, x3))
                close_b = qd.max(-func_dot(n, self.end[eb[0]] - x2), -func_dot(n, self.end[eb[1]] - x3))
                infl_a = qd.max(0.0, qd.max(self.inflation[ea[0]], self.inflation[ea[1]]))
                infl_b = qd.max(0.0, qd.max(self.inflation[eb[0]], self.inflation[eb[1]]))
                is_kept = self.is_directional_kept(proj_a, close_a, infl_a, proj_b, close_b, infl_b, band)
        return is_kept

    @qd.func
    def body_of_edge(self, edge):
        i_b = self.verts_body[edge[0]]
        if self.verts_body[edge[1]] != i_b:
            i_b = -1
        return i_b

    @qd.func
    def body_of_face(self, tri):
        i_b = self.verts_body[tri[0]]
        if self.verts_body[tri[1]] != i_b or self.verts_body[tri[2]] != i_b:
            i_b = -1
        return i_b

    @qd.func
    def is_ignored(self, i_b: qd.i32, j_b: qd.i32) -> bool:
        is_ignored = False
        if i_b < self.n_bodies:
            for k in range(self.ignore_ranges[i_b], self.ignore_ranges[i_b + 1]):
                if self.ignore_ids[k] == j_b:
                    is_ignored = True
        return is_ignored

    @qd.func
    def is_body_pair_kept(self, i_b: qd.i32, j_b: qd.i32) -> bool:
        """Rules (1) and (2) of CGQ's contact predicates: self collision and the ignored body pairs."""
        is_kept = True
        if i_b == j_b and i_b >= 0:
            if self.bodies_self_collision[i_b] == 0:
                is_kept = False
        if i_b >= 0 and j_b >= 0 and i_b != j_b:
            if self.is_ignored(i_b, j_b):
                is_kept = False
        return is_kept

    @qd.func
    def is_enabled(self, mask: qd.template(), e_a: qd.i32, e_b: qd.i32) -> bool:
        return mask[self.verts_contact_id[e_a] * self.n_ce + self.verts_contact_id[e_b]] != 0


@qd.data_oriented
class PTQuery(BoxQuery):
    """Point queries against the triangle tree, keeping the pairs CGQ's ContactPTPred keeps: rules 1 to 4, and the
    distance rule 7 when `is_reach` holds (the inline schedule)."""

    def __init__(self, results: BVHQueryResults, mesh: Mesh, is_reach: bool) -> None:
        super().__init__(results)
        self.mesh = mesh
        self.is_reach = is_reach

    @qd.func
    def is_kept(self, i_t: int, i_q: int, i_k: int, i_a: int) -> bool:
        mesh = self.mesh
        i_v = mesh.surf_verts[i_q]
        tri = mesh.faces[i_a]
        is_kept = mesh.is_body_pair_kept(mesh.verts_body[i_v], mesh.body_of_face(tri))
        if i_v == tri[0] or i_v == tri[1] or i_v == tri[2]:
            is_kept = False
        if qd.static(mesh.n_ce > 0):
            for k in qd.static(range(3)):
                if not mesh.is_enabled(mesh.mask_pt, i_v, tri[k]):
                    is_kept = False
        if qd.static(self.is_reach):
            if is_kept:
                is_kept = mesh.is_pt_within_reach(i_v, tri)
        return is_kept


@qd.data_oriented
class EEQuery(BoxQuery):
    """Edge queries against the edge tree, each pair once with the lower edge querying (CGQ's orientation), keeping
    the pairs CGQ's ContactEEPred keeps: rules 1 to 4, and the distance rule 5 when `is_reach` holds."""

    def __init__(self, results: BVHQueryResults, mesh: Mesh, is_reach: bool) -> None:
        super().__init__(results)
        self.mesh = mesh
        self.is_reach = is_reach

    @qd.func
    def is_kept(self, i_t: int, i_q: int, i_k: int, i_a: int) -> bool:
        mesh = self.mesh
        is_kept = i_q < i_a
        if is_kept:
            ea = mesh.edges[i_q]
            eb = mesh.edges[i_a]
            is_kept = mesh.is_body_pair_kept(mesh.body_of_edge(ea), mesh.body_of_edge(eb))
            if ea[0] == eb[0] or ea[0] == eb[1] or ea[1] == eb[0] or ea[1] == eb[1]:
                is_kept = False
            if qd.static(mesh.n_ce > 0):
                for i, j in qd.static(qd.ndrange(2, 2)):
                    if not mesh.is_enabled(mesh.mask_ee, ea[i], eb[j]):
                        is_kept = False
            if qd.static(self.is_reach):
                if is_kept:
                    is_kept = mesh.is_ee_within_reach(ea, eb)
        return is_kept


@qd.kernel
def kernel_reach_filter(mesh: qd.template(), raw: BVHQueryResults, results: BVHQueryResults, is_ee: qd.template()):
    """The deferred schedule: the distance rule over the raw hits of a query, appending the kept ones."""
    results.count[0] = 0
    results.trees_count[0] = 0
    for i_r in range(qd.min(raw.count[0], raw.triplets.shape[0])):
        i_a = raw.triplets[i_r][1]
        i_q = raw.triplets[i_r][2]
        is_kept = False
        if qd.static(is_ee):
            is_kept = mesh.is_ee_within_reach(mesh.edges[i_q], mesh.edges[i_a])
        else:
            is_kept = mesh.is_pt_within_reach(mesh.surf_verts[i_q], mesh.faces[i_a])
        if is_kept:
            i_out = qd.atomic_add(results.count[0], 1)
            if i_out < results.triplets.shape[0]:
                results.triplets[i_out] = raw.triplets[i_r]


@qd.kernel
def kernel_bounds(
    mesh: qd.template(),
    faces_leaves: BVHLeaves,
    edges_leaves: BVHLeaves,
    points_query: BVHLeaves,
    edges_query: BVHLeaves,
    d_hat: qd.f64,
):
    """Leaf boxes of the triangles and edges, query boxes of the points and edges (see upstream_side.py)."""
    for i_f in range(mesh.faces.shape[0]):
        tri = mesh.faces[i_f]
        lo = qd.min(mesh.pos[tri[0]], mesh.end[tri[0]])
        hi = qd.max(mesh.pos[tri[0]], mesh.end[tri[0]])
        thickness = mesh.verts_thickness[tri[0]]
        inflation = mesh.inflation[tri[0]]
        for k in qd.static(range(1, 3)):
            lo = qd.min(lo, qd.min(mesh.pos[tri[k]], mesh.end[tri[k]]))
            hi = qd.max(hi, qd.max(mesh.pos[tri[k]], mesh.end[tri[k]]))
            thickness = qd.max(thickness, mesh.verts_thickness[tri[k]])
            inflation = qd.max(inflation, mesh.inflation[tri[k]])
        margin = thickness + inflation + mesh.verts_d_hat[tri[0]]
        faces_leaves.aabbs_min[0, i_f] = func_f32_down(lo - margin)
        faces_leaves.aabbs_max[0, i_f] = func_f32_up(hi + margin)
    for i_e in range(mesh.edges.shape[0]):
        edge = mesh.edges[i_e]
        lo = qd.min(qd.min(mesh.pos[edge[0]], mesh.end[edge[0]]), qd.min(mesh.pos[edge[1]], mesh.end[edge[1]]))
        hi = qd.max(qd.max(mesh.pos[edge[0]], mesh.end[edge[0]]), qd.max(mesh.pos[edge[1]], mesh.end[edge[1]]))
        thickness = qd.max(mesh.verts_thickness[edge[0]], mesh.verts_thickness[edge[1]])
        inflation = qd.max(mesh.inflation[edge[0]], mesh.inflation[edge[1]])
        margin = thickness + inflation + mesh.verts_d_hat[edge[0]]
        edges_leaves.aabbs_min[0, i_e] = func_f32_down(lo - margin)
        edges_leaves.aabbs_max[0, i_e] = func_f32_up(hi + margin)
        # CGQ's EE leaf test adds the global d_hat as a gap between the two edge boxes
        edges_query.aabbs_min[0, i_e] = func_f32_down(lo - (margin + d_hat))
        edges_query.aabbs_max[0, i_e] = func_f32_up(hi + (margin + d_hat))
    for i_s in range(mesh.surf_verts.shape[0]):
        i_v = mesh.surf_verts[i_s]
        lo = qd.min(mesh.pos[i_v], mesh.end[i_v])
        hi = qd.max(mesh.pos[i_v], mesh.end[i_v])
        # The point gap of CGQ's PT leaf test, accumulated in its order: (d_hat + thickness) + inflation
        gap = (mesh.verts_d_hat[i_v] + mesh.verts_thickness[i_v]) + mesh.inflation[i_v]
        points_query.aabbs_min[0, i_s] = func_f32_down(lo - gap)
        points_query.aabbs_max[0, i_s] = func_f32_up(hi + gap)


@qd.kernel
def kernel_query(queries: BVHLeaves, tree: BVHTreeState, visitor: qd.template()):
    visitor.results.count[0] = 0
    visitor.results.trees_count[0] = 0
    func_bvh_query_leaves(queries, tree, visitor)


@qd.kernel
def kernel_ph_query(mesh: qd.template(), ph_pairs: qd.template(), ph_count: qd.template(), d_hat_scale: qd.f64):
    """halfplane_query_kernel of CGQ: every surface vertex against every half-plane, no BVH."""
    ph_count[0] = 0
    for i_s, i_p in qd.ndrange(mesh.surf_verts.shape[0], mesh.n_hp):
        i_v = mesh.surf_verts[i_s]
        thickness = mesh.verts_thickness[i_v]
        is_kept = mesh.mask_pt[mesh.verts_contact_id[i_v] * mesh.n_ce + mesh.hp_contact_id[i_p]] != 0
        i_b = mesh.verts_body[i_v]
        if is_kept and i_b >= 0:
            if mesh.is_ignored(i_b, mesh.verts_body[mesh.hp_off + i_p]):
                is_kept = False
        x = mesh.pos[i_v]
        normal = mesh.hp_normal[i_p]
        dist = func_dot(normal, x - mesh.pos[mesh.hp_off + i_p])
        dist_dx = func_dot(normal, mesh.end[i_v] - x)
        dist_min = dist + qd.min(0.0, dist_dx) - mesh.inflation[i_v]
        dist_max = dist + qd.max(0.0, dist_dx) + mesh.inflation[i_v]
        if is_kept and not (dist_max <= 0.0 or dist_min >= mesh.verts_d_hat[i_v] * d_hat_scale + thickness):
            i_r = qd.atomic_add(ph_count[0], 1)
            if i_r < ph_pairs.shape[0]:
                ph_pairs[i_r] = qd.Vector([i_s, i_p], dt=qd.i32)


@qd.func
def func_is_active(d2, d_hat, xi) -> qd.i32:
    d_hat_xi = d_hat + xi
    is_active = 1
    if d2 >= d_hat_xi * d_hat_xi or d2 <= xi * xi:
        is_active = 0
    return is_active


@qd.kernel
def kernel_narrow(
    mesh: qd.template(),
    pt_results: BVHQueryResults,
    ee_results: BVHQueryResults,
    ph_pairs: qd.template(),
    ph_count: qd.template(),
    out: qd.template(),
):
    """Distance flag, squared distance and activation of every candidate (see narrow_ref.py for the reference)."""
    out.n_active[0] = 0
    out.n_active[1] = 0
    out.n_active[2] = 0
    for i_r in range(qd.min(pt_results.count[0], pt_results.triplets.shape[0])):
        i_a = pt_results.triplets[i_r][1]
        i_q = pt_results.triplets[i_r][2]
        i_v = mesh.surf_verts[i_q]
        tri = mesh.faces[i_a]
        x0, x1, x2, x3 = mesh.pos[i_v], mesh.pos[tri[0]], mesh.pos[tri[1]], mesh.pos[tri[2]]
        flag = func_pt_flag(x0, x1, x2, x3)
        d2 = func_flagged_distance2(flag, x0, x1, x2, x3, False)
        d_hat = (mesh.verts_d_hat[i_v] + mesh.verts_d_hat[tri[0]]) / 2.0
        xi = mesh.verts_thickness[i_v] + qd.max(
            mesh.verts_thickness[tri[0]], qd.max(mesh.verts_thickness[tri[1]], mesh.verts_thickness[tri[2]])
        )
        is_active = func_is_active(d2, d_hat, xi)
        out.pt_flag[i_r] = flag
        out.pt_active[i_r] = is_active
        qd.atomic_add(out.n_active[0], is_active)
    for i_r in range(qd.min(ee_results.count[0], ee_results.triplets.shape[0])):
        ea = mesh.edges[ee_results.triplets[i_r][2]]
        eb = mesh.edges[ee_results.triplets[i_r][1]]
        x0, x1, x2, x3 = mesh.pos[ea[0]], mesh.pos[ea[1]], mesh.pos[eb[0]], mesh.pos[eb[1]]
        flag = func_ee_flag(x0, x1, x2, x3)
        d2 = func_flagged_distance2(flag, x0, x1, x2, x3, True)
        d_hat = (mesh.verts_d_hat[ea[0]] + mesh.verts_d_hat[eb[0]]) / 2.0
        xi = qd.max(mesh.verts_thickness[ea[0]], mesh.verts_thickness[ea[1]]) + qd.max(
            mesh.verts_thickness[eb[0]], mesh.verts_thickness[eb[1]]
        )
        is_active = func_is_active(d2, d_hat, xi)
        out.ee_flag[i_r] = flag
        out.ee_active[i_r] = is_active
        qd.atomic_add(out.n_active[1], is_active)
    for i_r in range(qd.min(ph_count[0], ph_pairs.shape[0])):
        i_v = mesh.surf_verts[ph_pairs[i_r][0]]
        i_p = ph_pairs[i_r][1]
        dist = func_dot(mesh.hp_normal[i_p], mesh.pos[i_v] - mesh.pos[mesh.hp_off + i_p])
        is_active = func_is_active(dist * dist, mesh.verts_d_hat[i_v], mesh.verts_thickness[i_v])
        out.ph_active[i_r] = is_active
        qd.atomic_add(out.n_active[2], is_active)


@qd.data_oriented
class NarrowOutput:
    def __init__(self, n_pt: int, n_ee: int, n_ph: int) -> None:
        self.pt_flag = qd.field(qd.i32, shape=n_pt)
        self.pt_active = qd.field(qd.i32, shape=n_pt)
        self.ee_flag = qd.field(qd.i32, shape=n_ee)
        self.ee_active = qd.field(qd.i32, shape=n_ee)
        self.ph_active = qd.field(qd.i32, shape=n_ph)
        self.n_active = qd.field(qd.i32, shape=3)


def nbytes(value) -> int:
    """Device bytes of a Quadrants field or of every array of a struct of them."""
    if dataclasses.is_dataclass(value):
        return sum(nbytes(getattr(value, f.name)) for f in dataclasses.fields(value))
    return qd_to_numpy(value).nbytes


CAPACITY_KEYS = ("pt", "ee", "ph", "pt_raw", "ee_raw")


class Pipeline:
    """The bvh.py broad phase and the narrow phase of one scene, its result capacities fixed at construction.

    `schedule` places CGQ's distance rule: "inline" in BoxQuery.is_kept during the traversal, "deferred" in a filter
    kernel over the raw hits the traversal records (CGQ's production schedule).
    """

    DEFAULT_CAPACITIES = dict(pt=1 << 22, ee=1 << 23, ph=1 << 18, pt_raw=1 << 23, ee_raw=1 << 24)

    def __init__(self, frames: dict, schedule: str) -> None:
        self.mesh = mesh = Mesh(frames)
        self.schedule = schedule
        self.d_hat = float(frames["d_hat"])
        self.faces_bvh = get_bvh_data(1, mesh.faces.shape[0])
        self.edges_bvh = get_bvh_data(1, mesh.edges.shape[0])
        self.points_query = BVHLeaves(sizes=dict(bvh_trees=1, bvh_leaves=mesh.surf_verts.shape[0]))
        self.edges_query = BVHLeaves(sizes=dict(bvh_trees=1, bvh_leaves=mesh.edges.shape[0]))
        self.resize(self.DEFAULT_CAPACITIES)

    def resize(self, capacities: dict) -> None:
        is_deferred = self.schedule == "deferred"
        self.capacities = {key: max(1, int(capacities[key])) if is_deferred or "raw" not in key else 1 for key in
                           CAPACITY_KEYS}
        self.pt_results = BVHQueryResults(sizes=dict(query_results=self.capacities["pt"], bvh_trees=1))
        self.ee_results = BVHQueryResults(sizes=dict(query_results=self.capacities["ee"], bvh_trees=1))
        self.pt_raw = BVHQueryResults(sizes=dict(query_results=self.capacities["pt_raw"], bvh_trees=1))
        self.ee_raw = BVHQueryResults(sizes=dict(query_results=self.capacities["ee_raw"], bvh_trees=1))
        self.pt_query = PTQuery(self.pt_raw if is_deferred else self.pt_results, self.mesh, is_reach=not is_deferred)
        self.ee_query = EEQuery(self.ee_raw if is_deferred else self.ee_results, self.mesh, is_reach=not is_deferred)
        self.ph_pairs = qd.Vector.field(2, qd.i32, shape=self.capacities["ph"])
        self.ph_count = qd.field(qd.i32, shape=1)
        self.out = NarrowOutput(self.capacities["pt"], self.capacities["ee"], self.capacities["ph"])

    def load_frame(self, frames: dict, i_f: int) -> None:
        self.mesh.pos.from_numpy(frames["positions"][i_f])
        self.mesh.end.from_numpy(frames["trajectory_end_positions"][i_f])
        self.mesh.inflation.from_numpy(frames["path_inflation"][i_f])
        self.mesh.path_speed.from_numpy(frames["path_speed"][i_f])
        self.mesh.path_kind.from_numpy(frames["path_kind"][i_f])
        qd.sync()

    def build(self) -> None:
        mesh = self.mesh
        faces_state, faces_config = self.faces_bvh
        edges_state, edges_config = self.edges_bvh
        kernel_bounds(mesh, faces_state.leaves, edges_state.leaves, self.points_query, self.edges_query, self.d_hat)
        build_bvh(faces_state, faces_config, eps=gs.EPS)
        build_bvh(edges_state, edges_config, eps=gs.EPS)

    def query_pt(self) -> None:
        kernel_query(self.points_query, self.faces_bvh.state.tree, self.pt_query)
        if self.schedule == "deferred":
            kernel_reach_filter(self.mesh, self.pt_raw, self.pt_results, False)

    def query_ee(self) -> None:
        kernel_query(self.edges_query, self.edges_bvh.state.tree, self.ee_query)
        if self.schedule == "deferred":
            kernel_reach_filter(self.mesh, self.ee_raw, self.ee_results, True)

    def query_ph(self) -> None:
        if self.mesh.n_hp > 0:
            kernel_ph_query(self.mesh, self.ph_pairs, self.ph_count, PAIR_REACH_SCALE)

    def narrow(self) -> None:
        kernel_narrow(self.mesh, self.pt_results, self.ee_results, self.ph_pairs, self.ph_count, self.out)

    def counts(self) -> dict:
        """The candidate and raw-hit counts, read back as the overflow check of a Newton iteration needs them."""
        counts = dict(
            pt=self.pt_results.count,
            ee=self.ee_results.count,
            ph=self.ph_count,
            pt_raw=self.pt_raw.count,
            ee_raw=self.ee_raw.count,
        )
        counts = {key: int(qd_to_numpy(count)[0]) for key, count in counts.items()}
        if self.mesh.n_hp == 0:
            counts["ph"] = 0
        if self.schedule != "deferred":
            counts["pt_raw"] = counts["ee_raw"] = 0
        return counts

    def run(self) -> dict:
        self.build()
        self.query_pt()
        self.query_ee()
        self.query_ph()
        self.narrow()
        return self.counts()

    def memory(self) -> dict:
        """Device bytes of the broad-phase buffers: boxes, trees and their build scratch, query results."""
        return {
            "trees_and_leaves": nbytes(self.faces_bvh.state) + nbytes(self.edges_bvh.state),
            "query_boxes": nbytes(self.points_query) + nbytes(self.edges_query),
            "query_results": nbytes(self.pt_results) + nbytes(self.ee_results) + nbytes(self.ph_pairs),
            "raw_hits": nbytes(self.pt_raw) + nbytes(self.ee_raw),
            "narrow_output": sum(nbytes(f) for f in vars(self.out).values()),
        }

    def candidates(self) -> dict:
        """The global stencils of the candidates and the Quadrants classification of each, on the host."""
        mesh = self.mesh
        counts = self.counts()
        if any(counts[key] > self.capacities[key] for key in CAPACITY_KEYS):
            raise RuntimeError(f"candidate capacity overflow: {counts} > {self.capacities}")
        n_pt, n_ee, n_ph = counts["pt"], counts["ee"], counts["ph"]
        surf_verts = qd_to_numpy(mesh.surf_verts)
        faces = qd_to_numpy(mesh.faces)
        edges = qd_to_numpy(mesh.edges)
        pt = qd_to_numpy(self.pt_results.triplets)[:n_pt]
        ee = qd_to_numpy(self.ee_results.triplets)[:n_ee]
        ph = qd_to_numpy(self.ph_pairs)[:n_ph]
        return {
            # Broad-phase-local pairs in CGQ's format: PT (surface slot, triangle), EE (lower edge, higher edge)
            "pt_pairs": np.column_stack([pt[:, 2], pt[:, 1]]),
            "ee_pairs": np.column_stack([ee[:, 2], ee[:, 1]]),
            "pt_stencil": np.column_stack([surf_verts[pt[:, 2]], faces[pt[:, 1]]]),
            "ee_stencil": np.column_stack([edges[ee[:, 2]], edges[ee[:, 1]]]),
            "ph_stencil": np.column_stack([surf_verts[ph[:, 0]], mesh.hp_off + ph[:, 1]]),
            "pt_flag": qd_to_numpy(self.out.pt_flag)[:n_pt],
            "ee_flag": qd_to_numpy(self.out.ee_flag)[:n_ee],
            "ph_flag": np.zeros(n_ph, dtype=np.int32),
            "pt_active": qd_to_numpy(self.out.pt_active)[:n_pt].astype(bool),
            "ee_active": qd_to_numpy(self.out.ee_active)[:n_ee].astype(bool),
            "ph_active": qd_to_numpy(self.out.ph_active)[:n_ph].astype(bool),
        }


def rows_view(rows: np.ndarray) -> np.ndarray:
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    return rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()


def set_diff_count(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """(|a - b|, |b - a|) of two sets of integer rows."""
    view_a, view_b = np.unique(rows_view(a)), np.unique(rows_view(b))
    common = np.intersect1d(view_a, view_b, assume_unique=True)
    return len(view_a) - len(common), len(view_b) - len(common)


def frame_slice(gt: dict, key: str, i_f: int) -> np.ndarray:
    offsets = gt[f"{key}_offsets"]
    return gt[key][offsets[i_f] : offsets[i_f + 1]]


def check_frame(bench: Pipeline, cand: dict, frames: dict, gt: dict, i_f: int) -> dict:
    """Compare the active pairs of this side to CGQ's on one frame (see upstream_side.py)."""
    device = "cuda"
    positions = torch.as_tensor(frames["positions"][i_f], device=device)
    d_hats = torch.as_tensor(frames["d_hats"], device=device)
    thicknesses = torch.as_tensor(frames["thicknesses"], device=device)
    normals = torch.as_tensor(frames["halfplane_normals"], device=device)
    n_hp = len(normals)
    planes_pos = positions[bench.mesh.hp_off : bench.mesh.hp_off + n_hp]
    row = {}
    for ch in CHANNELS:
        stencil = cand[f"{ch}_stencil"]
        flag, _, active = classify(
            ch, torch.as_tensor(stencil, device=device).long(), positions, d_hats, thicknesses,
            planes_pos=planes_pos, planes_normal=normals, hp_off=bench.mesh.hp_off,
        )
        flag, active = flag.cpu().numpy(), active.cpu().numpy()
        gt_stencil = frame_slice(gt, f"{ch}_stencil", i_f)
        gt_active = frame_slice(gt, f"{ch}_active", i_f).astype(bool)
        gt_flag = frame_slice(gt, f"{ch}_flag", i_f).astype(np.int64)
        upstream_pairs = np.column_stack([stencil[active], flag[active]])
        cgq_pairs = np.column_stack([gt_stencil[gt_active], gt_flag[gt_active] if ch != "ph" else 0 * gt_flag[gt_active]])
        only_upstream, only_cgq = set_diff_count(upstream_pairs, cgq_pairs)
        cgq_missed_by_upstream, upstream_not_in_cgq = set_diff_count(gt_stencil, stencil)
        duplicates = len(stencil) - len(np.unique(rows_view(stencil)))
        row[ch] = {
            "candidates_upstream": len(stencil),
            "candidates_cgq": len(gt_stencil),
            "active_upstream": int(active.sum()),
            "active_cgq": int(gt_active.sum()),
            "active_only_upstream": only_upstream,
            "active_only_cgq": only_cgq,
            "cgq_candidates_missing_upstream": cgq_missed_by_upstream,
            "upstream_candidates_not_in_cgq": upstream_not_in_cgq,
            "duplicates_upstream": duplicates,
            "quadrants_flag_mismatch": int((cand[f"{ch}_flag"] != flag).sum()) if ch != "ph" else 0,
            "quadrants_active_mismatch": int((cand[f"{ch}_active"] != active).sum()),
        }
    return row


def stats(samples: list[float]) -> dict:
    values = np.asarray(samples) * 1e3
    return {"median_ms": float(np.median(values)), "p90_ms": float(np.percentile(values, 90))}


def time_frame(bench: Pipeline, warmup: int, repeats: int, is_nvtx: bool = False) -> dict:
    """Warm GPU-synchronized timings of every stage, and of the whole sequence with one synchronization.

    With `is_nvtx`, every timed call is an NVTX range named after its stage, for an nsys trace (see upstream_nsys.py).
    """
    stages = {
        "build": bench.build,
        "pt_query": bench.query_pt,
        "ee_query": bench.query_ee,
        "ph_query": bench.query_ph,
        "narrow": bench.narrow,
        "counts_readback": bench.counts,
    }
    for _ in range(warmup):
        bench.run()
    samples = {name: [] for name in (*stages, "total")}
    for _ in range(repeats):
        for name, stage in (*stages.items(), ("total", bench.run)):
            qd.sync()
            if is_nvtx:
                torch.cuda.nvtx.range_push(name)
            start = time.perf_counter()
            stage()
            qd.sync()
            samples[name].append(time.perf_counter() - start)
            if is_nvtx:
                torch.cuda.nvtx.range_pop()
    return {name: stats(values) for name, values in samples.items()}


def run_benchmark(bench: Pipeline, frames: dict, gt: dict, args) -> dict:
    n_frames = len(frames["positions"])
    native_frames = gt["native_frames"].tolist()
    # Sizing pass at the default capacities, then the exact demand of the largest frame
    demand = dict.fromkeys(CAPACITY_KEYS, 0)
    for i_f in range(n_frames):
        bench.load_frame(frames, i_f)
        demand = {key: max(demand[key], count) for key, count in bench.run().items()}
    if any(demand[key] > bench.capacities[key] for key in CAPACITY_KEYS):
        raise RuntimeError(f"default capacities {bench.capacities} below the demand {demand}")
    bench.resize(demand)
    result = {
        "schedule": bench.schedule,
        "capacities": bench.capacities,
        "memory_bytes": bench.memory(),
        "frames": [],
    }
    frame_ids = range(n_frames)
    if args.nsys_frame is not None:
        n_cgq = [len(frame_slice(gt, "pt_stencil", i)) + len(frame_slice(gt, "ee_stencil", i)) for i in frame_ids]
        frame_ids = [int(np.argmax(n_cgq)) if args.nsys_frame < 0 else args.nsys_frame]
    local_pairs = {"pt": [], "ee": [], "frames": []}
    for i_f in frame_ids:
        bench.load_frame(frames, i_f)
        counts = bench.run()
        cand = bench.candidates()
        local_pairs["pt"].append(cand["pt_pairs"])
        local_pairs["ee"].append(cand["ee_pairs"])
        local_pairs["frames"].append(i_f)
        check = check_frame(bench, cand, frames, gt, i_f)
        check["raw_hits"] = {"pt": counts["pt_raw"], "ee": counts["ee_raw"]}
        if args.nsys_frame is not None:
            torch.cuda.profiler.start()
        timing = time_frame(bench, args.warmup, args.repeats, is_nvtx=args.nsys_frame is not None)
        if args.nsys_frame is not None:
            torch.cuda.profiler.stop()
        result["frames"].append({"frame": native_frames[i_f], "check": check, "timing": timing})
        summary = " ".join(
            f"{ch}: cand {c['candidates_upstream']}/{c['candidates_cgq']} active {c['active_upstream']}/"
            f"{c['active_cgq']} diff +{c['active_only_upstream']}/-{c['active_only_cgq']} "
            f"cand diff +{c['upstream_candidates_not_in_cgq']}/-{c['cgq_candidates_missing_upstream']} "
            f"qd {c['quadrants_flag_mismatch']}/{c['quadrants_active_mismatch']}"
            for ch, c in check.items()
            if ch in CHANNELS
        )
        print(f"frame {native_frames[i_f]}: {summary} | total {timing['total']['median_ms']:.3f} ms", flush=True)
    if args.nsys_frame is None:
        saved = {"frame_index": np.asarray(local_pairs["frames"], dtype=np.int32)}
        for ch in ("pt", "ee"):
            saved[f"{ch}_pairs"] = np.concatenate(local_pairs[ch]).astype(np.int32).reshape(-1, 2)
            saved[f"{ch}_pairs_offsets"] = np.concatenate([[0], np.cumsum([len(p) for p in local_pairs[ch]])])
        np.savez_compressed(args.data / args.scene / f"upstream_{bench.schedule}_candidates.npz", **saved)
    result["pass"] = all(
        row["check"][ch]["active_only_upstream"] == 0 and row["check"][ch]["active_only_cgq"] == 0
        for row in result["frames"]
        for ch in CHANNELS
    )
    print(f"PASS (active pairs identical on every frame): {result['pass']}")
    return result
