"""Cases of the IPC formula parity: the rows each case reads and the outputs it writes, shared by both sides.

A case evaluates one inventoried function (or the production composition of a few) on rows of float64 inputs. The
reference side is cgq_formulas/cgq_formulas.cu, the Genesis side genesis_side.py. Points are xyz triples. A stencil
lists side A then side B: (p, t0, t1, t2), (ea0, ea1, eb0, eb1), (p, e0, e1), (p0, p1).
"""

from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
# The CGQ commit taken as ground truth. Its inputs and outputs go to data/<commit>, its tables to results/<commit>. The
# tables of the previous ground truth, 0496c4fa, stay in results/.
GROUND_TRUTH = "aeb58956"
DATA = HERE / "data" / GROUND_TRUTH
RESULTS = HERE / "results" / GROUND_TRUTH

# Production constants of CGQ (contact_function headers and scene config defaults)
HESSIAN_CLAMP_RATIO = 1e-6
MOLLIFIER_RATIO = 1e-3
EE_PARALLEL_TOLERANCE = 1e-6
EE_TYPE_PARALLEL_TOLERANCE = 1e-14
INTERSECTION_TOLERANCE = 1e-12
CCD_MAX_ITERS = 50000
CCD_ETA = 0.5
CANDIDATE_REACH_SCALE = 1.0
# PSD projection of the mollified edge-edge Hessian (make_pd_lowrank_scalar.h): range tolerance relative to the
# Frobenius norm, Jacobi convergence (off-diagonal norm relative to the Frobenius norm) and sweep budget
PROJECTION_RANK_TOLERANCE = 1e-12
PROJECTION_EIGEN_TOLERANCE = 1e-15
PROJECTION_MAX_SWEEPS = 12

STENCIL_VERTS = {"pt": (1, 3), "ee": (2, 2), "pe": (1, 2), "pp": (1, 1)}


class Output(NamedTuple):
    name: str
    width: int
    is_discrete: bool


class Case(NamedTuple):
    name: str
    function: str
    n_inputs: int
    outputs: tuple


def _real(name, width=1):
    return Output(name, width, False)


def _discrete(name, width=1):
    return Output(name, width, True)


def _build_cases():
    cases = [
        # Distances: point_triangle / edge_edge / point_edge distance flags, flagged squared distances and gradients
        Case("pt_distance", "PT distance flag, flagged distance2 and gradient", 12,
             (_discrete("flag"), _real("distance2"), _real("gradient", 12))),
        Case("ee_distance", "EE distance flag, flagged distance2 and gradient", 12,
             (_discrete("flag"), _real("distance2"), _real("gradient", 12))),
        Case("pe_distance", "PE distance flag, flagged distance2 and gradient", 9,
             (_discrete("flag"), _real("distance2"), _real("gradient", 9))),
        Case("ee_distance_type", "EE distance type (parallel projection)", 12, (_discrete("flag"),)),
        Case("closest_pt", "PT closest point", 12, (_real("distance"), _real("closest_a", 3), _real("closest_b", 3))),
        Case("closest_ee", "EE closest points", 12, (_real("distance"), _real("closest_a", 3), _real("closest_b", 3))),
        Case("closest_pe", "PE closest point", 9, (_real("distance"), _real("closest_a", 3), _real("closest_b", 3))),
        # Barrier. Rows: distance2, d_hat, xi, kappa
        Case("barrier", "barrier, derivatives, normal force", 4,
             (_real("energy"), _real("first"), _real("second"), _real("normal_force"))),
        # Rows: distance2, cross_norm2, mollifier threshold, d_hat, xi, kappa
        Case("mollified_barrier", "mollified barrier", 6, (_real("energy"),)),
        # Rows: four points, taken as rest points for the threshold and current ones for the cross norm
        Case("mollifier", "mollifier threshold and edge cross norm2", 12, (_real("threshold"), _real("cross_norm2"))),
    ]
    # Rows: d_hat of side A and of side B, thicknesses of the stencil vertices (4, padded), area weights of A and B
    for kind in ("pt", "ee", "pe", "pp"):
        cases.append(Case(f"pair_{kind}", f"{kind.upper()} pair d_hat, thickness, area weight", 8,
                          (_real("d_hat"), _real("thickness"), _real("area_weight"))))
    cases += [
        # Rows: stencil points, [rest points,] d_hat, xi, kappa
        Case("barrier_pt", "PT barrier gradient/Hessian (flag dispatch)", 15,
             (_discrete("flag"), _real("gradient", 12), _real("hessian", 144))),
        Case("barrier_ee", "EE barrier gradient/Hessian (mollifier and flag dispatch)", 27,
             (_discrete("flag"), _discrete("is_mollified"), _real("gradient", 12), _real("hessian", 144))),
        Case("barrier_pe", "PE barrier gradient/Hessian (flag dispatch)", 12,
             (_discrete("flag"), _real("gradient", 9), _real("hessian", 81))),
        Case("barrier_pp", "PP barrier gradient/Hessian", 9, (_real("gradient", 6), _real("hessian", 36))),
        # Rows: x, plane point, unit normal, d_hat, xi, kappa
        Case("barrier_ph", "PH barrier gradient/Hessian", 12, (_real("gradient", 3), _real("hessian", 9))),
        # Rows: current stencil points, lagged stencil points, d_hat, kappa, mu, eps_vh, xi
        Case("friction_pt", "PT friction energy/gradient/Hessian", 29,
             (_real("energy"), _real("gradient", 12), _real("hessian", 144))),
        Case("friction_ee", "EE friction energy/gradient/Hessian", 29,
             (_real("energy"), _real("gradient", 12), _real("hessian", 144))),
        Case("friction_pe", "PE friction energy/gradient/Hessian", 23,
             (_real("energy"), _real("gradient", 9), _real("hessian", 81))),
        Case("friction_pp", "PP friction energy/gradient/Hessian", 17,
             (_real("energy"), _real("gradient", 6), _real("hessian", 36))),
        # Rows: x, lagged x, plane point, unit normal, d_hat, kappa, mu, eps_vh, xi
        Case("friction_ph", "PH friction energy/gradient/Hessian", 17,
             (_real("energy"), _real("gradient", 3), _real("hessian", 9))),
        # Rows: lagged stencil points (unit normal for PH). Bases as columns; PP and PH compare the projector T T^T
        Case("frame_pt", "PT tangent basis and closest-point weights", 12, (_real("basis", 6), _real("weights", 4))),
        Case("frame_ee", "EE tangent basis and closest-point weights", 12, (_real("basis", 6), _real("weights", 4))),
        Case("frame_pe", "PE tangent basis and closest-point weights", 9, (_real("basis", 6), _real("weights", 3))),
        Case("frame_pp", "PP tangent basis (projector)", 6, (_real("projector", 9),)),
        Case("frame_ph", "PH tangent basis (projector)", 3, (_real("projector", 9),)),
    ]
    for kind, (n_a, n_b) in STENCIL_VERTS.items():
        n = n_a + n_b
        # Rows: points, displacements, eta, xi
        cases.append(Case(f"ccd_additive_{kind}", f"{kind.upper()} additive CCD", 6 * n + 2, (_real("toi"),)))
        cases.append(Case(f"ccd_directional_{kind}", f"{kind.upper()} directional CCD", 6 * n + 2, (_real("toi"),)))
        # Rows: points, chords, rotation, pivot, pivot displacement of A then B, eta, xi, is_same_rigid_motion
        cases.append(Case(f"ccd_screw_{kind}", f"{kind.upper()} screw CCD (whole step)", 6 * n + 21, (_real("toi"),)))
    for kind in ("pt", "ee"):
        n = sum(STENCIL_VERTS[kind])
        # Rows: as ccd_screw_*, then t_start, t_end, distance_start, max_iters, is_same_rigid_motion
        cases.append(Case(f"ccd_screw_interval_{kind}", f"{kind.upper()} screw CCD (interval)", 6 * n + 25,
                          (_real("t"), _discrete("iters"), _discrete("status"))))
    cases += [
        # Rows: x, displacement, plane point, unit normal, eta, xi
        Case("ccd_ph", "PH CCD (linear)", 14, (_real("toi"), _discrete("is_active"))),
        # Rows: x, chord, rotation, pivot, pivot displacement, plane point, unit normal, eta, xi
        Case("ccd_ph_screw", "PH screw CCD", 23, (_real("toi"),)),
    ]
    for kind, (n_a, n_b) in STENCIL_VERTS.items():
        n = n_a + n_b
        # Rows: points, ends, inflations, speeds, thicknesses, d_hat, directional
        cases.append(Case(f"candidate_{kind}", f"{kind.upper()} broad-phase distance rule", 9 * n + 2,
                          (_discrete("verdict"),)))
    # Rows: x, end, plane point, unit normal, inflation, d_hat, xi
    cases.append(Case("candidate_ph", "PH broad-phase cull", 15, (_discrete("is_kept"),)))
    # Rows: t0, t1, t2, e0, e1
    cases.append(Case("intersect", "edge-triangle intersection", 15, (_discrete("is_hit"),)))
    return {case.name: case for case in cases}


CASES = _build_cases()


def n_outputs(case):
    return sum(output.width for output in case.outputs)
