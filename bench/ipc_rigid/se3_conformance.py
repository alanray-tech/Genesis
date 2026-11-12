"""Conformance of the SE(3) maps of the rigid contact proxies (genesis/engine/ipc/se3.py) with CGQ aeb58956.

The oracle is CGQ's own dense reference, tests/rigid_proxy_kkt_reference.py: the maps A and B of dc = A dg - B dq and
the FK-defect prefix cap, the rotation of the constraint c coming from CGQ's device logarithm (the reference's matrix
logarithm, compared too, holds away from pi only). Results are split by the angle of the rotation of c. Genesis evaluates c, the rotation blocks of the tangent map S = A^-1 B and of
the normal map N = A^-1, the particular step d_p = -A^-1 c and the cap in a kernel, in double precision. Checked:
c and the cap against CGQ's, S J against A^-1 B(J) and N against A^-1 for random link Jacobians J, and the identities
C P = 0 and C d_p = -c with CGQ's A and B and Genesis's maps.

    python bench/ipc_rigid/se3_conformance.py --backend gpu|cpu [--fast-math 0|1]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.ipc.se3 import func_fk_defect_prefix_cap, func_proxy_constraint, func_proxy_maps

CGQ_TESTS = Path(r"C:\Users\81946\Projects\CudaGraphQIPC\cuda-graph-qipc-p2a-aeb58956\tests")
sys.path.insert(0, str(CGQ_TESTS))
import rigid_proxy_kkt_reference as ref  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="gpu")
parser.add_argument("--fast-math", type=int, default=1)
parser.add_argument("--n", type=int, default=20000)
parser.add_argument("--results", default=str(Path(__file__).parent / "results"))
args = parser.parse_args()
gs.init(backend=getattr(gs, args.backend), precision="64", logging_level="warning", debug=not args.fast_math)


def quat_to_R(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotvec_to_quat(v):
    angle = np.linalg.norm(v)
    axis = v / angle if angle > 0 else np.array([1.0, 0.0, 0.0])
    return np.concatenate(([np.cos(angle / 2)], np.sin(angle / 2) * axis))


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


rng = np.random.default_rng(0)
n = args.n
# The rotation of the constraint spans the regimes of a proxy: tiny defects (the converged state), small, moderate and
# close to pi, around random link orientations
angles = np.concatenate(
    [
        10.0 ** rng.uniform(-12, -5, n // 4),
        10.0 ** rng.uniform(-5, -1, n // 4),
        rng.uniform(0.1, 3.0, n // 4),
        np.pi - 10.0 ** rng.uniform(-9, -2, n - 3 * (n // 4)),
    ]
)
axes = rng.normal(size=(n, 3))
axes /= np.linalg.norm(axes, axis=1, keepdims=True)
link_quat = rng.normal(size=(n, 4))
link_quat /= np.linalg.norm(link_quat, axis=1, keepdims=True)
proxy_quat = np.array([quat_mul(rotvec_to_quat(a * ax), q) for a, ax, q in zip(angles, axes, link_quat)])
link_pos = rng.normal(size=(n, 3))
proxy_pos = link_pos + rng.normal(size=(n, 3)) * 10.0 ** rng.uniform(-12, -1, (n, 1))
# FK-defect cap inputs: h0 <= limit, both signs of h_slack - h0, zero and positive curvatures
h0 = 10.0 ** rng.uniform(-9, -3, n)
limit = h0 + 10.0 ** rng.uniform(-9, -3, n)
h_slack = np.where(rng.uniform(size=n) < 0.3, 0.0, 10.0 ** rng.uniform(-9, -2, n))
curvature = np.where(rng.uniform(size=n) < 0.2, 0.0, 10.0 ** rng.uniform(-6, 3, n))


@qd.kernel
def kernel_maps(
    proxy_pos: qd.types.ndarray(),
    proxy_quat: qd.types.ndarray(),
    link_pos: qd.types.ndarray(),
    link_quat: qd.types.ndarray(),
    h0: qd.types.ndarray(),
    h_slack: qd.types.ndarray(),
    curvature: qd.types.ndarray(),
    limit: qd.types.ndarray(),
    constraint: qd.types.ndarray(),
    tangent: qd.types.ndarray(),
    normal: qd.types.ndarray(),
    particular: qd.types.ndarray(),
    cap: qd.types.ndarray(),
):
    for i in range(proxy_pos.shape[0]):
        c = func_proxy_constraint(
            qd.Vector([proxy_pos[i, k] for k in qd.static(range(3))], dt=gs.qd_float),
            qd.Vector([proxy_quat[i, k] for k in qd.static(range(4))], dt=gs.qd_float),
            qd.Vector([link_pos[i, k] for k in qd.static(range(3))], dt=gs.qd_float),
            qd.Vector([link_quat[i, k] for k in qd.static(range(4))], dt=gs.qd_float),
        )
        tangent_rotation, normal_rotation, dp = func_proxy_maps(c)
        for k in qd.static(range(6)):
            constraint[i, k] = c[k]
            particular[i, k] = dp[k]
        for a, b in qd.static(qd.ndrange(3, 3)):
            tangent[i, a, b] = tangent_rotation[a, b]
            normal[i, a, b] = normal_rotation[a, b]
        cap[i] = func_fk_defect_prefix_cap(h0[i], h_slack[i], curvature[i], limit[i])


constraint = np.zeros((n, 6))
tangent = np.zeros((n, 3, 3))
normal = np.zeros((n, 3, 3))
particular = np.zeros((n, 6))
cap = np.zeros(n)
kernel_maps(
    proxy_pos,
    proxy_quat,
    link_pos,
    link_quat,
    h0,
    h_slack,
    curvature,
    limit,
    constraint,
    tangent,
    normal,
    particular,
    cap,
)


def cgq_quat_log_map(q):
    """CGQ's device logarithm (rigid_quat_utils.h quat_log_map), hemisphere-aligned, which its constraint kernel runs.
    The dense reference takes the logarithm of the rotation matrix, which loses the axis close to pi."""
    q = -q if q[0] < 0.0 else q
    u = q[1:]
    s = np.linalg.norm(u)
    if s < 1e-9:
        return (2.0 / q[0]) * u
    return (2.0 * np.arctan2(s, q[0]) / s) * u


rows = {key: [] for key in ("constraint", "tangent_J", "normal", "C_P", "C_dp", "cap")}
for i in range(n):
    J = rng.normal(size=(6, 7))
    q_rel = quat_mul(proxy_quat[i], link_quat[i] * np.array([1.0, -1.0, -1.0, -1.0]))
    c_ref = np.concatenate((proxy_pos[i] - link_pos[i], cgq_quat_log_map(q_rel)))
    A_ref = np.eye(6)
    A_ref[3:, 3:] = ref.so3_left_jacobian_inverse(c_ref[3:])
    B_ref = J.copy()
    B_ref[3:, :] = ref.so3_right_jacobian_inverse(c_ref[3:]) @ J[3:, :]
    A_inv = np.linalg.inv(A_ref)
    S = np.eye(6)
    S[3:, 3:] = tangent[i]
    N = np.eye(6)
    N[3:, 3:] = normal[i]
    scale_c = 1.0 + np.abs(c_ref).max()
    rows["constraint"].append(np.abs(constraint[i] - c_ref).max() / scale_c)
    rows["tangent_J"].append(np.abs(S @ J - A_inv @ B_ref).max() / (1.0 + np.abs(A_inv @ B_ref).max()))
    rows["normal"].append(np.abs(N - A_inv).max())
    rows["C_P"].append(np.abs(A_ref @ S @ J - B_ref).max() / (1.0 + np.abs(B_ref).max()))
    rows["C_dp"].append(np.abs(A_ref @ particular[i] + c_ref).max() / scale_c)
    rows["cap"].append(abs(cap[i] - ref.fk_defect_prefix_cap(h0[i], h_slack[i], curvature[i], limit[i])))

# The dense reference agrees with CGQ's device logarithm away from pi, which the first three regimes check
for i in range(3 * (n // 4)):
    R_p, R_l = quat_to_R(proxy_quat[i]), quat_to_R(link_quat[i])
    c_dense = ref.pose_constraint(proxy_pos[i], R_p, link_pos[i], R_l)
    rows.setdefault("constraint_vs_dense", []).append(
        np.abs(constraint[i] - c_dense).max() / (1.0 + np.abs(c_dense).max())
    )

regimes = {"tiny": slice(0, n // 4), "small": slice(n // 4, n // 2), "moderate": slice(n // 2, 3 * (n // 4))}
regimes["near_pi"] = slice(3 * (n // 4), n)
summary = {"backend": args.backend, "fast_math": bool(args.fast_math), "n": n}
for key, values in rows.items():
    values = np.array(values)
    summary[key] = {name: float(values[part].max()) for name, part in regimes.items() if len(values[part]) > 0}
    print(f"{key:20s} " + ", ".join(f"{name} {value:.3e}" for name, value in summary[key].items()))
out = Path(args.results)
out.mkdir(parents=True, exist_ok=True)
tag = f"{args.backend}{'' if args.fast_math else '_ieee'}"
(out / f"se3_conformance_{tag}.json").write_text(json.dumps(summary, indent=1))
