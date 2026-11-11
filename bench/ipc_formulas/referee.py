"""50-digit exact values of the rows the pass1 and order clauses of compare.py leave, the referee of the parity.

    python bench/ipc_formulas/referee.py --summary acceptance-noreferee

Reads the failing rows of summary_<summary>.json in cases.RESULTS (every backend), evaluates the formula of their case
on the float64 inputs in high-precision arithmetic (mpmath, 120 digits, at least 50 of them exact), and writes
referee.npz to cases.DATA with '<case>/<set>/rows' and '<case>/<set>/exact' (NaN for the discrete outputs). The
edge-edge barrier rows are evaluated plain or mollified as CGQ's production dispatch decides (its flag and mollifier
test, which the row must already match). A mollified row is the product rule of m(I1) b(D) (ee_mollified_barrier.h)
projected onto the positive semi-definite cone through a 120-digit eigendecomposition. Every derivative of the
distance, the cross norm and the barrier is a central difference of its plain formula, independent of the analytic
derivatives of either side. A failing row of another case is reported and left without referee.
"""

import argparse
import json

import mpmath as mp
import numpy as np

from cases import CASES, DATA, HESSIAN_CLAMP_RATIO, MOLLIFIER_RATIO, RESULTS

mp.mp.dps = 120
# Central differences of step h lose the digits of h (gradient) and of h^2 (Hessian) to cancellation and are exact to
# h^2: 1e-30 leaves 60 exact digits of 120
FD_STEP = mp.mpf("1e-30")


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def dot(a, b):
    return sum(s * t for s, t in zip(a, b))


def sub(a, b):
    return [s - t for s, t in zip(a, b)]


def shifted(xs, steps):
    out = [list(x) for x in xs]
    for i, step in steps:
        out[i // 3][i % 3] += step
    return out


def gradient(f, xs):
    """Gradient over the 12 coordinates by central differences."""
    h = FD_STEP
    return mp.matrix([(f(shifted(xs, [(i, h)])) - f(shifted(xs, [(i, -h)]))) / (2 * h) for i in range(12)])


def hessian(f, xs):
    """Hessian over the 12 coordinates by central differences of central differences, exactly symmetric."""
    h = FD_STEP
    H = mp.matrix(12, 12)
    for i in range(12):
        for j in range(i, 12):
            value = (
                f(shifted(xs, [(i, h), (j, h)]))
                - f(shifted(xs, [(i, h), (j, -h)]))
                - f(shifted(xs, [(i, -h), (j, h)]))
                + f(shifted(xs, [(i, -h), (j, -h)]))
            ) / (4 * h * h)
            H[i, j] = H[j, i] = value
    return H


def flagged_distance2(flag):
    pc = bin(flag).count("1")

    def distance2(x):
        if pc == 4:
            n = cross(sub(x[1], x[0]), sub(x[3], x[2]))
            s = dot(sub(x[2], x[0]), n)
            return s * s / dot(n, n)
        if pc == 3:
            if (flag & 3) != 3:
                p, e0, e1 = x[0] if flag & 1 else x[1], x[2], x[3]
            else:
                p, e0, e1 = x[2] if flag & 4 else x[3], x[0], x[1]
            c = cross(sub(e0, p), sub(e1, p))
            return dot(c, c) / dot(sub(e1, e0), sub(e1, e0))
        a, b = x[0] if flag & 1 else x[1], x[2] if flag & 4 else x[3]
        return dot(sub(a, b), sub(a, b))

    return distance2


def cross_norm2(x):
    c = cross(sub(x[1], x[0]), sub(x[3], x[2]))
    return dot(c, c)


def curvature_eigenvalue(r, kappa):
    """8 kappa (r A^2 + (r^2 - 1) ln r / r + (r - 1) ln r A / 2), A = ln r + (r - 1) / r, r clamped from below."""
    r = max(r, mp.mpf(HESSIAN_CLAMP_RATIO))
    log_r = mp.log(r)
    log_term = log_r + (r - 1) / r
    return 8 * kappa * (r * log_term**2 + (r * r - 1) * log_r / r + (r - 1) * log_r * log_term / 2)


def barrier_ee_exact(row, flag, is_mollified):
    xs = [[mp.mpf(float(v)) for v in row[3 * k : 3 * k + 3]] for k in range(4)]
    rest = [[mp.mpf(float(v)) for v in row[12 + 3 * k : 15 + 3 * k]] for k in range(4)]
    d_hat, xi, kappa = (mp.mpf(float(v)) for v in row[24:27])
    distance2 = flagged_distance2(flag)
    D = distance2(xs)
    gD = gradient(distance2, xs)
    dH = d_hat * d_hat + 2 * d_hat * xi

    def barrier(D):
        r = (D - xi * xi) / dH
        return kappa * (r - 1) ** 2 * mp.log(r) ** 2

    if not is_mollified:
        dt = D - xi * xi
        r = dt / dH
        clamped = r if dt >= mp.mpf(HESSIAN_CLAMP_RATIO) * dH else mp.mpf(HESSIAN_CLAMP_RATIO)
        g = mp.diff(barrier, D) * gD
        H = curvature_eigenvalue(clamped, kappa) / (4 * dt * dH) * gD * gD.T
        return g, H
    b, b1, b2 = barrier(D), mp.diff(barrier, D), mp.diff(barrier, D, 2)
    HD = hessian(distance2, xs)
    I1 = cross_norm2(xs)
    gI = gradient(cross_norm2, xs)
    HI = hessian(cross_norm2, xs)
    eps = mp.mpf(MOLLIFIER_RATIO) * dot(sub(rest[0], rest[1]), sub(rest[0], rest[1])) * dot(
        sub(rest[2], rest[3]), sub(rest[2], rest[3])
    )
    m = 2 * I1 / eps - (I1 / eps) ** 2
    m1 = 2 / eps - 2 * I1 / eps**2
    m2 = -2 / eps**2
    gm = m1 * gI
    Hm = m2 * gI * gI.T + m1 * HI
    g = b * gm + m * b1 * gD
    H = b * Hm + b1 * (gm * gD.T + gD * gm.T) + m * (b2 * gD * gD.T + b1 * HD)
    eigenvalues, Q = mp.eigsy((H + H.T) / 2)
    return g, Q * mp.diag([max(e, 0) for e in eigenvalues]) * Q.T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", default="acceptance-noreferee")
    parser.add_argument("--reference", default="fmad-true")
    args = parser.parse_args()
    summary = json.loads((RESULTS / f"summary_{args.summary}.json").read_text())
    failing = {}
    for results in summary.values():
        for key, result in results.items():
            failing.setdefault(key, set()).update(result["failed_rows"])
    arrays = {}
    with np.load(DATA / "inputs.npz") as inputs, np.load(DATA / f"out_cgq_{args.reference}.npz") as reference:
        for key, rows in sorted(failing.items()):
            if not rows:
                continue
            case = CASES[key.split("/")[0]]
            if case.name != "barrier_ee":
                print(f"{key}: {len(rows)} failing rows, no referee for this case")
                continue
            rows = np.array(sorted(rows))
            exact = np.full((len(rows), sum(o.width for o in case.outputs)), np.nan)
            for i, i_r in enumerate(rows):
                flag, is_mollified = int(reference[key][i_r, 0]), bool(reference[key][i_r, 1])
                g, H = barrier_ee_exact(inputs[key][i_r], flag, is_mollified)
                exact[i, 2:14] = [float(v) for v in g]
                exact[i, 14:] = [float(H[a, b]) for a in range(12) for b in range(12)]
            arrays[f"{key}/rows"] = rows
            arrays[f"{key}/exact"] = exact
            print(f"{key}: {len(rows)} rows evaluated", flush=True)
    np.savez(DATA / "referee.npz", **arrays)
    print(f"wrote {DATA / 'referee.npz'}")


if __name__ == "__main__":
    main()
