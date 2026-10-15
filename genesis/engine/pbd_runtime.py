"""Free kernels and functions of PBD simulation over PBDSolverSystem-owned data."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.spatial_hash import func_compute_reordered_idx, func_grid_to_slot, func_pos_to_grid


@qd.func
def poly6(system: qd.template(), dist):
    # dist is a VECTOR
    result = gs.qd_float(0.0)
    d = dist.norm() / system.dist_scale
    if 0 < d < system.h:
        rhs = (system.h_2 - d * d) * (system.h_2 - d * d) * (system.h_2 - d * d)
        result = system.poly6_Coe * rhs / system.h_9
    return result


@qd.func
def poly6_scalar(system: qd.template(), dist):
    # dist is a SCALAR
    result = gs.qd_float(0.0)
    d = dist
    if 0 < d < system.h:
        rhs = (system.h_2 - d * d) * (system.h_2 - d * d) * (system.h_2 - d * d)
        result = system.poly6_Coe * rhs / system.h_9
    return result


@qd.func
def spiky(system: qd.template(), dist):
    # dist is a VECTOR
    result = qd.Vector.zero(gs.qd_float, 3)
    d = dist.norm() / system.dist_scale
    if 0 < d < system.h:
        m = (system.h - d) * (system.h - d)
        result = (system.spiky_Coe * m / (system.h_6 * d)) * dist / system.dist_scale
    return result


@qd.func
def S_Corr(system: qd.template(), dist):
    upper = poly6(system, dist)
    lower = poly6_scalar(system, system.S_Corr_delta_q)
    m = upper / lower
    return -1.0 * system.S_Corr_k * m * m * m * m


@qd.func(requires_top_level=True)
def func_store_initial_pos(system: qd.template(), f: int):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        system.particles.ipos[i_p, i_b] = system.particles.pos[i_p, i_b]


@qd.func(requires_top_level=True)
def func_reorder_particles(system: qd.template(), f: int):
    func_compute_reordered_idx(
        system._n_particles,
        system.particles.pos,
        system.particles_ng.active,
        system.particles_ng.reordered_idx,
        system.spatial_hasher,
        system.sh,
    )

    # copy to reordered
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        system.particles_ng_reordered.active[i_p, i_b] = False
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng.active[i_p, i_b]:
            reordered_idx = system.particles_ng.reordered_idx[i_p, i_b]

            system.particles_reordered.free[reordered_idx, i_b] = system.particles.free[i_p, i_b]
            system.particles_reordered.pos[reordered_idx, i_b] = system.particles.pos[i_p, i_b]
            system.particles_reordered.ipos[reordered_idx, i_b] = system.particles.ipos[i_p, i_b]
            system.particles_reordered.dpos[reordered_idx, i_b] = system.particles.dpos[i_p, i_b]
            system.particles_reordered.vel[reordered_idx, i_b] = system.particles.vel[i_p, i_b]
            system.particles_reordered.lam[reordered_idx, i_b] = system.particles.lam[i_p, i_b]
            system.particles_reordered.rho[reordered_idx, i_b] = system.particles.rho[i_p, i_b]
            system.particles_info_reordered.mass[reordered_idx, i_b] = system.particles_info.mass[i_p]
            system.particles_info_reordered.pos_rest[reordered_idx, i_b] = system.particles_info.pos_rest[i_p]
            system.particles_info_reordered.rho_rest[reordered_idx, i_b] = system.particles_info.rho_rest[i_p]
            system.particles_info_reordered.material_type[reordered_idx, i_b] = system.particles_info.material_type[i_p]
            system.particles_info_reordered.mu_s[reordered_idx, i_b] = system.particles_info.mu_s[i_p]
            system.particles_info_reordered.mu_k[reordered_idx, i_b] = system.particles_info.mu_k[i_p]
            system.particles_info_reordered.air_resistance[reordered_idx, i_b] = system.particles_info.air_resistance[
                i_p
            ]
            system.particles_info_reordered.density_relaxation[reordered_idx, i_b] = (
                system.particles_info.density_relaxation[i_p]
            )
            system.particles_info_reordered.viscosity_relaxation[reordered_idx, i_b] = (
                system.particles_info.viscosity_relaxation[i_p]
            )
            system.particles_ng_reordered.active[reordered_idx, i_b] = system.particles_ng.active[i_p, i_b]


@qd.func(requires_top_level=True)
def func_apply_external_force(system: qd.template(), f: int, t: qd.f32):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles.free[i_p, i_b]:
            # gravity
            system.particles.vel[i_p, i_b] = system.particles.vel[i_p, i_b] + system._gravity[i_b] * system._substep_dt

            # external force fields
            acc = qd.Vector.zero(gs.qd_float, 3)
            for i_ff in qd.static(range(len(system._ffs))):
                acc += system._ffs[i_ff].get_acc(system.particles.pos[i_p, i_b], system.particles.vel[i_p, i_b], t, i_p)
            system.particles.vel[i_p, i_b] = system.particles.vel[i_p, i_b] + acc * system._substep_dt

            if system.particles_info.material_type[i_p] == system.MATERIAL.CLOTH:
                f_air_resistance = (
                    system.particles_info.air_resistance[i_p]
                    * system.particles.vel[i_p, i_b].norm()
                    * system.particles.vel[i_p, i_b]
                )
                system.particles.vel[i_p, i_b] = (
                    system.particles.vel[i_p, i_b]
                    - f_air_resistance / system.particles_info.mass[i_p] * system._substep_dt
                )

        # attached particles are not free but still need to update position to follow the link
        system.particles.pos[i_p, i_b] = (
            system.particles.pos[i_p, i_b] + system.particles.vel[i_p, i_b] * system._substep_dt
        )


@qd.func(requires_top_level=True)
def func_solve_stretch(system: qd.template(), f: int):
    for _ in qd.static(range(system._max_stretch_solver_iterations)):
        for i_e, i_b in qd.ndrange(system._n_edges, system._B):
            v1 = system.edges_info.v1[i_e]
            v2 = system.edges_info.v2[i_e]

            w1 = system.particles.free[v1, i_b] / system.particles_info.mass[v1]
            w2 = system.particles.free[v2, i_b] / system.particles_info.mass[v2]
            n = system.particles.pos[v1, i_b] - system.particles.pos[v2, i_b]
            C = n.norm() - system.edges_info.len_rest[i_e]
            alpha = system.edges_info.stretch_compliance[i_e] / (system._substep_dt**2)
            dp = -C / (w1 + w2 + alpha) * n / n.norm(gs.EPS) * system.edges_info.stretch_relaxation[i_e]
            system.particles.dpos[v1, i_b] += dp * w1
            system.particles.dpos[v2, i_b] -= dp * w2

        for i_p, i_b in qd.ndrange(system._n_particles, system._B):
            if system.particles.free[i_p, i_b] and system.particles_info.material_type[i_p] != system.MATERIAL.PARTICLE:
                system.particles.pos[i_p, i_b] = system.particles.pos[i_p, i_b] + system.particles.dpos[i_p, i_b]
                system.particles.dpos[i_p, i_b].fill(0)


@qd.func(requires_top_level=True)
def func_solve_bending(system: qd.template(), f: int):
    for _ in qd.static(range(system._max_bending_solver_iterations)):
        for i_ie, i_b in qd.ndrange(system._n_inner_edges, system._B):  # 140 - 142
            v1 = system.inner_edges_info.v1[i_ie]
            v2 = system.inner_edges_info.v2[i_ie]
            v3 = system.inner_edges_info.v3[i_ie]
            v4 = system.inner_edges_info.v4[i_ie]

            w1 = system.particles.free[v1, i_b] / system.particles_info.mass[v1]
            w2 = system.particles.free[v2, i_b] / system.particles_info.mass[v2]
            w3 = system.particles.free[v3, i_b] / system.particles_info.mass[v3]
            w4 = system.particles.free[v4, i_b] / system.particles_info.mass[v4]

            if w1 + w2 + w3 + w4 > 0.0:
                # https://matthias-research.github.io/pages/publications/posBasedDyn.pdf
                # Appendix A: Bending Constraint Projection
                p2 = system.particles.pos[v2, i_b] - system.particles.pos[v1, i_b]
                p3 = system.particles.pos[v3, i_b] - system.particles.pos[v1, i_b]
                p4 = system.particles.pos[v4, i_b] - system.particles.pos[v1, i_b]
                l23 = p2.cross(p3).norm()
                l24 = p2.cross(p4).norm()
                n1 = p2.cross(p3) / l23
                n2 = p2.cross(p4) / l24
                d = qd.math.clamp(n1.dot(n2), -1.0, 1.0)

                q3 = (p2.cross(n2) + n1.cross(p2) * d) / l23  # eq. (25)
                q4 = (p2.cross(n1) + n2.cross(p2) * d) / l24  # eq. (26)
                q2 = -(p3.cross(n2) + n1.cross(p3) * d) / l23 - (p4.cross(n1) + n2.cross(p4) * d) / l24  # eq. (27)
                q1 = -q2 - q3 - q4
                # eq. (29)
                sum_wq = w1 * q1.norm_sqr() + w2 * q2.norm_sqr() + w3 * q3.norm_sqr() + w4 * q4.norm_sqr()
                constraint = qd.acos(d) - qd.acos(-1.0)

                # XPBD
                alpha = system.inner_edges_info.bending_compliance[i_ie] / (system._substep_dt**2)
                constraint = (
                    -qd.sqrt(1 - d**2)
                    * constraint
                    / (sum_wq + alpha)
                    * system.inner_edges_info.bending_relaxation[i_ie]
                )

                system.particles.dpos[v1, i_b] += w1 * constraint * q1
                system.particles.dpos[v2, i_b] += w2 * constraint * q2
                system.particles.dpos[v3, i_b] += w3 * constraint * q3
                system.particles.dpos[v4, i_b] += w4 * constraint * q4

        for i_p, i_b in qd.ndrange(system._n_particles, system._B):
            if system.particles.free[i_p, i_b] and system.particles_info.material_type[i_p] != system.MATERIAL.PARTICLE:
                system.particles.pos[i_p, i_b] = system.particles.pos[i_p, i_b] + system.particles.dpos[i_p, i_b]
                system.particles.dpos[i_p, i_b].fill(0)


@qd.func(requires_top_level=True)
def func_solve_volume(system: qd.template(), f: int):
    for _ in qd.static(range(system._max_volume_solver_iterations)):
        for i_el, i_b in qd.ndrange(system._n_elems, system._B):
            v1 = system.elems_info.v1[i_el]
            v2 = system.elems_info.v2[i_el]
            v3 = system.elems_info.v3[i_el]
            v4 = system.elems_info.v4[i_el]

            p1 = system.particles.pos[v1, i_b]
            p2 = system.particles.pos[v2, i_b]
            p3 = system.particles.pos[v3, i_b]
            p4 = system.particles.pos[v4, i_b]

            grad1 = (p4 - p2).cross(p3 - p2) / 6.0
            grad2 = (p3 - p1).cross(p4 - p1) / 6.0
            grad3 = (p4 - p1).cross(p2 - p1) / 6.0
            grad4 = (p2 - p1).cross(p3 - p1) / 6.0

            w1 = system.particles.free[v1, i_b] / system.particles_info.mass[v1] * grad1.norm_sqr()
            w2 = system.particles.free[v2, i_b] / system.particles_info.mass[v2] * grad2.norm_sqr()
            w3 = system.particles.free[v3, i_b] / system.particles_info.mass[v3] * grad3.norm_sqr()
            w4 = system.particles.free[v4, i_b] / system.particles_info.mass[v4] * grad4.norm_sqr()

            if w1 + w2 + w3 + w4 > 0.0:
                vol = gu.qd_tet_vol(p1, p2, p3, p4)
                C = vol - system.elems_info.vol_rest[i_el]
                alpha = system.elems_info.volume_compliance[i_el] / (system._substep_dt**2)
                s = -C / (w1 + w2 + w3 + w4 + alpha) * system.elems_info.volume_relaxation[i_el]

                system.particles.dpos[v1, i_b] += s * w1 * grad1
                system.particles.dpos[v2, i_b] += s * w2 * grad2
                system.particles.dpos[v3, i_b] += s * w3 * grad3
                system.particles.dpos[v4, i_b] += s * w4 * grad4

        for i_p, i_b in qd.ndrange(system._n_particles, system._B):
            if system.particles.free[i_p, i_b] and system.particles_info.material_type[i_p] != system.MATERIAL.PARTICLE:
                system.particles.pos[i_p, i_b] = system.particles.pos[i_p, i_b] + system.particles.dpos[i_p, i_b]
                system.particles.dpos[i_p, i_b].fill(0)


@qd.func
def _func_solve_collision(system: qd.template(), i, j, i_b):
    """j -> i"""

    cur_dist = (system.particles_reordered.pos[i, i_b] - system.particles_reordered.pos[j, i_b]).norm(gs.EPS)
    rest_dist = (
        system.particles_info_reordered.pos_rest[i, i_b] - system.particles_info_reordered.pos_rest[j, i_b]
    ).norm(gs.EPS)
    target_dist = system._particle_size  # target particle distance is 2 * particle radius, i.e. particle_size
    if cur_dist < target_dist and rest_dist > target_dist:
        wi = system.particles_reordered.free[i, i_b] / system.particles_info_reordered.mass[i, i_b]
        wj = system.particles_reordered.free[j, i_b] / system.particles_info_reordered.mass[j, i_b]
        n = (system.particles_reordered.pos[i, i_b] - system.particles_reordered.pos[j, i_b]) / cur_dist

        ### resolve collision ###
        system.particles_reordered.dpos[i, i_b] += wi / (wi + wj) * (target_dist - cur_dist) * n

        ### apply friction ###
        # https://mmacklin.com/uppfrta_preprint.pdf
        # equation (23)
        dv = (system.particles_reordered.pos[i, i_b] - system.particles_reordered.ipos[i, i_b]) - (
            system.particles_reordered.pos[j, i_b] - system.particles_reordered.ipos[j, i_b]
        )
        dpos = -(dv - n * n.dot(dv))
        # equation (24)
        d = target_dist - cur_dist
        mu_s = qd.max(system.particles_info_reordered.mu_s[i, i_b], system.particles_info_reordered.mu_s[j, i_b])
        mu_k = qd.max(system.particles_info_reordered.mu_k[i, i_b], system.particles_info_reordered.mu_k[j, i_b])
        if dpos.norm() < mu_s * d:
            system.particles_reordered.dpos[i, i_b] += wi / (wi + wj) * dpos
        else:
            system.particles_reordered.dpos[i, i_b] += wi / (wi + wj) * dpos * qd.min(1.0, mu_k * d / dpos.norm(gs.EPS))


@qd.func(requires_top_level=True)
def func_solve_collision(system: qd.template(), f: int):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_info_reordered.material_type[i_p, i_b] != system.MATERIAL.PARTICLE:
            base = func_pos_to_grid(system.particles_reordered.pos[i_p, i_b], system.sh)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = func_grid_to_slot(base + offset, system.sh)
                for j in range(
                    system.spatial_hasher.slot_start[slot_idx, i_b],
                    system.spatial_hasher.slot_size[slot_idx, i_b] + system.spatial_hasher.slot_start[slot_idx, i_b],
                ):
                    if (
                        i_p != j
                        and (system.particles_reordered.free[i_p, i_b] or system.particles_reordered.free[j, i_b])
                        and not (
                            system.particles_info_reordered.material_type[i_p, i_b] == system.MATERIAL.LIQUID
                            and system.particles_info_reordered.material_type[j, i_b] == system.MATERIAL.LIQUID
                        )
                    ):
                        _func_solve_collision(system, i_p, j, i_b)

    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if (
            system.particles_reordered.free[i_p, i_b]
            and system.particles_info_reordered.material_type[i_p, i_b] != system.MATERIAL.PARTICLE
        ):
            system.particles_reordered.pos[i_p, i_b] = (
                system.particles_reordered.pos[i_p, i_b] + system.particles_reordered.dpos[i_p, i_b]
            )
            system.particles_reordered.dpos[i_p, i_b].fill(0)


@qd.func(requires_top_level=True)
def func_solve_boundary_collision(system: qd.template(), f: int):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        # boundary is enforced regardless of whether free
        pos_new, vel_new = system.boundary.impose_pos_vel(
            system.particles.pos[i_p, i_b], system.particles.vel[i_p, i_b]
        )
        system.particles.pos[i_p, i_b] = pos_new
        system.particles.vel[i_p, i_b] = vel_new


@qd.func(requires_top_level=True)
def func_solve_density(system: qd.template(), f: int):
    for _ in qd.static(range(system._max_density_solver_iterations)):
        # ---Calculate lambdas---
        for i_p, i_b in qd.ndrange(system._n_particles, system._B):
            if system.particles_info_reordered.material_type[i_p, i_b] == system.MATERIAL.LIQUID:
                pos_i = system.particles_reordered.pos[i_p, i_b]
                base = func_pos_to_grid(pos_i, system.sh)
                lower_sum = gs.qd_float(0.0)
                rho = gs.qd_float(0.0)
                spiky_i = qd.Vector.zero(gs.qd_float, 3)
                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    slot_idx = func_grid_to_slot(base + offset, system.sh)
                    for j in range(
                        system.spatial_hasher.slot_start[slot_idx, i_b],
                        system.spatial_hasher.slot_size[slot_idx, i_b]
                        + system.spatial_hasher.slot_start[slot_idx, i_b],
                    ):
                        pos_j = system.particles_reordered.pos[j, i_b]
                        # ---Poly6---
                        rho += poly6(system, pos_i - pos_j) * system.particles_info_reordered.mass[j, i_b]
                        # ---Spiky---
                        s = spiky(system, pos_i - pos_j) / system.particles_info_reordered.rho_rest[i_p, i_b]
                        spiky_i += s
                        lower_sum += s.dot(s)
                constraint = (rho / system.particles_info_reordered.rho_rest[i_p, i_b]) - 1.0
                lower_sum += spiky_i.dot(spiky_i)
                system.particles_reordered.lam[i_p, i_b] = -1.0 * (constraint / (lower_sum + system.lambda_epsilon))

        # ---Calculate delta pos---
        for i_p, i_b in qd.ndrange(system._n_particles, system._B):
            if system.particles_info_reordered.material_type[i_p, i_b] == system.MATERIAL.LIQUID:
                pos_i = system.particles_reordered.pos[i_p, i_b]
                base = func_pos_to_grid(pos_i, system.sh)
                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    slot_idx = func_grid_to_slot(base + offset, system.sh)
                    for j in range(
                        system.spatial_hasher.slot_start[slot_idx, i_b],
                        system.spatial_hasher.slot_size[slot_idx, i_b]
                        + system.spatial_hasher.slot_start[slot_idx, i_b],
                    ):
                        if i_p != j:
                            pos_j = system.particles_reordered.pos[j, i_b]
                            # ---S_Corr---
                            scorr = S_Corr(system, pos_i - pos_j)
                            left = (
                                system.particles_reordered.lam[i_p, i_b]
                                + system.particles_reordered.lam[j, i_b]
                                + scorr
                            )
                            right = spiky(system, pos_i - pos_j)
                            system.particles_reordered.dpos[i_p, i_b] = (
                                system.particles_reordered.dpos[i_p, i_b]
                                + left
                                * right
                                / system.particles_info_reordered.rho_rest[i_p, i_b]
                                * system.dist_scale
                                * system.particles_info_reordered.density_relaxation[i_p, i_b]
                            )

        for i_p, i_b in qd.ndrange(system._n_particles, system._B):
            if (
                system.particles_info_reordered.material_type[i_p, i_b] == system.MATERIAL.LIQUID
                and system.particles_reordered.free[i_p, i_b]
            ):
                system.particles_reordered.pos[i_p, i_b] = (
                    system.particles_reordered.pos[i_p, i_b] + system.particles_reordered.dpos[i_p, i_b]
                )
                system.particles_reordered.dpos[i_p, i_b].fill(0)


@qd.func(requires_top_level=True)
def func_solve_viscosity(system: qd.template(), f: int):
    for _ in qd.static(range(system._max_viscosity_solver_iterations)):
        for i_p, i_b in qd.ndrange(system._n_particles, system._B):
            if system.particles_info_reordered.material_type[i_p, i_b] == system.MATERIAL.LIQUID:
                pos_i = system.particles_reordered.pos[i_p, i_b]
                base = func_pos_to_grid(pos_i, system.sh)
                xsph_sum = qd.Vector.zero(gs.qd_float, 3)
                omega_sum = qd.Vector.zero(gs.qd_float, 3)
                # -For Gradient Approx.-
                dx_sum = qd.Vector.zero(gs.qd_float, 3)
                dy_sum = qd.Vector.zero(gs.qd_float, 3)
                dz_sum = qd.Vector.zero(gs.qd_float, 3)
                n_dx_sum = qd.Vector.zero(gs.qd_float, 3)
                n_dy_sum = qd.Vector.zero(gs.qd_float, 3)
                n_dz_sum = qd.Vector.zero(gs.qd_float, 3)
                dx = qd.Vector([system.g_del, 0.0, 0.0], dt=gs.qd_float)
                dy = qd.Vector([0.0, system.g_del, 0.0], dt=gs.qd_float)
                dz = qd.Vector([0.0, 0.0, system.g_del], dt=gs.qd_float)

                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    slot_idx = func_grid_to_slot(base + offset, system.sh)
                    for j in range(
                        system.spatial_hasher.slot_start[slot_idx, i_b],
                        system.spatial_hasher.slot_size[slot_idx, i_b]
                        + system.spatial_hasher.slot_start[slot_idx, i_b],
                    ):
                        pos_j = system.particles_reordered.pos[j, i_b]
                        v_ij = (system.particles_reordered.pos[j, i_b] - system.particles_reordered.ipos[j, i_b]) - (
                            system.particles_reordered.pos[i_p, i_b] - system.particles_reordered.ipos[i_p, i_b]
                        )

                        dist = pos_i - pos_j
                        # ---Vorticity---
                        omega_sum += v_ij.cross(spiky(system, dist))
                        # -Gradient Approx.-
                        dx_sum += v_ij.cross(spiky(system, dist + dx))
                        dy_sum += v_ij.cross(spiky(system, dist + dy))
                        dz_sum += v_ij.cross(spiky(system, dist + dz))
                        n_dx_sum += v_ij.cross(spiky(system, dist - dx))
                        n_dy_sum += v_ij.cross(spiky(system, dist - dy))
                        n_dz_sum += v_ij.cross(spiky(system, dist - dz))
                        # ---Viscosity---
                        poly = poly6(system, dist)
                        xsph_sum += poly * v_ij

                # # ---Vorticity---
                # n_x = (dx_sum.norm() - n_dx_sum.norm()) / (2 * system.g_del)
                # n_y = (dy_sum.norm() - n_dy_sum.norm()) / (2 * system.g_del)
                # n_z = (dz_sum.norm() - n_dz_sum.norm()) / (2 * system.g_del)
                # n = qd.Vector([n_x, n_y, n_z])
                # big_n = n.normalized()
                # if not omega_sum.norm() == 0.0:
                #     vorticity[p] = vorqd_epsilon * big_n.cross(omega_sum)

                # ---Viscosity---
                system.particles_reordered.dpos[i_p, i_b] = (
                    system.particles_reordered.dpos[i_p, i_b]
                    + xsph_sum * system.particles_info_reordered.viscosity_relaxation[i_p, i_b]
                )

        for i_p, i_b in qd.ndrange(system._n_particles, system._B):
            if (
                system.particles_info_reordered.material_type[i_p, i_b] == system.MATERIAL.LIQUID
                and system.particles_reordered.free[i_p, i_b]
            ):
                system.particles_reordered.pos[i_p, i_b] = (
                    system.particles_reordered.pos[i_p, i_b] + system.particles_reordered.dpos[i_p, i_b]
                )
                system.particles_reordered.dpos[i_p, i_b].fill(0)


@qd.func(requires_top_level=True)
def func_compute_velocity(system: qd.template(), f: int):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        system.particles_reordered.vel[i_p, i_b] = (
            system.particles_reordered.pos[i_p, i_b] - system.particles_reordered.ipos[i_p, i_b]
        ) / system._substep_dt


@qd.func(requires_top_level=True)
def func_copy_from_reordered(system: qd.template(), f: int):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng.active[i_p, i_b]:
            reordered_idx = system.particles_ng.reordered_idx[i_p, i_b]
            system.particles.free[i_p, i_b] = system.particles_reordered.free[reordered_idx, i_b]
            system.particles.pos[i_p, i_b] = system.particles_reordered.pos[reordered_idx, i_b]
            system.particles.ipos[i_p, i_b] = system.particles_reordered.ipos[reordered_idx, i_b]
            system.particles.dpos[i_p, i_b] = system.particles_reordered.dpos[reordered_idx, i_b]
            system.particles.vel[i_p, i_b] = system.particles_reordered.vel[reordered_idx, i_b]
            system.particles.lam[i_p, i_b] = system.particles_reordered.lam[reordered_idx, i_b]
            system.particles.rho[i_p, i_b] = system.particles_reordered.rho[reordered_idx, i_b]


@qd.kernel
def kernel_pre_coupling(system: qd.template(), f: int, t: qd.f32):
    func_store_initial_pos(system, f)
    func_apply_external_force(system, f, t)
    if qd.static(system._n_edges > 0):
        func_solve_stretch(system, f)
    if qd.static(system._n_inner_edges > 0):
        func_solve_bending(system, f)
    if qd.static(system._n_elems > 0):
        func_solve_volume(system, f)
    func_reorder_particles(system, f)
    func_solve_density(system, f)
    func_solve_viscosity(system, f)
    func_solve_collision(system, f)
    func_compute_velocity(system, f)


@qd.kernel
def kernel_post_coupling(system: qd.template(), f: int):
    func_copy_from_reordered(system, f)
    func_solve_boundary_collision(system, f)


@qd.kernel
def _kernel_set_state(
    system: qd.template(),
    f: qd.i32,
    pos: qd.types.ndarray(),  # shape [B, _n_particles, 3]
    vel: qd.types.ndarray(),  # shape [B, _n_particles, 3]
    free: qd.types.ndarray(),  # shape [B, _n_particles]
):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        for j in qd.static(range(3)):
            system.particles.pos[i_p, i_b][j] = pos[i_b, i_p, j]
            system.particles.vel[i_p, i_b][j] = vel[i_b, i_p, j]
        system.particles.free[i_p, i_b] = free[i_b, i_p]


@qd.kernel
def _kernel_get_state(
    system: qd.template(),
    f: qd.i32,
    pos: qd.types.ndarray(),  # shape [B, _n_particles, 3]
    vel: qd.types.ndarray(),  # shape [B, _n_particles, 3]
    free: qd.types.ndarray(),  # shape [B, _n_particles]
):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        for j in qd.static(range(3)):
            pos[i_b, i_p, j] = system.particles.pos[i_p, i_b][j]
            vel[i_b, i_p, j] = system.particles.vel[i_p, i_b][j]
        free[i_b, i_p] = qd.cast(system.particles.free[i_p, i_b], gs.qd_bool)


@qd.kernel
def _kernel_update_render_fields(system: qd.template(), f: qd.i32):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng.active[i_p, i_b]:
            system.particles_render.pos[i_p, i_b] = system.particles.pos[i_p, i_b]
            system.particles_render.vel[i_p, i_b] = system.particles.vel[i_p, i_b]
        else:
            system.particles_render.pos[i_p, i_b] = gu.qd_nowhere()
        system.particles_render.active[i_p, i_b] = system.particles_ng.active[i_p, i_b]

    for i_v, i_b in qd.ndrange(system._n_vverts, system._B):
        vvert_pos = qd.Vector.zero(gs.qd_float, 3)
        for j in range(system._n_vvert_supports):
            i_p = system.vverts_info.support_idxs[i_v][j]
            vvert_pos += system.particles.pos[i_p, i_b] * system.vverts_info.support_weights[i_v][j]
        system.vverts_render.pos[i_v, i_b] = vvert_pos
        i_p = system.vverts_info.support_idxs[i_v][0]
        system.vverts_render.active[i_v, i_b] = system.particles_render.active[i_p, i_b]


@qd.kernel
def _kernel_set_particles_pos(
    system: qd.template(),
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        for i in qd.static(range(3)):
            system.particles.pos[i_p, i_b][i] = poss[i_b_, i_p_, i]
        system.particles.vel[i_p, i_b].fill(0.0)


@qd.kernel
def _kernel_get_particles_pos(
    system: qd.template(),
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
):
    for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
        i_p = i_p_ + particle_start
        i_b = envs_idx[i_b_]
        for i in qd.static(range(3)):
            poss[i_b_, i_p_, i] = system.particles.pos[i_p, i_b][i]


@qd.kernel
def _kernel_set_particles_vel(
    system: qd.template(),
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    vels: qd.types.ndarray(),
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        for i in qd.static(range(3)):
            system.particles.vel[i_p, i_b][i] = vels[i_b_, i_p_, i]


@qd.kernel
def _kernel_get_particles_vel(
    system: qd.template(),
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    vels: qd.types.ndarray(),
):
    for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
        i_p = i_p_ + particle_start
        i_b = envs_idx[i_b_]
        for i in qd.static(range(3)):
            vels[i_b_, i_p_, i] = system.particles.vel[i_p, i_b][i]


@qd.kernel
def _kernel_set_particles_active(
    system: qd.template(),
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    actives: qd.types.ndarray(),  # shape [B, n_particles]
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        system.particles_ng.active[i_p, i_b] = qd.cast(actives[i_b_, i_p_], gs.qd_bool)


@qd.kernel
def _kernel_get_particles_active(
    system: qd.template(),
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    actives: qd.types.ndarray(),  # shape [B, n_particles]
):
    for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
        i_p = i_p_ + particle_start
        i_b = envs_idx[i_b_]
        actives[i_b_, i_p_] = system.particles_ng.active[i_p, i_b]


@qd.kernel
def _kernel_fix_particles(system: qd.template(), particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray()):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        system.particles.free[i_p, i_b] = False


@qd.kernel
def _kernel_release_particle(system: qd.template(), particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray()):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        system.particles.free[i_p, i_b] = True


@qd.kernel
def _kernel_get_mass(
    system: qd.template(),
    particle_start: qd.i32,
    n_particles: qd.i32,
    mass: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
):
    total_mass = gs.qd_float(0.0)
    for i_p_ in range(n_particles):
        i_p = i_p_ + particle_start
        total_mass += system.particles_info.mass[i_p]
    for i_b_ in range(envs_idx.shape[0]):
        mass[i_b_] = total_mass
