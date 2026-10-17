"""Free kernels and functions of FEM simulation over FEMSolverSystem-owned data."""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.geom import qd_transform_by_quat, qd_transform_quat_by_quat


@qd.kernel
def compute_surface_vertices(system: qd.template()):
    for i_v in range(system.n_vertices):
        system.vertices_on_surface[i_v] = False

    for i_s in range(system.n_surfaces):
        tri2v = system.surface.tri2v[i_s]
        for i in qd.static(range(3)):
            system.vertices_on_surface[tri2v[i]] = True


@qd.kernel
def compute_surface_elements(system: qd.template()):
    for i_e in range(system.n_elements):
        i_v = system.elements_i.el2v[i_e]
        system.elements_on_surface[i_e] = (
            system.vertices_on_surface[i_v[0]]
            or system.vertices_on_surface[i_v[1]]
            or system.vertices_on_surface[i_v[2]]
            or system.vertices_on_surface[i_v[3]]
        )


@qd.kernel
def init_pos_and_vel(system: qd.template(), f: qd.i32):
    for i_v, i_b in qd.ndrange(system.n_vertices, system._B):
        system.elements_v.pos[f + 1, i_v, i_b] = system.elements_v.pos[f, i_v, i_b]
        system.elements_v.vel[f + 1, i_v, i_b] = system.elements_v.vel[f, i_v, i_b]


@qd.kernel
def compute_vel(system: qd.template(), f: qd.i32):
    for i_e, i_b in qd.ndrange(system.n_elements, system._B):
        i_v0, i_v1, i_v2, i_v3 = system.elements_i.el2v[i_e]
        pos_v0 = system.elements_v.pos[f, i_v0, i_b]
        pos_v1 = system.elements_v.pos[f, i_v1, i_b]
        pos_v2 = system.elements_v.pos[f, i_v2, i_b]
        pos_v3 = system.elements_v.pos[f, i_v3, i_b]
        D = qd.Matrix.cols([pos_v0 - pos_v3, pos_v1 - pos_v3, pos_v2 - pos_v3])

        V_scaled = system.elements_i.V_scaled[i_e]
        B = system.elements_i.B[i_e]
        F = D @ B
        J = F.determinant()

        stress = qd.Matrix.zero(gs.qd_float, 3, 3)
        for mat_idx in qd.static(system.material_config.indices):
            if system.elements_i.mat_idx[i_e] == mat_idx:
                stress = system.material_config.update_stress[mat_idx](
                    mu=system.elements_i.mu[i_e],
                    lam=system.elements_i.lam[i_e],
                    J=J,
                    F=F,
                    actu=system.elements_el.actu[f, i_e, i_b],
                    m_dir=system.elements_i.muscle_direction[i_e],
                )

        verts = system.elements_i.el2v[i_e]
        mass_scaled = system.elements_i.mass_scaled[i_e]
        H_scaled = -V_scaled * stress @ B.transpose()
        for k in qd.static(range(3)):
            force_scaled = qd.Vector([H_scaled[j, k] for j in range(3)])

            # store so forces can be read out
            system.elements_v_energy.force[i_b, verts[k]] = force_scaled

            dv = system.substep_dt * force_scaled / mass_scaled
            system.elements_v.vel[f + 1, verts[k], i_b] += dv
            system.elements_v.vel[f + 1, verts[3], i_b] -= dv


@qd.kernel
def apply_uniform_force(system: qd.template(), f: qd.i32):
    for i_v, i_b in qd.ndrange(system.n_vertices, system._B):
        # NOTE: damping should only be applied to velocity from internal force and thus come first here
        #       given the immediate previous function call is compute_internal_vel --> however, shouldn't
        #       be done at dv only and need to wait for all elements updated (cannot be in the compute_internal_vel kernel)
        #       however, this inevitably damp the gravity.
        system.elements_v.vel[f + 1, i_v, i_b] *= qd.exp(-system.substep_dt * system.damping)
        # Add gravity (avoiding damping on gravity)
        system.elements_v.vel[f + 1, i_v, i_b] += system.substep_dt * system._gravity[i_b]


@qd.kernel
def compute_pos(system: qd.template(), f: qd.i32):
    for i_v, i_b in qd.ndrange(system.n_vertices, system._B):
        system.elements_v.pos[f + 1, i_v, i_b] = (
            system.substep_dt * system.elements_v.vel[f + 1, i_v, i_b] + system.elements_v.pos[f, i_v, i_b]
        )


@qd.kernel
def precompute_material_data(system: qd.template(), f: qd.i32):
    for i_b, i_e in qd.ndrange(system._B, system.n_elements):
        J, F = _compute_ele_J_F(system, f, i_e, i_b)  # use last time step's pos to compute
        for mat_idx in qd.static(system.material_config.indices):
            if system.elements_i.mat_idx[i_e] == mat_idx:
                system.material_config.materials[mat_idx].pre_compute(J=J, F=F, i_e=i_e, i_b=i_b)


@qd.kernel
def init_pos_and_inertia(system: qd.template(), f: qd.i32):
    dt2 = system.substep_dt**2
    for i_v, i_b in qd.ndrange(system.n_vertices, system._B):
        if qd.static(system._enable_vertex_constraints):
            if system.vertex_constraints.is_constrained[i_v, i_b]:
                system.elements_v.pos[f + 1, i_v, i_b] = system.vertex_constraints.target_pos[i_v, i_b]
                system.elements_v_energy.inertia[i_b, i_v] = system.vertex_constraints.target_pos[i_v, i_b]
            else:
                system.elements_v_energy.inertia[i_b, i_v] = (
                    system.elements_v.pos[f, i_v, i_b]
                    + system.elements_v.vel[f, i_v, i_b] * system.substep_dt
                    + system._gravity[i_b] * dt2
                )
                system.elements_v.pos[f + 1, i_v, i_b] = system.elements_v.pos[f, i_v, i_b]
        else:
            system.elements_v_energy.inertia[i_b, i_v] = (
                system.elements_v.pos[f, i_v, i_b]
                + system.elements_v.vel[f, i_v, i_b] * system.substep_dt
                + system._gravity[i_b] * dt2
            )
            system.elements_v.pos[f + 1, i_v, i_b] = system.elements_v.pos[f, i_v, i_b]


@qd.func
def _compute_ele_J_F(system: qd.template(), f: qd.i32, i_e: qd.i32, i_b: qd.i32):
    """
    Compute the determinant (J) and deformation gradient (F) for an element.
    """
    i_v0, i_v1, i_v2, i_v3 = system.elements_i.el2v[i_e]
    pos_v0 = system.elements_v.pos[f, i_v0, i_b]
    pos_v1 = system.elements_v.pos[f, i_v1, i_b]
    pos_v2 = system.elements_v.pos[f, i_v2, i_b]
    pos_v3 = system.elements_v.pos[f, i_v3, i_b]
    D = qd.Matrix.cols([pos_v0 - pos_v3, pos_v1 - pos_v3, pos_v2 - pos_v3])

    B = system.elements_i.B[i_e]
    F = D @ B
    J = F.determinant()

    return J, F


@qd.kernel
def compute_ele_hessian_gradient(system: qd.template(), f: qd.i32):
    # The material callbacks take the Hessian store as a template so they work with both ndarray and field storage.
    hessian_field = qd.static(system.elements_el_hessian)
    for i_b, i_e in qd.ndrange(system._B, system.n_elements):
        if not system.batch_active[i_b]:
            continue

        J, F = _compute_ele_J_F(system, f + 1, i_e, i_b)

        for mat_idx in qd.static(system.material_config.indices):
            if system.elements_i.mat_idx[i_e] == mat_idx:
                if system.material_config.materials[mat_idx]._hessian_ready:
                    (
                        system.elements_el_energy.energy[i_b, i_e],
                        system.elements_el_energy.gradient[i_b, i_e],
                    ) = system.material_config.materials[mat_idx].compute_energy_gradient(
                        mu=system.elements_i.mu[i_e],
                        lam=system.elements_i.lam[i_e],
                        J=J,
                        F=F,
                        actu=system.elements_el.actu[f, i_e, i_b],
                        m_dir=system.elements_i.muscle_direction[i_e],
                        i_e=i_e,
                        i_b=i_b,
                    )
                else:
                    (
                        system.elements_el_energy.energy[i_b, i_e],
                        system.elements_el_energy.gradient[i_b, i_e],
                    ) = system.material_config.materials[mat_idx].compute_energy_gradient_hessian(
                        mu=system.elements_i.mu[i_e],
                        lam=system.elements_i.lam[i_e],
                        J=J,
                        F=F,
                        actu=system.elements_el.actu[f, i_e, i_b],
                        m_dir=system.elements_i.muscle_direction[i_e],
                        i_e=i_e,
                        i_b=i_b,
                        hessian_field=hessian_field,
                    )


@qd.func
def _func_compute_element_mapping_matrix(system: qd.template(), i_vs, B, i_b):
    """
    Compute the element mapping matrix S for an element.
    """
    S = qd.Matrix.zero(gs.qd_float, 4, 3)
    S[:3, :] = B
    S[3, :] = -B[0, :] - B[1, :] - B[2, :]

    if qd.static(system._enable_vertex_constraints):
        for i in qd.static(range(4)):
            if system.vertex_constraints.is_constrained[i_vs[i], i_b]:
                S[i, :] = qd.Vector.zero(gs.qd_float, 3)
    return S


@qd.func
def _func_compute_ele_energy(system: qd.template(), f: qd.i32):
    """
    Compute the energy for each element in the batch. Should only be used in linesearch.
    """
    for i_b, i_e in qd.ndrange(system._B, system.n_elements):
        if not system.batch_linesearch_active[i_b]:
            continue

        J, F = _compute_ele_J_F(system, f + 1, i_e, i_b)

        for mat_idx in qd.static(system.material_config.indices):
            if system.elements_i.mat_idx[i_e] == mat_idx:
                system.elements_el_energy.energy[i_b, i_e] = system.material_config.materials[mat_idx].compute_energy(
                    mu=system.elements_i.mu[i_e],
                    lam=system.elements_i.lam[i_e],
                    J=J,
                    F=F,
                    actu=system.elements_el.actu[f, i_e, i_b],
                    m_dir=system.elements_i.muscle_direction[i_e],
                    i_e=i_e,
                    i_b=i_b,
                )

        # add linearized damping energy
        if system._damping_beta > gs.EPS:
            damping_beta_over_dt = system._damping_beta / system._substep_dt
            i_vs = system.elements_i.el2v[i_e]
            B = system.elements_i.B[i_e]
            S = _func_compute_element_mapping_matrix(system, i_vs, B, i_b)

            x_diff = qd.Vector.zero(gs.qd_float, 12)
            for i in qd.static(range(4)):
                x_diff[i * 3 : i * 3 + 3] = (
                    system.elements_v.pos[f + 1, i_vs[i], i_b] - system.elements_v.pos[f, i_vs[i], i_b]
                )
            St_x_diff = qd.Vector.zero(gs.qd_float, 9)
            for i, j in qd.static(qd.ndrange(3, 4)):
                St_x_diff[i * 3 : i * 3 + 3] += S[j, i] * x_diff[j * 3 : j * 3 + 3]

            H_St_x_diff = qd.Vector.zero(gs.qd_float, 9)
            for i, j in qd.static(qd.ndrange(3, 3)):
                H_St_x_diff[i * 3 : i * 3 + 3] += (
                    system.elements_el_hessian[i_b, i, j, i_e] @ St_x_diff[j * 3 : j * 3 + 3]
                )

            system.elements_el_energy.energy[i_b, i_e] += 0.5 * damping_beta_over_dt * St_x_diff.dot(H_St_x_diff)


@qd.kernel
def accumulate_vertex_force_preconditioner(system: qd.template(), f: qd.i32):
    damping_alpha_dt = system._damping_alpha * system._substep_dt
    damping_alpha_factor = damping_alpha_dt + 1.0
    damping_beta_over_dt = system._damping_beta / system._substep_dt
    damping_beta_factor = damping_beta_over_dt + 1.0
    # inertia
    for i_b, i_v in qd.ndrange(system._B, system.n_vertices):
        if not system.batch_active[i_b]:
            continue
        system.elements_v_energy.force[i_b, i_v] = -system.elements_v_info.mass_over_dt2[i_v] * (
            (system.elements_v.pos[f + 1, i_v, i_b] - system.elements_v_energy.inertia[i_b, i_v])
            + (system.elements_v.pos[f + 1, i_v, i_b] - system.elements_v.pos[f, i_v, i_b]) * damping_alpha_dt
        )
        system.pcg_state_v.diag3x3[i_b, i_v] = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i in qd.static(range(3)):
            system.pcg_state_v.diag3x3[i_b, i_v][i, i] = (
                system.elements_v_info.mass_over_dt2[i_v] * damping_alpha_factor
            )

    # elastic
    for i_b, i_e in qd.ndrange(system._B, system.n_elements):
        if not system.batch_active[i_b]:
            continue
        V = system.elements_i.V[i_e]
        B = system.elements_i.B[i_e]
        gradient = system.elements_el_energy.gradient[i_b, i_e]
        i_vs = system.elements_i.el2v[i_e]
        S = _func_compute_element_mapping_matrix(system, i_vs, B, i_b)
        force = -V * gradient @ S.transpose()

        # atomic
        for i in qd.static(range(4)):
            system.elements_v_energy.force[i_b, i_vs[i]] += force[:, i]

        if system._damping_beta > gs.EPS:
            x_diff = qd.Vector.zero(gs.qd_float, 12)
            for i in qd.static(range(4)):
                x_diff[i * 3 : i * 3 + 3] = (
                    system.elements_v.pos[f + 1, i_vs[i], i_b] - system.elements_v.pos[f, i_vs[i], i_b]
                )
            St_x_diff = qd.Vector.zero(gs.qd_float, 9)
            for i, j in qd.static(qd.ndrange(3, 4)):
                St_x_diff[i * 3 : i * 3 + 3] += S[j, i] * x_diff[j * 3 : j * 3 + 3]

            H_St_x_diff = qd.Vector.zero(gs.qd_float, 9)
            for i, j in qd.static(qd.ndrange(3, 3)):
                H_St_x_diff[i * 3 : i * 3 + 3] += (
                    system.elements_el_hessian[i_b, i, j, i_e] @ St_x_diff[j * 3 : j * 3 + 3]
                )
            S_H_St_x_diff = qd.Vector.zero(gs.qd_float, 12)
            for i, j in qd.static(qd.ndrange(4, 3)):
                S_H_St_x_diff[i * 3 : i * 3 + 3] += S[i, j] * H_St_x_diff[j * 3 : j * 3 + 3]
            for i in qd.static(range(4)):
                system.elements_v_energy.force[i_b, i_vs[i]] += (
                    -damping_beta_over_dt * V * S_H_St_x_diff[i * 3 : i * 3 + 3]
                )

        # diagonal 3-by-3 block of hessian
        for k, i, j in qd.ndrange(4, 3, 3):
            system.pcg_state_v.diag3x3[i_b, i_vs[k]] += (
                V * damping_beta_factor * S[k, i] * S[k, j] * system.elements_el_hessian[i_b, i, j, i_e]
            )

    # inverse
    for i_b, i_v in qd.ndrange(system._B, system.n_vertices):
        if not system.batch_active[i_b]:
            continue
        # Use 3-by-3 diagonal block inverse for preconditioner
        system.pcg_state_v.prec[i_b, i_v] = system.pcg_state_v.diag3x3[i_b, i_v].inverse()


@qd.func
def compute_Ap(system: qd.template()):
    damping_alpha_dt = system._damping_alpha * system._substep_dt
    damping_alpha_factor = damping_alpha_dt + 1.0
    damping_beta_over_dt = system._damping_beta / system._substep_dt
    damping_beta_factor = damping_beta_over_dt + 1.0
    for i_b, i_v in qd.ndrange(system._B, system.n_vertices):
        if not system.batch_pcg_active[i_b]:
            continue
        system.pcg_state_v.Ap[i_b, i_v] = (
            system.elements_v_info.mass_over_dt2[i_v] * damping_alpha_factor * system.pcg_state_v.p[i_b, i_v]
        )

    for i_b, i_e in qd.ndrange(system._B, system.n_elements):
        if not system.batch_pcg_active[i_b]:
            continue
        V = system.elements_i.V[i_e]
        B = system.elements_i.B[i_e]
        i_vs = system.elements_i.el2v[i_e]
        S = _func_compute_element_mapping_matrix(system, i_vs, B, i_b)

        p9 = qd.Vector([0.0] * 9, dt=gs.qd_float)

        for i, j in qd.static(qd.ndrange(3, 4)):
            p9[i * 3 : i * 3 + 3] = p9[i * 3 : i * 3 + 3] + S[j, i] * system.pcg_state_v.p[i_b, i_vs[j]]

        new_p9 = qd.Vector([0.0] * 9, dt=gs.qd_float)

        for i, j in qd.static(qd.ndrange(3, 3)):
            new_p9[i * 3 : i * 3 + 3] = (
                new_p9[i * 3 : i * 3 + 3] + system.elements_el_hessian[i_b, i, j, i_e] @ p9[j * 3 : j * 3 + 3]
            )

        # atomic
        for i in qd.static(range(4)):
            system.pcg_state_v.Ap[i_b, i_vs[i]] += (
                (S[i, 0] * new_p9[0:3] + S[i, 1] * new_p9[3:6] + S[i, 2] * new_p9[6:9]) * V * damping_beta_factor
            )


@qd.kernel
def init_pcg_solve(system: qd.template()):
    for i_b in range(system._B):
        system.batch_pcg_active[i_b] = system.batch_active[i_b]
        if not system.batch_pcg_active[i_b]:
            continue
        system.pcg_state.rTr[i_b] = 0.0
        system.pcg_state.rTz[i_b] = 0.0
    for i_b, i_v in qd.ndrange(system._B, system.n_vertices):
        if not system.batch_pcg_active[i_b]:
            continue
        system.pcg_state_v.x[i_b, i_v] = 0
        system.pcg_state_v.r[i_b, i_v] = system.elements_v_energy.force[i_b, i_v]
        system.pcg_state_v.z[i_b, i_v] = system.pcg_state_v.prec[i_b, i_v] @ system.pcg_state_v.r[i_b, i_v]
        system.pcg_state_v.p[i_b, i_v] = system.pcg_state_v.z[i_b, i_v]
        qd.atomic_add(system.pcg_state.rTr[i_b], system.pcg_state_v.r[i_b, i_v].dot(system.pcg_state_v.r[i_b, i_v]))
        qd.atomic_add(system.pcg_state.rTz[i_b], system.pcg_state_v.r[i_b, i_v].dot(system.pcg_state_v.z[i_b, i_v]))
    for i_b in range(system._B):
        if not system.batch_pcg_active[i_b]:
            continue
        system.batch_pcg_active[i_b] = system.pcg_state.rTr[i_b] > system._pcg_threshold


@qd.kernel
def one_pcg_iter(system: qd.template()):
    compute_Ap(system)

    # compute pTAp
    for i_b in range(system._B):
        if not system.batch_pcg_active[i_b]:
            continue
        system.pcg_state.pTAp[i_b] = 0.0
    for i_b, i_v in qd.ndrange(system._B, system.n_vertices):
        if not system.batch_pcg_active[i_b]:
            continue
        qd.atomic_add(system.pcg_state.pTAp[i_b], system.pcg_state_v.p[i_b, i_v].dot(system.pcg_state_v.Ap[i_b, i_v]))

    # compute alpha and update x, r, z, rTr, rTz
    for i_b in range(system._B):
        if not system.batch_pcg_active[i_b]:
            continue
        system.pcg_state.alpha[i_b] = system.pcg_state.rTz[i_b] / system.pcg_state.pTAp[i_b]
        system.pcg_state.rTr_new[i_b] = 0.0
        system.pcg_state.rTz_new[i_b] = 0.0
    for i_b, i_v in qd.ndrange(system._B, system.n_vertices):
        if not system.batch_pcg_active[i_b]:
            continue
        system.pcg_state_v.x[i_b, i_v] += system.pcg_state.alpha[i_b] * system.pcg_state_v.p[i_b, i_v]
        system.pcg_state_v.r[i_b, i_v] -= system.pcg_state.alpha[i_b] * system.pcg_state_v.Ap[i_b, i_v]
        system.pcg_state_v.z[i_b, i_v] = system.pcg_state_v.prec[i_b, i_v] @ system.pcg_state_v.r[i_b, i_v]
        qd.atomic_add(system.pcg_state.rTr_new[i_b], system.pcg_state_v.r[i_b, i_v].dot(system.pcg_state_v.r[i_b, i_v]))
        qd.atomic_add(system.pcg_state.rTz_new[i_b], system.pcg_state_v.r[i_b, i_v].dot(system.pcg_state_v.z[i_b, i_v]))

    # check convergence
    for i_b in range(system._B):
        if not system.batch_pcg_active[i_b]:
            continue
        system.batch_pcg_active[i_b] = system.pcg_state.rTr_new[i_b] > system._pcg_threshold

    # update beta, rTr, rTz
    for i_b in range(system._B):
        if not system.batch_pcg_active[i_b]:
            continue
        system.pcg_state.beta[i_b] = system.pcg_state.rTz_new[i_b] / system.pcg_state.rTz[i_b]
        system.pcg_state.rTr[i_b] = system.pcg_state.rTr_new[i_b]
        system.pcg_state.rTz[i_b] = system.pcg_state.rTz_new[i_b]

    # update p
    for i_b, i_v in qd.ndrange(system._B, system.n_vertices):
        if not system.batch_pcg_active[i_b]:
            continue
        system.pcg_state_v.p[i_b, i_v] = (
            system.pcg_state_v.z[i_b, i_v] + system.pcg_state.beta[i_b] * system.pcg_state_v.p[i_b, i_v]
        )


@qd.kernel
def init_linesearch(system: qd.template(), f: qd.i32):
    for i_b in range(system._B):
        system.batch_linesearch_active[i_b] = system.batch_active[i_b]
        if not system.batch_linesearch_active[i_b]:
            continue
        system.linesearch_state.prev_energy[i_b] = 0.0
        system.linesearch_state.step_size[i_b] = 1.0
        system.linesearch_state.m[i_b] = 0.0

    # Inertia, x_prev, m
    for i_b, i_v in qd.ndrange(system._B, system.n_vertices):
        if not system.batch_linesearch_active[i_b]:
            continue
        diff = system.elements_v.pos[f + 1, i_v, i_b] - system.elements_v_energy.inertia[i_b, i_v]
        system.linesearch_state.prev_energy[i_b] += 0.5 * system.elements_v_info.mass_over_dt2[i_v] * diff.dot(diff)
        system.linesearch_state_v.x_prev[i_b, i_v] = system.elements_v.pos[f + 1, i_v, i_b]
        system.linesearch_state.m[i_b] -= system.pcg_state_v.x[i_b, i_v].dot(system.elements_v_energy.force[i_b, i_v])
    # Elastic
    for i_b, i_e in qd.ndrange(system._B, system.n_elements):
        if not system.batch_linesearch_active[i_b]:
            continue
        system.linesearch_state.prev_energy[i_b] += (
            system.elements_el_energy.energy[i_b, i_e] * system.elements_i.V[i_e]
        )


@qd.kernel
def one_linesearch_iter(system: qd.template(), f: qd.i32):
    for i_b in range(system._B):
        if not system.batch_linesearch_active[i_b]:
            continue
        system.linesearch_state.energy[i_b] = 0.0

    # update pos and compute Inertia energy
    for i_b, i_v in qd.ndrange(system._B, system.n_vertices):
        if not system.batch_linesearch_active[i_b]:
            continue
        system.elements_v.pos[f + 1, i_v, i_b] = (
            system.linesearch_state_v.x_prev[i_b, i_v]
            + system.linesearch_state.step_size[i_b] * system.pcg_state_v.x[i_b, i_v]
        )
        diff = system.elements_v.pos[f + 1, i_v, i_b] - system.elements_v_energy.inertia[i_b, i_v]
        system.linesearch_state.energy[i_b] += 0.5 * system.elements_v_info.mass_over_dt2[i_v] * diff.dot(diff)
        # damping
        if system._damping_alpha > 0.0:
            damping_alpha_dt = system._damping_alpha * system._substep_dt
            diff = system.elements_v.pos[f + 1, i_v, i_b] - system.elements_v.pos[f, i_v, i_b]
            system.linesearch_state.energy[i_b] += (
                0.5 * system.elements_v_info.mass_over_dt2[i_v] * diff.dot(diff) * damping_alpha_dt
            )

    # compute elastic energy
    _func_compute_ele_energy(system, f)
    for i_b, i_e in qd.ndrange(system._B, system.n_elements):
        if not system.batch_linesearch_active[i_b]:
            continue
        system.linesearch_state.energy[i_b] += system.elements_el_energy.energy[i_b, i_e] * system.elements_i.V[i_e]

    # check condition
    for i_b in range(system._B):
        if not system.batch_linesearch_active[i_b]:
            continue
        system.batch_linesearch_active[i_b] = (
            system.linesearch_state.energy[i_b]
            > system.linesearch_state.prev_energy[i_b]
            + system._linesearch_c * system.linesearch_state.step_size[i_b] * system.linesearch_state.m[i_b]
        )
        if not system.batch_linesearch_active[i_b]:
            continue
        system.linesearch_state.step_size[i_b] *= system._linesearch_tau


@qd.kernel
def skip_linesearch(system: qd.template(), f: qd.i32):
    # Inertia, x_prev, m
    for i_b, i_v in qd.ndrange(system._B, system.n_vertices):
        if not system.batch_active[i_b]:
            continue
        system.elements_v.pos[f + 1, i_v, i_b] = system.elements_v.pos[f + 1, i_v, i_b] + system.pcg_state_v.x[i_b, i_v]


@qd.kernel
def setup_pos_vel(system: qd.template(), f: qd.i32):
    for i_v, i_b in qd.ndrange(system.n_vertices, system._B):
        # set pos and vel
        system.elements_v.vel[f + 1, i_v, i_b] = (
            system.elements_v.pos[f + 1, i_v, i_b] - system.elements_v.pos[f, i_v, i_b]
        ) / system.substep_dt


@qd.kernel
def copy_frame(system: qd.template(), source: qd.i32, target: qd.i32):
    # Copy pos/vel for all vertices and all batch indices
    for i_v, i_b in qd.ndrange(system.n_vertices_max, system._B):
        system.elements_v.pos[target, i_v, i_b] = system.elements_v.pos[source, i_v, i_b]
        system.elements_v.vel[target, i_v, i_b] = system.elements_v.vel[source, i_v, i_b]

    # Copy 'active' for all elements and all batch indices
    for i_e, i_b in qd.ndrange(system.n_elements_max, system._B):
        system.elements_el_ng.active[target, i_e, i_b] = system.elements_el_ng.active[source, i_e, i_b]


@qd.kernel
def copy_grad(system: qd.template(), source: qd.i32, target: qd.i32):
    # Copy gradients for vertices
    for i_v, i_b in qd.ndrange(system.n_vertices_max, system._B):
        system.elements_v.pos.grad[target, i_v, i_b] = system.elements_v.pos.grad[source, i_v, i_b]
        system.elements_v.vel.grad[target, i_v, i_b] = system.elements_v.vel.grad[source, i_v, i_b]

    # Copy 'active' for elements
    for i_e, i_b in qd.ndrange(system.n_elements_max, system._B):
        system.elements_el_ng.active[target, i_e, i_b] = system.elements_el_ng.active[source, i_e, i_b]


@qd.kernel
def copy_grad_ndarray(
    source: qd.i32,
    target: qd.i32,
    n_vertices: qd.template(),
    n_elements: qd.template(),
    n_envs: qd.template(),
    pos_grad: qd.types.ndarray(),
    vel_grad: qd.types.ndarray(),
    active: qd.types.ndarray(),
):
    # Quadrants 1.3.3 does not copy nested ndarray adjoints reached through a template root; pass members explicitly.
    for i_v, i_b in qd.ndrange(n_vertices, n_envs):
        pos_grad[target, i_v, i_b] = pos_grad[source, i_v, i_b]
        vel_grad[target, i_v, i_b] = vel_grad[source, i_v, i_b]
    for i_e, i_b in qd.ndrange(n_elements, n_envs):
        active[target, i_e, i_b] = active[source, i_e, i_b]


@qd.kernel
def reset_grad_till_frame(system: qd.template(), f: qd.i32):
    # Zero out v.grad in frame 0..(f-1) for all vertices, all batch indices
    for frame_i, vert_i, i_b in qd.ndrange(f, system.n_vertices_max, system._B):
        system.elements_v.pos.grad[frame_i, vert_i, i_b] = 0
        system.elements_v.vel.grad[frame_i, vert_i, i_b] = 0

    # Zero out elements_el.grad in frame 0..(f-1) for all elements, all batch indices
    for frame_i, elem_i, i_b in qd.ndrange(f, system.n_elements_max, system._B):
        system.elements_el.actu.grad[frame_i, elem_i, i_b] = 0


@qd.kernel
def _kernel_add_elements(
    system: qd.template(),
    f: qd.i32,
    mat_idx: qd.i32,
    mat_mu: qd.f32,
    mat_lam: qd.f32,
    mat_rho: qd.f32,
    mat_friction_mu: qd.f32,
    v_start: qd.i32,
    el_start: qd.i32,
    s_start: qd.i32,
    verts: qd.types.ndarray(),
    elems: qd.types.ndarray(),
    tri2v: qd.types.ndarray(),
    tri2el: qd.types.ndarray(),
):
    n_verts_local = verts.shape[0]
    for i_v, i_b in qd.ndrange(n_verts_local, system._B):
        i_global = i_v + v_start
        for j in qd.static(range(3)):
            system.elements_v.pos[f, i_global, i_b][j] = verts[i_v, j]
        system.elements_v.vel[f, i_global, i_b] = qd.Vector.zero(gs.qd_float, 3)

    for i_v in range(n_verts_local):
        i_global = i_v + v_start
        system.elements_v_info.mass[i_global] = 0.0
        system.elements_v_info.mass_over_dt2[i_global] = 0.0
        system.elements_v_info.friction_mu[i_global] = mat_friction_mu

    dt2_inv = 1.0 / (system.substep_dt**2)
    n_elems_local = elems.shape[0]
    for i_e in range(n_elems_local):
        i_global = i_e + el_start

        a = system.elements_v.pos[f, elems[i_e, 0] + v_start, 0]
        b = system.elements_v.pos[f, elems[i_e, 1] + v_start, 0]
        c = system.elements_v.pos[f, elems[i_e, 2] + v_start, 0]
        d = system.elements_v.pos[f, elems[i_e, 3] + v_start, 0]
        B_inv = qd.Matrix.cols([a - d, b - d, c - d])
        system.elements_i.B[i_global] = B_inv.inverse()
        det = B_inv.determinant()
        # Determinant should be consistently smaller than 0
        if det >= 0.0:
            system.tet_wrong_order[None] = True
        V = qd.abs(det) / 6.0
        system.elements_i.V[i_global] = V
        V_scaled = V * system._vol_scale
        system.elements_i.V_scaled[i_global] = V_scaled

        for j in qd.static(range(4)):
            system.elements_i.el2v[i_global][j] = elems[i_e, j] + v_start
        system.elements_i.mat_idx[i_global] = mat_idx
        system.elements_i.mu[i_global] = mat_mu
        system.elements_i.lam[i_global] = mat_lam
        system.elements_i.friction_mu[i_global] = mat_friction_mu
        system.elements_i.mass_scaled[i_global] = mat_rho * V_scaled
        for j in qd.static(range(4)):
            mass = 0.25 * mat_rho * V
            system.elements_v_info.mass[system.elements_i.el2v[i_global][j]] += mass
            system.elements_v_info.mass_over_dt2[system.elements_i.el2v[i_global][j]] += mass * dt2_inv
        system.elements_i.muscle_group[i_global] = 0
        system.elements_i.muscle_direction[i_global] = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)

    for i_v in range(n_verts_local):
        i_global = i_v + v_start
        system.elements_v_info.mass_inv[i_global] = 1.0 / system.elements_v_info.mass[i_global]

    for i_e, i_b in qd.ndrange(n_elems_local, system._B):
        i_global = i_e + el_start
        system.elements_el.actu[f, i_global, i_b] = 0.0
        system.elements_el_ng.active[f, i_global, i_b] = True

    for i_s in range(tri2v.shape[0]):
        i_global = i_s + s_start
        for j in qd.static(range(3)):
            system.surface.tri2v[i_global][j] = tri2v[i_s, j] + v_start
        system.surface.tri2el[i_global] = tri2el[i_s] + el_start
        system.surface.active[i_global] = True


@qd.kernel
def _kernel_add_cloth(
    system: qd.template(),
    f: qd.i32,
    v_start: qd.i32,
    s_start: qd.i32,
    verts: qd.types.ndarray(),
    tri2v: qd.types.ndarray(),
):
    """
    Add cloth vertices and surface triangles to the solver, for position tracking and coupling only.

    Cloth elements and mass are owned by the IPC coupler, so the vertex info holds placeholder values and each
    surface triangle references itself as element.
    """
    n_verts_local = verts.shape[0]
    for i_v, i_b in qd.ndrange(n_verts_local, system._B):
        i_global = i_v + v_start
        for j in qd.static(range(3)):
            system.elements_v.pos[f, i_global, i_b][j] = verts[i_v, j]
        system.elements_v.vel[f, i_global, i_b] = qd.Vector.zero(gs.qd_float, 3)

    for i_v in range(n_verts_local):
        i_global = i_v + v_start
        system.elements_v_info.mass[i_global] = 1.0
        system.elements_v_info.mass_over_dt2[i_global] = 0.0
        system.elements_v_info.friction_mu[i_global] = 0.0

    for i_s in range(tri2v.shape[0]):
        i_global = i_s + s_start
        for j in qd.static(range(3)):
            system.surface.tri2v[i_global][j] = tri2v[i_s, j] + v_start
        system.surface.tri2el[i_global] = i_global
        system.surface.active[i_global] = True


@qd.kernel
def _kernel_set_elements_pos(
    system: qd.template(),
    f: qd.i32,
    element_v_start: qd.i32,
    n_vertices: qd.i32,
    pos: qd.types.ndarray(),
):
    for i_v, i_b in qd.ndrange(n_vertices, system._B):
        i_global = i_v + element_v_start
        for k in qd.static(range(3)):
            system.elements_v.pos[f, i_global, i_b][k] = pos[i_b, i_v, k]


@qd.kernel
def _kernel_set_elements_pos_grad(
    system: qd.template(),
    f: qd.i32,
    element_v_start: qd.i32,
    n_vertices: qd.i32,
    pos_grad: qd.types.ndarray(),
):
    for i_v, i_b in qd.ndrange(n_vertices, system._B):
        i_global = i_v + element_v_start
        for k in qd.static(range(3)):
            system.elements_v.pos.grad[f, i_global, i_b][k] = pos_grad[i_b, i_v, k]


@qd.kernel
def _kernel_set_elements_vel(
    system: qd.template(),
    f: qd.i32,
    element_v_start: qd.i32,
    n_vertices: qd.i32,
    vel: qd.types.ndarray(),  # shape [B, n_vertices, 3]
):
    for i_v, i_b in qd.ndrange(n_vertices, system._B):
        i_global = i_v + element_v_start
        for k in qd.static(range(3)):
            system.elements_v.vel[f, i_global, i_b][k] = vel[i_b, i_v, k]


@qd.kernel
def _kernel_set_elements_vel_grad(
    system: qd.template(),
    f: qd.i32,
    element_v_start: qd.i32,
    n_vertices: qd.i32,
    vel_grad: qd.types.ndarray(),  # shape [B, n_vertices, 3]
):
    for i_v, i_b in qd.ndrange(n_vertices, system._B):
        i_global = i_v + element_v_start
        for k in qd.static(range(3)):
            system.elements_v.vel.grad[f, i_global, i_b][k] = vel_grad[i_b, i_v, k]


@qd.kernel
def _kernel_set_elements_actu(
    system: qd.template(),
    f: qd.i32,
    element_el_start: qd.i32,
    n_elements: qd.i32,
    n_groups: qd.i32,
    actu: qd.types.ndarray(),  # shape [B, n_elements, n_groups]
):
    for i_e, j_g, i_b in qd.ndrange(n_elements, n_groups, system._B):
        i_global = i_e + element_el_start
        if system.elements_i.muscle_group[i_global] == j_g:
            system.elements_el.actu[f, i_global, i_b] = actu[i_b, j_g]


@qd.kernel
def _kernel_set_elements_actu_grad(
    system: qd.template(),
    f: qd.i32,
    element_el_start: qd.i32,
    n_elements: qd.i32,
    actu_grad: qd.types.ndarray(),  # shape [B, n_elements]
):
    for i_e, i_b in qd.ndrange(n_elements, system._B):
        i_global = i_e + element_el_start
        system.elements_el.actu.grad[f, i_global, i_b] = actu_grad[i_b, i_e]


@qd.kernel
def _kernel_set_active(
    system: qd.template(),
    f: qd.i32,
    element_el_start: qd.i32,
    n_elements: qd.i32,
    active: qd.types.ndarray(),  # shape [B, n_elements]
):
    for i_e, i_b in qd.ndrange(n_elements, system._B):
        i_global = i_e + element_el_start
        system.elements_el_ng.active[f, i_global, i_b] = active[i_b, i_e]


@qd.kernel
def _kernel_set_muscle_group(
    system: qd.template(),
    element_el_start: qd.i32,
    n_elements: qd.i32,
    muscle_group: qd.types.ndarray(),
):
    for i_e in range(n_elements):
        i_global = i_e + element_el_start
        system.elements_i.muscle_group[i_global] = muscle_group[i_e]


@qd.kernel
def _kernel_set_muscle_direction(
    system: qd.template(),
    element_el_start: qd.i32,
    n_elements: qd.i32,
    muscle_direction: qd.types.ndarray(),
):
    for i_e in range(n_elements):
        i_global = i_e + element_el_start
        for j in qd.static(range(3)):
            system.elements_i.muscle_direction[i_global][j] = muscle_direction[i_e, j]


@qd.kernel
def _kernel_get_el2v(
    system: qd.template(),
    element_el_start: qd.i32,
    element_v_start: qd.i32,
    el2v: qd.types.ndarray(),
    n_elements: qd.i32,
):
    for i_e_ in range(n_elements):
        i_e = i_e_ + element_el_start
        for j in qd.static(range(4)):
            el2v[i_e_, j] = system.elements_i.el2v[i_e][j] - element_v_start


@qd.kernel
def _kernel_get_state(
    system: qd.template(),
    f: qd.i32,
    pos: qd.types.ndarray(),  # shape [B, n_vertices, 3]
    vel: qd.types.ndarray(),  # shape [B, n_vertices, 3]
    active: qd.types.ndarray(),  # shape [B, n_elements]
):
    for i_v, i_b in qd.ndrange(system.n_vertices, system._B):
        for j in qd.static(range(3)):
            pos[i_b, i_v, j] = system.elements_v.pos[f, i_v, i_b][j]
            vel[i_b, i_v, j] = system.elements_v.vel[f, i_v, i_b][j]

    for i_e, i_b in qd.ndrange(system.n_elements, system._B):
        active[i_b, i_e] = system.elements_el_ng.active[f, i_e, i_b]


@qd.kernel
def _kernel_get_state_render(system: qd.template(), f: qd.i32):
    for i_vv, i_b in qd.ndrange(system._n_vverts, system._B):
        i_v = system.vverts_info.vert_idx[i_vv]
        for j in qd.static(range(3)):
            system.vverts_render.pos[i_vv, i_b][j] = qd.cast(system.elements_v.pos[f, i_v, i_b][j], qd.f32)


@qd.kernel
def _kernel_add_vverts(
    system: qd.template(),
    vvert_start: qd.i32,
    vface_start: qd.i32,
    v_start: qd.i32,
    verts_idx: qd.types.ndarray(),
    uvs: qd.types.ndarray(element_dim=1),
    vfaces: qd.types.ndarray(element_dim=1),
):
    n_vverts_local = verts_idx.shape[0]
    for i_vv_ in range(n_vverts_local):
        system.vverts_info.vert_idx[i_vv_ + vvert_start] = verts_idx[i_vv_] + v_start

    n_uvs = uvs.shape[0]
    for i_vv_ in range(n_uvs):
        system.vverts_uvs[i_vv_ + vvert_start] = uvs[i_vv_]

    n_vfaces_local = vfaces.shape[0]
    for i_vf_ in range(n_vfaces_local):
        system.vfaces_indices[i_vf_ + vface_start] = vfaces[i_vf_] + vvert_start


@qd.kernel
def _kernel_set_state(
    system: qd.template(),
    f: qd.i32,
    pos: qd.types.ndarray(),  # shape [B, n_vertices, 3]
    vel: qd.types.ndarray(),  # shape [B, n_vertices, 3]
    active: qd.types.ndarray(),  # shape [B, n_elements]
):
    for i_v, i_b in qd.ndrange(system.n_vertices, system._B):
        for j in qd.static(range(3)):
            system.elements_v.pos[f, i_v, i_b][j] = pos[i_b, i_v, j]
            system.elements_v.vel[f, i_v, i_b][j] = vel[i_b, i_v, j]

    for i_e, i_b in qd.ndrange(system.n_elements, system._B):
        system.elements_el_ng.active[f, i_e, i_b] = active[i_b, i_e]


@qd.kernel
def _kernel_add_grad_from_pos(system: qd.template(), f: qd.i32, pos_grad: qd.types.ndarray()):
    for i_v, i_b in qd.ndrange(system.n_vertices, system._B):
        for j in qd.static(range(3)):
            system.elements_v.pos.grad[f, i_v, i_b][j] += pos_grad[i_b, i_v, j]


@qd.kernel
def _kernel_add_grad_from_vel(system: qd.template(), f: qd.i32, vel_grad: qd.types.ndarray()):
    for i_v, i_b in qd.ndrange(system.n_vertices, system._B):
        for j in qd.static(range(3)):
            system.elements_v.vel.grad[f, i_v, i_b][j] += vel_grad[i_b, i_v, j]


@qd.kernel
def _kernel_update_linked_vertex_constraints(
    system: qd.template(),
    links_state: array_class.LinksState,
):
    for i_v, i_b in qd.ndrange(system.n_vertices, system._B):
        if system.vertex_constraints.is_constrained[i_v, i_b] and system.vertex_constraints.link_idx[i_v, i_b] >= 0:
            i_l = system.vertex_constraints.link_idx[i_v, i_b]
            pos = links_state.pos[i_l, i_b]
            quat = links_state.quat[i_l, i_b]

            offset_pos = system.vertex_constraints.link_offset_pos[i_v, i_b]
            offset_quat = qd_transform_quat_by_quat(system.vertex_constraints.link_init_quat[i_v, i_b], quat)
            system.vertex_constraints.target_pos[i_v, i_b] = pos + qd_transform_by_quat(offset_pos, offset_quat)


@qd.kernel
def apply_hard_constraints(system: qd.template(), f: qd.i32):
    """Apply hard constraints by directly overriding positions and velocities."""
    for i_v, i_b in qd.ndrange(system.n_vertices, system._B):
        if (
            system.vertex_constraints.is_constrained[i_v, i_b]
            and not system.vertex_constraints.is_soft_constraint[i_v, i_b]
        ):
            system.elements_v.pos[f + 1, i_v, i_b] = system.vertex_constraints.target_pos[i_v, i_b]
            system.elements_v.vel[f + 1, i_v, i_b].fill(0.0)


@qd.kernel
def apply_soft_constraints(system: qd.template(), f: qd.i32):
    """Apply soft constraints as spring forces for explicit solver."""
    for i_v, i_b in qd.ndrange(system.n_vertices, system._B):
        if (
            system.vertex_constraints.is_constrained[i_v, i_b]
            and system.vertex_constraints.is_soft_constraint[i_v, i_b]
        ):
            pos_error = system.elements_v.pos[f, i_v, i_b] - system.vertex_constraints.target_pos[i_v, i_b]
            vel_error = system.elements_v.vel[f + 1, i_v, i_b] - system.elements_v.vel[f, i_v, i_b]
            stiffness = system.vertex_constraints.stiffness[i_v, i_b]
            spring_force = -stiffness * pos_error
            damping_force = -2.0 * qd.math.sqrt(stiffness) * vel_error

            dv = system.substep_dt * (spring_force + damping_force)
            system.elements_v.vel[f + 1, i_v, i_b] += dv


@qd.kernel
def _kernel_set_vertex_constraints(
    system: qd.template(),
    f: qd.i32,
    verts_idx: qd.types.ndarray(),  # shape [n_selected_envs, n_verts]
    envs_idx: qd.types.ndarray(),  # shape [n_selected_envs]
    link_idx: qd.i32,
    is_soft_constraint: qd.i32,
    stiffness: qd.f32,
    target_poss: qd.types.ndarray(),  # shape [n_selected_envs, n_verts, 3]
    link_init_pos: qd.types.ndarray(),  # shape [n_selected_envs, 3]
    link_init_quat: qd.types.ndarray(),  # shape [n_selected_envs, 4]
):
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_v = verts_idx[i_b_, i_v_]
        system.vertex_constraints.is_constrained[i_v, i_b] = True
        system.vertex_constraints.is_soft_constraint[i_v, i_b] = qd.cast(is_soft_constraint, gs.qd_bool)
        system.vertex_constraints.stiffness[i_v, i_b] = stiffness
        system.vertex_constraints.link_idx[i_v, i_b] = link_idx

        cur_pos = system.elements_v.pos[f, i_v, i_b]
        for j in qd.static(range(3)):
            system.vertex_constraints.target_pos[i_v, i_b][j] = target_poss[i_b_, i_v_, j]
            system.vertex_constraints.link_offset_pos[i_v, i_b][j] = cur_pos[j] - link_init_pos[i_b_, j]
        for j in qd.static(range(4)):
            system.vertex_constraints.link_init_quat[i_v, i_b][j] = link_init_quat[i_b_, j]


@qd.kernel
def _kernel_update_constraint_targets(
    system: qd.template(),
    verts_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    new_target_poss: qd.types.ndarray(),
):
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_v = verts_idx[i_b_, i_v_]
        for j in qd.static(range(3)):
            system.vertex_constraints.target_pos[i_v, i_b][j] = new_target_poss[i_b_, i_v_, j]


@qd.kernel
def _kernel_remove_specific_constraints(
    system: qd.template(), verts_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray()
):
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_v = verts_idx[i_b_, i_v_]
        system.vertex_constraints.is_constrained[i_v, i_b] = False
