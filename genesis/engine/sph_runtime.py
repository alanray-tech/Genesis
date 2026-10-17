"""Free kernels and functions of SPH simulation over SPHSolverSystem-owned data."""

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.spatial_hash import func_compute_reordered_idx, func_for_all_neighbors


@qd.func(requires_top_level=True)
def func_reorder_particles(
    system: qd.template(),
    f: int,
    has_rigid_coupling: qd.template(),
    rigid_coupling: qd.template(),
    n_geoms: qd.template(),
):
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

            system.particles_reordered.pos[reordered_idx, i_b] = system.particles.pos[i_p, i_b]
            system.particles_reordered.vel[reordered_idx, i_b] = system.particles.vel[i_p, i_b]
            system.particles_reordered.acc[reordered_idx, i_b] = system.particles.acc[i_p, i_b]
            system.particles_reordered.rho[reordered_idx, i_b] = system.particles.rho[i_p, i_b]
            system.particles_reordered.p[reordered_idx, i_b] = system.particles.p[i_p, i_b]
            system.particles_reordered.dfsph_factor[reordered_idx, i_b] = system.particles.dfsph_factor[i_p, i_b]
            system.particles_reordered.drho[reordered_idx, i_b] = system.particles.drho[i_p, i_b]
            system.particles_info_reordered.rho[reordered_idx, i_b] = system.particles_info.rho[i_p]
            system.particles_info_reordered.mass[reordered_idx, i_b] = system.particles_info.mass[i_p]
            system.particles_info_reordered.stiffness[reordered_idx, i_b] = system.particles_info.stiffness[i_p]
            system.particles_info_reordered.exponent[reordered_idx, i_b] = system.particles_info.exponent[i_p]
            system.particles_info_reordered.mu[reordered_idx, i_b] = system.particles_info.mu[i_p]
            system.particles_info_reordered.gamma[reordered_idx, i_b] = system.particles_info.gamma[i_p]
            system.particles_ng_reordered.active[reordered_idx, i_b] = system.particles_ng.active[i_p, i_b]

    if qd.static(has_rigid_coupling):
        for i_p, i_g, i_b in qd.ndrange(system._n_particles, n_geoms, system._B):
            if system.particles_ng.active[i_p, i_b]:
                rigid_coupling.rigid_normal_reordered[system.particles_ng.reordered_idx[i_p, i_b], i_g, i_b] = (
                    rigid_coupling.rigid_normal[i_p, i_g, i_b]
                )


@qd.func(requires_top_level=True)
def func_copy_from_reordered(
    system: qd.template(),
    f: int,
    has_rigid_coupling: qd.template(),
    rigid_coupling: qd.template(),
    n_geoms: qd.template(),
):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng.active[i_p, i_b]:
            # only need to copy back dynamic state, i.e. system.particles
            reordered_idx = system.particles_ng.reordered_idx[i_p, i_b]
            system.particles.pos[i_p, i_b] = system.particles_reordered.pos[reordered_idx, i_b]
            system.particles.vel[i_p, i_b] = system.particles_reordered.vel[reordered_idx, i_b]
            system.particles.acc[i_p, i_b] = system.particles_reordered.acc[reordered_idx, i_b]
            system.particles.rho[i_p, i_b] = system.particles_reordered.rho[reordered_idx, i_b]
            system.particles.p[i_p, i_b] = system.particles_reordered.p[reordered_idx, i_b]
            system.particles.dfsph_factor[i_p, i_b] = system.particles_reordered.dfsph_factor[reordered_idx, i_b]
            system.particles.drho[i_p, i_b] = system.particles_reordered.drho[reordered_idx, i_b]

    if qd.static(has_rigid_coupling):
        for i_p, i_g, i_b in qd.ndrange(system._n_particles, n_geoms, system._B):
            if system.particles_ng.active[i_p, i_b]:
                rigid_coupling.rigid_normal[i_p, i_g, i_b] = rigid_coupling.rigid_normal_reordered[
                    system.particles_ng.reordered_idx[i_p, i_b], i_g, i_b
                ]


@qd.func
def _task_compute_rho(system: qd.template(), i, j, ret: qd.template(), i_b):
    ret += system._particle_volume * cubic_kernel(
        system, (system.particles_reordered.pos[i, i_b] - system.particles_reordered.pos[j, i_b]).norm()
    )


@qd.func(requires_top_level=True)
def func_compute_rho(system: qd.template(), f: int):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng_reordered.active[i_p, i_b]:
            # Base density using the kernel at distance 0
            system.particles_reordered.rho[i_p, i_b] = system._particle_volume * cubic_kernel(system, 0.0)

            den = gs.qd_float(0.0)
            func_for_all_neighbors(
                i_p,
                system.particles_reordered.pos,
                system._support_radius,
                den,
                _task_compute_rho,
                system,
                i_b,
                system.spatial_hasher,
                system.sh,
            )
            system.particles_reordered.rho[i_p, i_b] += den

            system.particles_reordered.rho[i_p, i_b] *= system.particles_info_reordered.rho[i_p, i_b]


@qd.func
def _task_compute_non_pressure_forces(system: qd.template(), i, j, ret: qd.template(), i_b: qd.i32):
    d_ij = system.particles_reordered.pos[i, i_b] - system.particles_reordered.pos[j, i_b]
    dist = d_ij.norm()

    gamma_i = system.particles_info_reordered.gamma[i, i_b]
    mass_i = system.particles_info_reordered.mass[i, i_b]
    mu_i = system.particles_info_reordered.mu[i, i_b]

    mass_j = system.particles_info_reordered.mass[j, i_b]

    # -----------------------------
    # Surface Tension term
    # -----------------------------
    # If distance is bigger than _particle_size, use d_ij.norm() directly; otherwise clamp.
    effective_dist = dist if dist > system._particle_size else system._particle_size

    ret -= gamma_i / mass_i * mass_j * d_ij * cubic_kernel(system, effective_dist)

    # -----------------------------
    # Viscosity Force
    # -----------------------------
    v_ij = (system.particles_reordered.vel[i, i_b] - system.particles_reordered.vel[j, i_b]).dot(d_ij)

    # Some constant factor used in the viscosity formula
    d = 2 * (3 + 2)

    # The density of the neighbor j in batch b
    rho_j = system.particles_reordered.rho[j, i_b]

    f_v = (
        d
        * mu_i
        * (mass_j / rho_j)
        * v_ij
        / (dist**2 + 0.01 * system._support_radius**2)
        * cubic_kernel_derivative(system, d_ij)
    )
    ret += f_v


@qd.func(requires_top_level=True)
def func_compute_non_pressure_forces(system: qd.template(), f: int, t: qd.f32):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng_reordered.active[i_p, i_b]:
            acc = system._gravity[i_b]
            func_for_all_neighbors(
                i_p,
                system.particles_reordered.pos,
                system._support_radius,
                acc,
                _task_compute_non_pressure_forces,
                system,
                i_b,
                system.spatial_hasher,
                system.sh,
            )

            # external force fields
            for i_ff in qd.static(range(len(system._ffs))):
                acc += system._ffs[i_ff].get_acc(
                    system.particles_reordered.pos[i_p, i_b], system.particles_reordered.vel[i_p, i_b], t, i_p
                )
            system.particles_reordered.acc[i_p, i_b] = acc


@qd.func
def _task_compute_pressure_forces(system: qd.template(), i, j, ret: qd.template(), i_b):
    dp_i = system.particles_reordered.p[i, i_b] / system.particles_reordered.rho[i, i_b] ** 2
    rho_j = (
        system.particles_reordered.rho[j, i_b]
        * system.particles_info_reordered.rho[j, i_b]
        / system.particles_info_reordered.rho[j, i_b]
    )
    dp_j = system.particles_reordered.p[j, i_b] / rho_j**2

    # Compute the pressure force contribution, Symmetric Formula
    ret += (
        -system.particles_info_reordered.rho[j, i_b]
        * system._particle_volume
        * (dp_i + dp_j)
        * cubic_kernel_derivative(
            system, system.particles_reordered.pos[i, i_b] - system.particles_reordered.pos[j, i_b]
        )
    )


@qd.func(requires_top_level=True)
def func_compute_pressure_forces(system: qd.template(), f: int):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng_reordered.active[i_p, i_b]:
            rho0 = system.particles_info_reordered.rho[i_p, i_b]
            stiff = system.particles_info_reordered.stiffness[i_p, i_b]
            expnt = system.particles_info_reordered.exponent[i_p, i_b]

            system.particles_reordered.rho[i_p, i_b] = qd.max(system.particles_reordered.rho[i_p, i_b], rho0)

            system.particles_reordered.p[i_p, i_b] = stiff * (
                qd.pow(system.particles_reordered.rho[i_p, i_b] / rho0, expnt) - 1.0
            )

    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng_reordered.active[i_p, i_b]:
            acc = qd.Vector.zero(gs.qd_float, 3)

            func_for_all_neighbors(
                i_p,
                system.particles_reordered.pos,  # shape [n_particles, B, 3] or similar
                system._support_radius,
                acc,
                _task_compute_pressure_forces,
                system,
                i_b,
                system.spatial_hasher,
                system.sh,
            )
            system.particles_reordered.acc[i_p, i_b] += acc


@qd.func(requires_top_level=True)
def func_advect_velocity(system: qd.template(), f: int):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng_reordered.active[i_p, i_b]:
            system.particles_reordered.vel[i_p, i_b] += system._substep_dt * system.particles_reordered.acc[i_p, i_b]


@qd.func(requires_top_level=True)
def func_advect_position(system: qd.template(), f: int):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng_reordered.active[i_p, i_b]:
            pos = system.particles_reordered.pos[i_p, i_b]
            new_vel = system.particles_reordered.vel[i_p, i_b]

            # advect
            new_pos = pos + system._substep_dt * new_vel

            # impose boundary (pass b if needed)
            corrected_pos, corrected_vel = system.boundary.impose_pos_vel(new_pos, new_vel)

            # update
            system.particles_reordered.vel[i_p, i_b] = corrected_vel
            system.particles_reordered.pos[i_p, i_b] = corrected_pos


@qd.func
def _task_compute_DFSPH_factor(system: qd.template(), i, j, ret: qd.template(), i_b):
    # Fluid neighbors
    grad_j = -system._particle_volume * cubic_kernel_derivative(
        system, system.particles_reordered.pos[i, i_b] - system.particles_reordered.pos[j, i_b]
    )
    ret[3] += grad_j.norm_sqr()  # sum_grad_p_k
    for ii in qd.static(range(3)):  # grad_p_i
        ret[ii] -= grad_j[ii]


@qd.func(requires_top_level=True)
def func_compute_DFSPH_factor(system: qd.template(), f: int):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng_reordered.active[i_p, i_b]:
            sum_grad_p_k = gs.qd_float(0.0)
            grad_p_i = qd.Vector.zero(gs.qd_float, 3)

            # `ret` concatenates `grad_p_i` and `sum_grad_p_k`
            ret = qd.Vector.zero(gs.qd_float, 4)

            func_for_all_neighbors(
                i_p,
                system.particles_reordered.pos,
                system._support_radius,
                ret,
                _task_compute_DFSPH_factor,
                system,
                i_b,
                system.spatial_hasher,
                system.sh,
            )

            sum_grad_p_k = ret[3]
            for ii in qd.static(range(3)):
                grad_p_i[ii] = ret[ii]
            sum_grad_p_k += grad_p_i.norm_sqr()

            # Compute pressure stiffness denominator
            factor = gs.qd_float(0.0)
            if sum_grad_p_k > 1e-6:
                factor = -1.0 / sum_grad_p_k
            else:
                factor = 0.0
            system.particles_reordered.dfsph_factor[i_p, i_b] = factor

            # Reset the implicit pressure accumulated by the density solve (see
            # _kernel_density_solve_iteration).
            system.particles_reordered.p[i_p, i_b] = 0.0


@qd.func
def _task_compute_density_time_derivative(system: qd.template(), i, j, ret: qd.template(), i_b):
    v_i = system.particles_reordered.vel[i, i_b]
    v_j = system.particles_reordered.vel[j, i_b]

    x_i = system.particles_reordered.pos[i, i_b]
    x_j = system.particles_reordered.pos[j, i_b]

    # Fluid neighbors
    ret.drho += system._particle_volume * (v_i - v_j).dot(cubic_kernel_derivative(system, x_i - x_j))
    ret.num_neighbors += 1


@qd.func(requires_top_level=True)
def func_compute_density_time_derivative(system: qd.template()):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng_reordered.active[i_p, i_b]:
            ret = qd.Struct(drho=0.0, num_neighbors=0)
            func_for_all_neighbors(
                i_p,
                system.particles_reordered.pos,
                system._support_radius,
                ret,
                _task_compute_density_time_derivative,
                system,
                i_b,
                system.spatial_hasher,
                system.sh,
            )

            # only correct positive divergence
            drho = qd.max(ret.drho, 0.0)
            num_neighbors = ret.num_neighbors

            # Do not perform divergence solve when paritlce deficiency happens
            if num_neighbors < 20:
                drho = 0.0

            system.particles_reordered.drho[i_p, i_b] = drho


@qd.func
def _task_divergence_solver_iteration(system: qd.template(), i, j, ret: qd.template(), i_b):
    # Fluid neighbors
    b_j = system.particles_reordered.drho[j, i_b]
    k_j = b_j * system.particles_reordered.dfsph_factor[j, i_b]
    k_sum = (
        system._density0 / system._density0 * ret.k_i + k_j
    )  # TODO: make the neighbor density different for multiphase fluid
    if qd.abs(k_sum) > system._df_eps:
        grad_p_j = -system._particle_volume * cubic_kernel_derivative(
            system, system.particles_reordered.pos[i, i_b] - system.particles_reordered.pos[j, i_b]
        )
        ret.dv -= (
            k_sum * grad_p_j
        )  # ki, kj already contain inverse density, i.e., density canceled if not mutiphase flow


@qd.func(requires_top_level=True)
def func_divergence_solver_iteration(system: qd.template()):
    # Perform Jacobi iteration
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng_reordered.active[i_p, i_b]:
            # evaluate rhs
            b_i = system.particles_reordered.drho[i_p, i_b]
            k_i = b_i * system.particles_reordered.dfsph_factor[i_p, i_b]
            ret = qd.Struct(dv=qd.Vector.zero(gs.qd_float, 3), k_i=k_i)
            # TODO: if warm start
            # get_kappa_V += k_i
            func_for_all_neighbors(
                i_p,
                system.particles_reordered.pos,
                system._support_radius,
                ret,
                _task_divergence_solver_iteration,
                system,
                i_b,
                system.spatial_hasher,
                system.sh,
            )
            system.particles_reordered.vel[i_p, i_b] = system.particles_reordered.vel[i_p, i_b] + ret.dv


@qd.func(requires_top_level=True)
def func_compute_density_error(system: qd.template(), offset: float):
    for _ in range(1):
        system.density_error[None] = 0.0
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng_reordered.active[i_p, i_b]:
            qd.atomic_add(
                system.density_error[None],
                system._density0 * system.particles_reordered.drho[i_p, i_b] - offset,
            )


@qd.func(requires_top_level=True)
def func_predict_velocity(system: qd.template(), f: int):
    # compute new velocities only considering non-pressure forces
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng_reordered.active[i_p, i_b]:
            system.particles_reordered.vel[i_p, i_b] += system._substep_dt * system.particles_reordered.acc[i_p, i_b]


@qd.func
def _task_compute_density_star(system: qd.template(), i, j, ret: qd.template(), i_b):
    v_i = system.particles_reordered.vel[i, i_b]
    v_j = system.particles_reordered.vel[j, i_b]
    x_i = system.particles_reordered.pos[i, i_b]
    x_j = system.particles_reordered.pos[j, i_b]
    ret += system._particle_volume * (v_i - v_j).dot(cubic_kernel_derivative(system, x_i - x_j))


@qd.func(requires_top_level=True)
def func_compute_density_star(system: qd.template()):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng_reordered.active[i_p, i_b]:
            delta = gs.qd_float(0.0)
            func_for_all_neighbors(
                i_p,
                system.particles_reordered.pos,
                system._support_radius,
                delta,
                _task_compute_density_star,
                system,
                i_b,
                system.spatial_hasher,
                system.sh,
            )
            drho = system.particles_reordered.rho[i_p, i_b] / system._density0 + system._substep_dt * delta
            system.particles_reordered.drho[i_p, i_b] = qd.max(drho, 1.0)  # - 1.0


@qd.func
def density_solve_iteration_task(system: qd.template(), i, j, ret: qd.template(), i_b):
    # Fluid neighbors
    b_j = system.particles_reordered.drho[j, i_b] - 1.0
    k_j = b_j * system.particles_reordered.dfsph_factor[j, i_b]
    k_sum = (
        system._density0 / system._density0 * ret.k_i + k_j
    )  # TODO: make the neighbor density0 different for multiphase fluid
    if qd.abs(k_sum) > system._df_eps:
        grad_p_j = -system._particle_volume * cubic_kernel_derivative(
            system, system.particles_reordered.pos[i, i_b] - system.particles_reordered.pos[j, i_b]
        )
        # Directly update velocities instead of storing pressure accelerations
        ret.dv -= (
            system._substep_dt * k_sum * grad_p_j
        )  # ki, kj already contain inverse density, i.e., density canceled if not mutiphase flow


@qd.func(requires_top_level=True)
def func_density_solve_iteration(system: qd.template()):
    # Compute pressure forces
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng_reordered.active[i_p, i_b]:
            # Evaluate rhs
            b_i = system.particles_reordered.drho[i_p, i_b] - 1.0
            k_i = b_i * system.particles_reordered.dfsph_factor[i_p, i_b]

            ret = qd.Struct(dv=qd.Vector.zero(gs.qd_float, 3), k_i=k_i)

            # TODO: if warmstart
            # get kappa V
            func_for_all_neighbors(
                i_p,
                system.particles_reordered.pos,
                system._support_radius,
                ret,
                density_solve_iteration_task,
                system,
                i_b,
                system.spatial_hasher,
                system.sh,
            )
            system.particles_reordered.vel[i_p, i_b] = system.particles_reordered.vel[i_p, i_b] + ret.dv

            # The stiffness k_i is the pressure this Jacobi iteration applies to cancel the current density
            # error, up to the -1/rho0 scaling carried by dfsph_factor, so summing -k_i * rho0 over the
            # iterations of the density solve yields the implicit pressure field that enforces
            # incompressibility over the substep. This exposes in particles.p the same physical quantity
            # that the equation of state provides under WCSPH. The field is zeroed once per substep in
            # _kernel_compute_DFSPH_factor.
            rho0 = system.particles_info_reordered.rho[i_p, i_b]
            system.particles_reordered.p[i_p, i_b] = system.particles_reordered.p[i_p, i_b] - k_i * rho0


@qd.func(requires_top_level=True)
def func_multiply_time_step(system: qd.template(), field: qd.Tensor, time_step: float):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng_reordered.active[i_p, i_b]:
            field[i_p, i_b] *= time_step


@qd.func(requires_top_level=True)
def func_set_graph_counter(graph_counter: qd.types.ndarray(qd.i32, ndim=0), n_iterations: int):
    for _ in range(1):
        graph_counter[()] = n_iterations


@qd.func(requires_top_level=True)
def func_check_divergence_convergence(system: qd.template(), graph_counter: qd.types.ndarray(qd.i32, ndim=0)):
    for _ in range(1):
        graph_counter[()] -= 1
        avg_density_err = system.density_error[None] / system._n_particles
        eta = 1.0 / system._substep_dt * system._df_max_error_div * 0.01 * system._density0
        if avg_density_err <= eta:
            graph_counter[()] = 0


@qd.func(requires_top_level=True)
def func_check_density_convergence(system: qd.template(), graph_counter: qd.types.ndarray(qd.i32, ndim=0)):
    for _ in range(1):
        graph_counter[()] -= 1
        avg_density_err = system.density_error[None] / system._n_particles
        eta = system._df_max_error_den * 0.01 * system._density0
        if avg_density_err <= eta:
            graph_counter[()] = 0


@qd.func(requires_top_level=True)
def func_divergence_solve(system: qd.template(), graph_counter: qd.types.ndarray(qd.i32, ndim=0)):
    func_compute_density_time_derivative(system)
    func_set_graph_counter(graph_counter, system._df_max_div_iters)
    while qd.graph.do_while(graph_counter):
        func_divergence_solver_iteration(system)
        func_compute_density_time_derivative(system)
        func_compute_density_error(system, 0.0)
        func_check_divergence_convergence(system, graph_counter)


@qd.func(requires_top_level=True)
def func_density_solve(system: qd.template(), graph_counter: qd.types.ndarray(qd.i32, ndim=0)):
    func_compute_density_star(system)
    func_multiply_time_step(
        system=system,
        field=system.particles_reordered.dfsph_factor,
        time_step=1.0 / system._substep_dt**2,
    )
    func_set_graph_counter(graph_counter, system._df_max_den_iters)
    while qd.graph.do_while(graph_counter):
        func_density_solve_iteration(system)
        func_compute_density_star(system)
        func_compute_density_error(system, system._density0)
        func_check_density_convergence(system, graph_counter)


@qd.kernel
def _kernel_reorder_particles(
    system: qd.template(),
    f: int,
    has_rigid_coupling: qd.template(),
    rigid_coupling: qd.template(),
    n_geoms: qd.template(),
):
    func_reorder_particles(system, f, has_rigid_coupling, rigid_coupling, n_geoms)


@qd.kernel
def _kernel_compute_rho(system: qd.template(), f: int):
    func_compute_rho(system, f)


@qd.kernel
def kernel_wcsph_pre_coupling(
    system: qd.template(),
    f: int,
    t: qd.f32,
    has_rigid_coupling: qd.template(),
    rigid_coupling: qd.template(),
    n_geoms: qd.template(),
):
    func_reorder_particles(system, f, has_rigid_coupling, rigid_coupling, n_geoms)
    func_compute_rho(system, f)
    func_compute_non_pressure_forces(system, f, t)
    func_compute_pressure_forces(system, f)
    func_advect_velocity(system, f)


@qd.kernel(graph=True)
def kernel_dfsph_pre_coupling(
    system: qd.template(),
    f: int,
    t: qd.f32,
    graph_counter: qd.types.ndarray(qd.i32, ndim=0),
    has_rigid_coupling: qd.template(),
    rigid_coupling: qd.template(),
    n_geoms: qd.template(),
):
    func_reorder_particles(system, f, has_rigid_coupling, rigid_coupling, n_geoms)
    func_compute_rho(system, f)
    func_compute_DFSPH_factor(system, f)
    func_divergence_solve(system, graph_counter)
    func_advect_velocity(system, f)
    func_compute_non_pressure_forces(system, f, t)
    func_predict_velocity(system, f)
    func_density_solve(system, graph_counter)


@qd.kernel
def kernel_post_coupling(
    system: qd.template(),
    f: int,
    has_rigid_coupling: qd.template(),
    rigid_coupling: qd.template(),
    n_geoms: qd.template(),
):
    func_advect_position(system, f)
    func_copy_from_reordered(system, f, has_rigid_coupling, rigid_coupling, n_geoms)


@qd.func
def cubic_kernel(system: qd.template(), r_norm):
    """
    Cubic spline smoothing kernel.
    """
    res = gs.qd_float(0.0)
    h = system._support_radius
    k = 8.0 / np.pi / h**3
    q = r_norm / h
    if q <= 1.0:
        if q <= 0.5:
            q2 = q**2
            q3 = q2 * q
            res = k * (6.0 * q3 - 6.0 * q2 + 1.0)
        else:
            res = 2 * k * (1.0 - q) ** 3
    return res


@qd.func
def cubic_kernel_derivative(system: qd.template(), r):
    """
    Derivative of cubic spline smoothing kernel.
    """
    res = qd.Vector.zero(gs.qd_float, 3)

    r_norm = r.norm()
    h = system._support_radius
    k = 8.0 / np.pi / h**3
    q = r_norm / h
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            res = 6.0 * k * q * (3.0 * q - 2.0) * grad_q
        else:
            res = -6.0 * k * (1.0 - q) ** 2 * grad_q
    return res


@qd.func
def cubic_kernel_plane_integral(system: qd.template(), dist):
    """
    Integral of the cubic spline smoothing kernel over the plane at distance dist from its center.

    Closed form of 2 * pi * int_{|dist|}^{h} W(r) r dr. Its own integral along the plane normal equals
    half the kernel normalization: int_{0}^{h} of this quantity over dist is exactly 1/2, which makes
    forces weighted by it integrate to a per-unit-area magnitude over a covering layer of particles.
    """
    res = gs.qd_float(0.0)
    h = system._support_radius
    q = qd.abs(dist) / h
    if q <= 1.0:
        q2 = q**2
        q3 = q2 * q
        if q <= 0.5:
            res = 0.0875 - (1.2 * q3 * q2 - 1.5 * q2**2 + 0.5 * q2)
        else:
            res = 0.1 - (q2 - 2.0 * q3 + 1.5 * q2**2 - 0.4 * q3 * q2)
        res = res * 16.0 / h
    return res


@qd.kernel
def _kernel_set_state(
    system: qd.template(),
    f: qd.i32,
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    active: qd.types.ndarray(),
):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        for j in qd.static(range(3)):
            system.particles.pos[i_p, i_b][j] = pos[i_b, i_p, j]
            system.particles.vel[i_p, i_b][j] = vel[i_b, i_p, j]
        system.particles_ng.active[i_p, i_b] = active[i_b, i_p]


@qd.kernel
def _kernel_get_state(
    system: qd.template(),
    f: qd.i32,
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    active: qd.types.ndarray(),
):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        for j in qd.static(range(3)):
            pos[i_b, i_p, j] = system.particles.pos[i_p, i_b][j]
            vel[i_b, i_p, j] = system.particles.vel[i_p, i_b][j]
        active[i_b, i_p] = system.particles_ng.active[i_p, i_b]


@qd.kernel
def _kernel_update_render_fields(system: qd.template(), f: qd.i32):
    for i_p, i_b in qd.ndrange(system._n_particles, system._B):
        if system.particles_ng.active[i_p, i_b]:
            system.particles_render.pos[i_p, i_b] = system.particles.pos[i_p, i_b]
            system.particles_render.vel[i_p, i_b] = system.particles.vel[i_p, i_b]
        else:
            system.particles_render.pos[i_p, i_b] = gu.qd_nowhere()
        system.particles_render.active[i_p, i_b] = system.particles_ng.active[i_p, i_b]


@qd.kernel
def _kernel_add_particles(
    system: qd.template(),
    f: qd.i32,
    active: qd.i32,
    particle_start: qd.i32,
    n_particles: qd.i32,
    mat_rho: qd.f32,
    mat_stiffness: qd.f32,
    mat_exponent: qd.f32,
    mat_mu: qd.f32,
    mat_gamma: qd.f32,
    pos: qd.types.ndarray(),
):
    for i_p_, i_b in qd.ndrange(n_particles, system._B):
        i_p = i_p_ + particle_start
        system.particles_ng.active[i_p, i_b] = qd.cast(active, gs.qd_bool)
        for i in qd.static(range(3)):
            system.particles.pos[i_p, i_b][i] = pos[i_p_, i]
        system.particles.vel[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        system.particles.p[i_p, i_b] = 0

    for i_p_ in range(n_particles):
        i_p = i_p_ + particle_start
        system.particles_info.rho[i_p] = mat_rho
        system.particles_info.stiffness[i_p] = mat_stiffness
        system.particles_info.exponent[i_p] = mat_exponent
        system.particles_info.mu[i_p] = mat_mu
        system.particles_info.gamma[i_p] = mat_gamma
        system.particles_info.mass[i_p] = system._particle_volume * mat_rho


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
        system.particles.acc[i_p, i_b].fill(0.0)


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
        system.particles.acc[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)


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
        system.particles_ng.active[i_p, i_b] = actives[i_b_, i_p_]


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
def _kernel_get_mass(
    system: qd.template(),
    particle_start: qd.i32,
    n_particles: qd.i32,
    mass: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
):
    for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
        i_p = i_p_ + particle_start
        i_b = envs_idx[i_b_]
        mass[i_b_] += system.particles_info.mass[i_p]


@qd.kernel
def kernel_get_entity_frame(
    system: qd.template(),
    f: int,
    particle_start: int,
    n_particles: int,
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
):
    for i_p_, i_b in qd.ndrange(n_particles, system._B):
        i_p = i_p_ + particle_start
        for k in qd.static(range(3)):
            pos[i_b, i_p_, k] = system.particles.pos[i_p, i_b][k]
            vel[i_b, i_p_, k] = system.particles.vel[i_p, i_b][k]
