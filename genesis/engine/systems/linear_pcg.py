from __future__ import annotations

import numpy as np
import quadrants as qd

from .sim_system import SimSystem


@qd.data_oriented
class LinearPCG(SimSystem):
    """Solve the global Newton system with participant operators and preconditioners."""

    def __init__(self, n_dofs: int) -> None:
        super().__init__()
        self.n_dofs = n_dofs
        n_storage_dofs = max(n_dofs, 1)

        self.n_dofs_device = qd.ndarray(qd.i32, shape=(1,))
        self.residual = qd.ndarray(qd.f64, shape=(n_storage_dofs,))
        self.preconditioned_residual = qd.ndarray(qd.f64, shape=(n_storage_dofs,))
        self.direction = qd.ndarray(qd.f64, shape=(n_storage_dofs,))
        self.operator_direction = qd.ndarray(qd.f64, shape=(n_storage_dofs,))

        self.residual_preconditioned = qd.ndarray(qd.f64, shape=(1,))
        self.residual_preconditioned_initial = qd.ndarray(qd.f64, shape=(1,))
        self.residual_preconditioned_next = qd.ndarray(qd.f64, shape=(1,))
        self.direction_operator_direction = qd.ndarray(qd.f64, shape=(1,))
        self.alpha = qd.ndarray(qd.f64, shape=(1,))
        self.beta = qd.ndarray(qd.f64, shape=(1,))
        self.condition = qd.ndarray(qd.i32, shape=())
        self.is_active = qd.ndarray(qd.i32, shape=())
        self.n_iterations = qd.ndarray(qd.i32, shape=(1,))
        self.is_failed = qd.ndarray(qd.i32, shape=())

    def build(self) -> None:
        self.n_dofs_device.from_numpy(np.array([self.n_dofs], dtype=np.int32))

    @qd.func(requires_top_level=True)
    def initialize(self, linear_system: qd.template(), solve_plan: qd.template()):
        for i_d in range(self.n_dofs_device[0]):
            linear_system.solution[i_d] = qd.f64(0.0)
            self.residual[i_d] = -linear_system.rhs[i_d]
            self.preconditioned_residual[i_d] = qd.f64(0.0)

        solve_plan.apply_preconditioner(self.residual, self.preconditioned_residual)

        for i_d in range(self.n_dofs_device[0]):
            self.direction[i_d] = self.preconditioned_residual[i_d]

        for _ in range(1):
            self.residual_preconditioned[0] = qd.f64(0.0)
            self.n_iterations[0] = 0
            self.is_failed[()] = 0
        for i_d in range(self.n_dofs_device[0]):
            qd.atomic_add(
                self.residual_preconditioned[0],
                self.residual[i_d] * self.preconditioned_residual[i_d],
            )

        for _ in range(1):
            self.residual_preconditioned_initial[0] = self.residual_preconditioned[0]
            self.condition[()] = 1
            self.is_active[()] = qd.i32(self.residual_preconditioned[0] > 0.0)

    @qd.func(requires_top_level=True)
    def iteration(
        self,
        linear_system: qd.template(),
        solve_plan: qd.template(),
        tolerance: qd.template(),
        max_iterations: qd.template(),
    ):
        for i_d in range(self.n_dofs_device[0]):
            if self.is_active[()] != 0:
                self.operator_direction[i_d] = qd.f64(0.0)

        linear_system.apply_bcoo(self.direction, self.operator_direction)
        solve_plan.apply_hessian(self.direction, self.operator_direction)

        for _ in range(1):
            self.direction_operator_direction[0] = qd.f64(0.0)
        for i_d in range(self.n_dofs_device[0]):
            if self.is_active[()] != 0:
                qd.atomic_add(
                    self.direction_operator_direction[0],
                    self.direction[i_d] * self.operator_direction[i_d],
                )

        for _ in range(1):
            if self.is_active[()] != 0:
                if self.direction_operator_direction[0] > 0.0:
                    self.alpha[0] = self.residual_preconditioned[0] / self.direction_operator_direction[0]
                else:
                    self.alpha[0] = qd.f64(0.0)
                    self.is_failed[()] = 1
                    self.is_active[()] = 0

        for i_d in range(self.n_dofs_device[0]):
            if self.is_active[()] != 0:
                linear_system.solution[i_d] = linear_system.solution[i_d] + self.alpha[0] * self.direction[i_d]
                self.residual[i_d] = self.residual[i_d] - self.alpha[0] * self.operator_direction[i_d]
                self.preconditioned_residual[i_d] = qd.f64(0.0)

        solve_plan.apply_preconditioner(self.residual, self.preconditioned_residual)

        for _ in range(1):
            self.residual_preconditioned_next[0] = qd.f64(0.0)
        for i_d in range(self.n_dofs_device[0]):
            if self.is_active[()] != 0:
                qd.atomic_add(
                    self.residual_preconditioned_next[0],
                    self.residual[i_d] * self.preconditioned_residual[i_d],
                )

        for _ in range(1):
            if self.is_active[()] != 0:
                is_converged = qd.abs(self.residual_preconditioned_next[0]) <= tolerance * qd.abs(
                    self.residual_preconditioned_initial[0]
                )
                if is_converged:
                    self.is_active[()] = 0
                elif qd.abs(self.residual_preconditioned[0]) > 0.0:
                    self.beta[0] = self.residual_preconditioned_next[0] / self.residual_preconditioned[0]
                    self.residual_preconditioned[0] = self.residual_preconditioned_next[0]
                else:
                    self.is_failed[()] = 1
                    self.is_active[()] = 0

        for i_d in range(self.n_dofs_device[0]):
            if self.is_active[()] != 0:
                self.direction[i_d] = self.preconditioned_residual[i_d] + self.beta[0] * self.direction[i_d]

        for _ in range(1):
            if self.is_failed[()] == 0 and self.residual_preconditioned_initial[0] > 0.0:
                self.n_iterations[0] = self.n_iterations[0] + 1
            if self.n_iterations[0] >= max_iterations and self.is_active[()] != 0:
                self.is_failed[()] = 1
                self.is_active[()] = 0
            self.condition[()] = self.is_active[()]
