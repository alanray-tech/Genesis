from __future__ import annotations

import numpy as np
import quadrants as qd

from .fsr_reduce import _head_segmented_reduce_add


@qd.data_oriented
class MaskedLinearPCG:
    """Per-component PCG with progressive masking."""

    def __init__(self, total_dof: int, n_block_rows: int) -> None:
        if total_dof < 0 or n_block_rows < 0:
            raise ValueError("MaskedLinearPCG sizes must be non-negative")
        if total_dof != n_block_rows * 3:
            raise ValueError("MaskedLinearPCG requires three scalar DOFs per block row")

        self.total_dof_host = total_dof
        self.n_block_rows_host = n_block_rows
        self.padded_block_rows_host = max(
            ((n_block_rows + 31) // 32) * 32,
            32,
        )
        dof_storage = max(total_dof, 1)
        component_storage = max(n_block_rows, 1)

        self.residual = qd.ndarray(qd.f64, shape=(dof_storage,))
        self.preconditioned_residual = qd.ndarray(
            qd.f64,
            shape=(dof_storage,),
        )
        self.direction = qd.ndarray(qd.f64, shape=(dof_storage,))
        self.operator_direction = qd.ndarray(
            qd.f64,
            shape=(dof_storage,),
        )

        self.rz = qd.ndarray(qd.f64, shape=(component_storage,))
        self.rz_new = qd.ndarray(qd.f64, shape=(component_storage,))
        self.pAp = qd.ndarray(qd.f64, shape=(component_storage,))
        self.alpha = qd.ndarray(qd.f64, shape=(component_storage,))
        self.beta = qd.ndarray(qd.f64, shape=(component_storage,))
        self.tolerance = qd.ndarray(
            qd.f64,
            shape=(component_storage,),
        )
        self.converged = qd.ndarray(qd.i32, shape=(component_storage,))
        self.component_iterations = qd.ndarray(
            qd.i32,
            shape=(component_storage,),
        )

        self.condition = qd.ndarray(qd.i32, shape=())
        self.n_iterations = qd.ndarray(qd.i32, shape=())
        self.is_failed = qd.ndarray(qd.i32, shape=())
        self.rz_total = qd.ndarray(qd.f64, shape=())

        self.condition.from_numpy(np.array(0, dtype=np.int32))
        self.n_iterations.from_numpy(np.array(0, dtype=np.int32))
        self.is_failed.from_numpy(np.array(0, dtype=np.int32))
        self.rz_total.from_numpy(np.array(0.0, dtype=np.float64))

    @qd.func(requires_top_level=True)
    def _segmented_dot(
        self,
        lhs: qd.template(),
        rhs: qd.template(),
        output: qd.template(),
        partitioner: qd.template(),
    ):
        for component in range(self.n_block_rows_host):
            if component < partitioner.K[()]:
                output[component] = 0.0

        qd.loop_config(name="masked_pcg_segmented_dot", block_dim=256)
        for row in range(self.padded_block_rows_host):
            lane = qd.i32(qd.simt.subgroup.invocation_id()) & 31
            valid = row < self.n_block_rows_host
            component = qd.i32(-1)
            value = qd.f64(0.0)
            previous = qd.i32(-2)
            following = qd.i32(-3)
            if valid:
                component = partitioner.comp_label[row]
                if self.converged[component] == 0:
                    offset = row * 3
                    for axis in qd.static(range(3)):
                        value = value + lhs[offset + axis] * rhs[offset + axis]
                if lane > 0 and row > 0:
                    previous = partitioner.comp_label[row - 1]
                if lane < 31 and row + 1 < self.n_block_rows_host:
                    following = partitioner.comp_label[row + 1]

            is_head = valid and (lane == 0 or row == 0 or previous != component)
            is_tail = valid and (lane == 31 or row == self.n_block_rows_host - 1 or following != component)
            reduced = _head_segmented_reduce_add(
                value,
                qd.i32(is_tail),
            )
            if is_head and reduced != 0.0:
                qd.atomic_add(output[component], reduced)

    @qd.func(requires_top_level=True)
    def _apply_preconditioner(
        self,
        partitioner: qd.template(),
        rigid: qd.template(),
        rigid_forest: qd.template(),
        fem_preconditioner: qd.template(),
        has_rigid: qd.template(),
        has_rigid_forest: qd.template(),
        has_fem: qd.template(),
    ):
        for dof in range(self.total_dof_host):
            self.preconditioned_residual[dof] = 0.0

        if qd.static(has_rigid_forest):
            if qd.static(hasattr(rigid_forest, "apply_preconditioner_masked")):
                rigid_forest.apply_preconditioner_masked(
                    self.residual,
                    self.preconditioned_residual,
                    partitioner.comp_label,
                    self.converged,
                )
            else:
                rigid_forest.apply_preconditioner(
                    self.residual,
                    self.preconditioned_residual,
                )
        elif qd.static(has_rigid):
            if qd.static(hasattr(rigid, "apply_preconditioner_masked")):
                rigid.apply_preconditioner_masked(
                    self.residual,
                    self.preconditioned_residual,
                    False,
                    partitioner.comp_label,
                    self.converged,
                )
            else:
                rigid.apply_preconditioner(
                    self.residual,
                    self.preconditioned_residual,
                    False,
                )
        if qd.static(has_fem):
            if qd.static(hasattr(fem_preconditioner, "apply_preconditioner_masked")):
                fem_preconditioner.apply_preconditioner_masked(
                    self.residual,
                    self.preconditioned_residual,
                    partitioner.comp_label,
                    self.converged,
                )
            elif qd.static(hasattr(fem_preconditioner, "apply_masked")):
                fem_preconditioner.apply_masked(
                    self.residual,
                    self.preconditioned_residual,
                    partitioner.comp_label,
                    self.converged,
                )
            else:
                fem_preconditioner.apply(
                    self.residual,
                    self.preconditioned_residual,
                )

        for row in range(self.n_block_rows_host):
            component = partitioner.comp_label[row]
            if self.converged[component] != 0:
                offset = row * 3
                for axis in qd.static(range(3)):
                    self.preconditioned_residual[offset + axis] = 0.0

    @qd.func(requires_top_level=True)
    def _apply_operator(
        self,
        linear_system: qd.template(),
        partitioner: qd.template(),
        rigid: qd.template(),
        rigid_forest: qd.template(),
        has_rigid: qd.template(),
        has_rigid_forest: qd.template(),
    ):
        for row in range(self.n_block_rows_host):
            component = partitioner.comp_label[row]
            offset = row * 3
            for axis in qd.static(range(3)):
                self.operator_direction[offset + axis] = 0.0
                if self.converged[component] != 0:
                    self.direction[offset + axis] = 0.0
                if qd.static(has_rigid_forest):
                    rigid_forest.physical_Ap[offset + axis] = 0.0

        if qd.static(has_rigid_forest):
            if qd.static(
                hasattr(
                    rigid_forest,
                    "prepare_physical_direction_masked",
                )
            ):
                rigid_forest.prepare_physical_direction_masked(
                    self.direction,
                    partitioner.comp_label,
                    self.converged,
                )
            else:
                rigid_forest.prepare_physical_direction(self.direction)
            if qd.static(hasattr(linear_system, "spmv_masked")):
                linear_system.spmv_masked(
                    rigid_forest.physical_p,
                    rigid_forest.physical_Ap,
                    partitioner.comp_label,
                    self.converged,
                )
            else:
                linear_system.spmv(
                    rigid_forest.physical_p,
                    rigid_forest.physical_Ap,
                )
            if qd.static(
                hasattr(
                    rigid_forest,
                    "finish_reduced_spmv_masked",
                )
            ):
                rigid_forest.finish_reduced_spmv_masked(
                    self.direction,
                    self.operator_direction,
                    partitioner.comp_label,
                    self.converged,
                )
            else:
                rigid_forest.finish_reduced_spmv(
                    self.direction,
                    self.operator_direction,
                )
        else:
            if qd.static(hasattr(linear_system, "spmv_masked")):
                linear_system.spmv_masked(
                    self.direction,
                    self.operator_direction,
                    partitioner.comp_label,
                    self.converged,
                )
            else:
                linear_system.spmv(
                    self.direction,
                    self.operator_direction,
                )
        if qd.static(has_rigid):
            if qd.static(hasattr(rigid, "apply_hessian_masked")):
                rigid.apply_hessian_masked(
                    self.direction,
                    self.operator_direction,
                    has_rigid_forest,
                    partitioner.comp_label,
                    self.converged,
                )
            else:
                rigid.apply_hessian(
                    self.direction,
                    self.operator_direction,
                    has_rigid_forest,
                )

    @qd.func(requires_top_level=True)
    def initialize(
        self,
        linear_system: qd.template(),
        partitioner: qd.template(),
        rigid: qd.template(),
        rigid_forest: qd.template(),
        fem_preconditioner: qd.template(),
        has_rigid: qd.template(),
        has_rigid_forest: qd.template(),
        has_fem: qd.template(),
        tolerance_rate,
        max_iterations,
    ):
        for dof in range(self.total_dof_host):
            linear_system.x_sol[dof] = 0.0
            self.residual[dof] = linear_system.b_rhs[dof]
            self.preconditioned_residual[dof] = 0.0
            self.direction[dof] = 0.0

        for component in range(self.n_block_rows_host):
            self.converged[component] = qd.i32(component >= partitioner.K[()])
            self.component_iterations[component] = 0
            self.rz[component] = 0.0
            self.rz_new[component] = 0.0
            self.pAp[component] = 0.0
            self.alpha[component] = 0.0
            self.beta[component] = 0.0
            self.tolerance[component] = 0.0

        for _ in range(1):
            self.condition[()] = 0
            self.n_iterations[()] = 0
            self.is_failed[()] = 0

        self._apply_preconditioner(
            partitioner,
            rigid,
            rigid_forest,
            fem_preconditioner,
            has_rigid,
            has_rigid_forest,
            has_fem,
        )

        for dof in range(self.total_dof_host):
            self.direction[dof] = self.preconditioned_residual[dof]

        self._segmented_dot(
            self.residual,
            self.preconditioned_residual,
            self.rz,
            partitioner,
        )

        qd.loop_config(name="masked_pcg_tolerance_total")
        for _ in range(1):
            self.rz_total[()] = qd.f64(0.0)
            for component in range(partitioner.K[()]):
                self.rz_total[()] = self.rz_total[()] + qd.abs(self.rz[component])

        qd.loop_config(name="masked_pcg_initialize_tolerances")
        for component in range(partitioner.K[()]):
            floor = tolerance_rate * self.rz_total[()]
            self.tolerance[component] = tolerance_rate * qd.max(qd.abs(self.rz[component]), floor)
            component_converged = qd.abs(self.rz[component]) <= self.tolerance[component]
            self.converged[component] = qd.i32(component_converged)
            if not component_converged:
                qd.atomic_or(self.condition[()], qd.i32(1))

        qd.loop_config(name="masked_pcg_validate_iteration_budget")
        for _ in range(1):
            if self.condition[()] != 0 and max_iterations <= 0:
                self.is_failed[()] = 1
                self.condition[()] = 0

    @qd.func(requires_top_level=True)
    def iteration(
        self,
        linear_system: qd.template(),
        partitioner: qd.template(),
        rigid: qd.template(),
        rigid_forest: qd.template(),
        fem_preconditioner: qd.template(),
        has_rigid: qd.template(),
        has_rigid_forest: qd.template(),
        has_fem: qd.template(),
        max_iterations,
    ):
        self._apply_operator(
            linear_system,
            partitioner,
            rigid,
            rigid_forest,
            has_rigid,
            has_rigid_forest,
        )
        self._segmented_dot(
            self.direction,
            self.operator_direction,
            self.pAp,
            partitioner,
        )

        qd.loop_config(name="masked_pcg_compute_alpha")
        for component in range(partitioner.K[()]):
            if self.converged[component] != 0:
                self.alpha[component] = qd.f64(0.0)
            elif self.pAp[component] > 0.0:
                self.alpha[component] = self.rz[component] / self.pAp[component]
            else:
                self.alpha[component] = qd.f64(0.0)
                qd.atomic_or(self.is_failed[()], qd.i32(1))

        qd.loop_config(name="masked_pcg_stop_on_breakdown")
        for _ in range(1):
            if self.is_failed[()] != 0:
                self.condition[()] = 0

        for row in range(self.n_block_rows_host):
            component = partitioner.comp_label[row]
            if self.is_failed[()] == 0 and self.converged[component] == 0:
                offset = row * 3
                step = self.alpha[component]
                for axis in qd.static(range(3)):
                    dof = offset + axis
                    linear_system.x_sol[dof] = linear_system.x_sol[dof] + step * self.direction[dof]
                    self.residual[dof] = self.residual[dof] - step * self.operator_direction[dof]

        self._apply_preconditioner(
            partitioner,
            rigid,
            rigid_forest,
            fem_preconditioner,
            has_rigid,
            has_rigid_forest,
            has_fem,
        )
        self._segmented_dot(
            self.residual,
            self.preconditioned_residual,
            self.rz_new,
            partitioner,
        )

        qd.loop_config(name="masked_pcg_compute_beta")
        for component in range(partitioner.K[()]):
            self.beta[component] = qd.f64(0.0)
            if self.is_failed[()] == 0 and self.converged[component] == 0:
                old_rz = self.rz[component]
                if qd.abs(old_rz) > 0.0:
                    self.beta[component] = self.rz_new[component] / old_rz

        qd.loop_config(name="masked_pcg_update_direction")
        for row in range(self.n_block_rows_host):
            component = partitioner.comp_label[row]
            if self.is_failed[()] == 0 and self.converged[component] == 0:
                offset = row * 3
                scale = self.beta[component]
                for axis in qd.static(range(3)):
                    dof = offset + axis
                    self.direction[dof] = self.preconditioned_residual[dof] + scale * self.direction[dof]

        qd.loop_config(name="masked_pcg_reset_condition")
        for _ in range(1):
            self.condition[()] = 0

        qd.loop_config(name="masked_pcg_convergence")
        for component in range(partitioner.K[()]):
            if self.is_failed[()] == 0 and self.converged[component] == 0:
                self.component_iterations[component] = self.component_iterations[component] + 1
                self.rz[component] = self.rz_new[component]
                if qd.abs(self.rz[component]) <= self.tolerance[component]:
                    self.converged[component] = 1
                else:
                    qd.atomic_or(self.condition[()], qd.i32(1))

        qd.loop_config(name="masked_pcg_finish_iteration")
        for _ in range(1):
            if self.is_failed[()] != 0:
                self.condition[()] = 0
            else:
                self.n_iterations[()] = self.n_iterations[()] + 1
                if self.condition[()] != 0 and self.n_iterations[()] >= max_iterations:
                    self.is_failed[()] = 1
                    self.condition[()] = 0
