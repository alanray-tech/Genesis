from __future__ import annotations

import quadrants as qd

from .physics_system import PhysicsSystem


@qd.data_oriented
class SolvePlan:
    """Lower registered physics systems into statically dispatched graph phases."""

    def __init__(self, systems: list[PhysicsSystem]) -> None:
        if not systems:
            raise RuntimeError("SimEngine requires at least one PhysicsSystem")

        self.n_systems = len(systems)
        self.predict_functions = [system.predict for system in systems]
        self.candidate_functions = [system.assemble_candidate_rows for system in systems]
        self.initialize_functions = [system.initialize_newton for system in systems]
        self.energy_functions = [system.add_current_energy for system in systems]
        self.gradient_functions = [system.assemble_gradient for system in systems]
        self.hessian_functions = [system.apply_hessian for system in systems]
        self.preconditioner_functions = [system.apply_preconditioner for system in systems]
        self.direction_functions = [system.prepare_direction for system in systems]
        self.energy_delta_functions = [system.evaluate_energy_delta for system in systems]
        self.accept_functions = [system.accept for system in systems]
        self.active_functions = [system.set_newton_active for system in systems]
        self.build_preconditioner_functions = [system.build_preconditioner for system in systems]
        self.finalize_functions = [system.finalize for system in systems]

    @qd.func(requires_top_level=True)
    def predict(self):
        for i_system in qd.static(range(self.n_systems)):
            self.predict_functions[i_system]()

    @qd.func(requires_top_level=True)
    def assemble_candidate_rows(self):
        for i_system in qd.static(range(self.n_systems)):
            self.candidate_functions[i_system]()

    @qd.func(requires_top_level=True)
    def initialize_newton(self):
        for i_system in qd.static(range(self.n_systems)):
            self.initialize_functions[i_system]()

    @qd.func(requires_top_level=True)
    def add_current_energy(self, energy: qd.template()):
        for i_system in qd.static(range(self.n_systems)):
            self.energy_functions[i_system](energy)

    @qd.func(requires_top_level=True)
    def assemble_gradient(self, linear_system: qd.template(), gradient_squared: qd.template()):
        for i_system in qd.static(range(self.n_systems)):
            self.gradient_functions[i_system](linear_system, gradient_squared)

    @qd.func(requires_top_level=True)
    def apply_hessian(self, x: qd.template(), y: qd.template()):
        for i_system in qd.static(range(self.n_systems)):
            self.hessian_functions[i_system](x, y)

    @qd.func(requires_top_level=True)
    def apply_preconditioner(self, residual: qd.template(), result: qd.template()):
        for i_system in qd.static(range(self.n_systems)):
            self.preconditioner_functions[i_system](residual, result)

    @qd.func(requires_top_level=True)
    def prepare_direction(self, direction: qd.template()):
        for i_system in qd.static(range(self.n_systems)):
            self.direction_functions[i_system](direction)

    @qd.func(requires_top_level=True)
    def evaluate_energy_delta(self, alpha, energy_delta: qd.template()):
        for i_system in qd.static(range(self.n_systems)):
            self.energy_delta_functions[i_system](alpha, energy_delta)

    @qd.func(requires_top_level=True)
    def accept(self, alpha):
        for i_system in qd.static(range(self.n_systems)):
            self.accept_functions[i_system](alpha)

    @qd.func(requires_top_level=True)
    def set_newton_active(self, is_active):
        for i_system in qd.static(range(self.n_systems)):
            self.active_functions[i_system](is_active)

    @qd.func(requires_top_level=True)
    def build_preconditioner(self, compute_envelope: qd.template()):
        for i_system in qd.static(range(self.n_systems)):
            self.build_preconditioner_functions[i_system](compute_envelope)

    @qd.func(requires_top_level=True)
    def finalize(self):
        for i_system in qd.static(range(self.n_systems)):
            self.finalize_functions[i_system]()
