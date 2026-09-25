from __future__ import annotations

import weakref
from typing import TypeVar

import quadrants as qd

from genesis.utils.misc import qd_to_numpy

from .global_layout import GlobalLayout
from .global_linear_system import GlobalLinearSystem
from .linear_pcg import LinearPCG
from .physics_system import PhysicsSystem
from .sim_config import SimConfig
from .sim_system import SimSystem
from .solve_plan import SolvePlan

T = TypeVar("T", bound=SimSystem)


class _EngineState:
    __slots__ = ("is_initialized", "systems")

    def __init__(self) -> None:
        self.systems: dict[type, SimSystem] = {}
        self.is_initialized = False


_ENGINE_STATE: dict[int, _EngineState] = {}


@qd.data_oriented
class SimEngine:
    """Advance registered numerical systems through one graph-native Newton timestep."""

    def __init__(self) -> None:
        key = id(self)
        _ENGINE_STATE[key] = _EngineState()
        weakref.finalize(self, _ENGINE_STATE.pop, key, None)

        self.newton_condition = qd.ndarray(qd.i32, shape=())
        self.is_newton_active = qd.ndarray(qd.i32, shape=())
        self.newton_iterations = qd.ndarray(qd.i32, shape=(1,))
        self.line_search_condition = qd.ndarray(qd.i32, shape=())
        self.is_line_search_active = qd.ndarray(qd.i32, shape=())
        self.line_search_iterations = qd.ndarray(qd.i32, shape=(1,))
        self.line_search_trial = qd.ndarray(qd.i32, shape=(1,))
        self.line_search_alpha = qd.ndarray(qd.f64, shape=(1,))
        self.is_line_search_accepted = qd.ndarray(qd.i32, shape=())
        self.is_failed = qd.ndarray(qd.i32, shape=())
        self.energy = qd.ndarray(qd.f64, shape=())
        self.energy_delta = qd.ndarray(qd.f64, shape=())
        self.gradient_squared = qd.ndarray(qd.f64, shape=())

    @property
    def _engine_state(self) -> _EngineState:
        state = _ENGINE_STATE.get(id(self))
        if state is None:
            raise RuntimeError("SimEngine.__init__() was not called")
        return state

    @property
    def n_newton_iterations(self) -> int:
        return int(qd_to_numpy(self.newton_iterations)[0])

    @property
    def n_pcg_iterations(self) -> int:
        return int(qd_to_numpy(self._pcg.n_iterations)[0])

    @property
    def n_line_search_iterations(self) -> int:
        return int(qd_to_numpy(self.line_search_iterations)[0])

    @property
    def failed(self) -> bool:
        return bool(qd_to_numpy(self.is_failed))

    def add_system(self, system: SimSystem) -> None:
        if self._engine_state.is_initialized:
            raise RuntimeError("Systems cannot be added after SimEngine initialization")
        system_type = type(system)
        if system_type in self._engine_state.systems:
            raise RuntimeError(f"System {system_type.__name__} is already registered")
        system._set_engine(self)
        self._engine_state.systems[system_type] = system
        setattr(self, system_type.__name__, system)

    def find(self, system_type: type[T]) -> T | None:
        system = self._engine_state.systems.get(system_type)
        if system is None or not system.is_valid:
            return None
        return system

    def require(self, system_type: type[T]) -> T:
        system = self.find(system_type)
        if system is None:
            raise RuntimeError(f"Required system {system_type.__name__} is not registered")
        return system

    def initialize(self) -> None:
        if self._engine_state.is_initialized:
            raise RuntimeError("SimEngine is already initialized")
        for system in self._engine_state.systems.values():
            system.build()

        self._config = self.require(SimConfig)
        self._layout = self.require(GlobalLayout)
        self._linear_system = self.require(GlobalLinearSystem)
        self._pcg = self.require(LinearPCG)
        physics_systems = [
            system for system in self._engine_state.systems.values() if isinstance(system, PhysicsSystem)
        ]
        self._solve_plan = SolvePlan(physics_systems)
        self._engine_state.is_initialized = True

    @qd.kernel(graph=True, fastcache=True)
    def _step_kernel(self):
        self._solve_plan.predict()
        self._solve_plan.assemble_candidate_rows()
        self._solve_plan.initialize_newton()
        self._linear_system.zero_assembly()

        for _ in range(1):
            self.energy[()] = qd.f64(0.0)
            self.gradient_squared[()] = qd.f64(0.0)
            self.newton_iterations[0] = 0
            self.line_search_iterations[0] = 0
            self.is_failed[()] = 0
        self._solve_plan.add_current_energy(self.energy)
        self._solve_plan.assemble_gradient(self._linear_system, self.gradient_squared)

        for _ in range(1):
            tolerance_squared = self._config.newton_tolerance * self._config.newton_tolerance
            self.newton_condition[()] = 1
            self.is_newton_active[()] = qd.i32(self.gradient_squared[()] > tolerance_squared)

        while qd.graph.do_while(self.newton_condition):
            if qd.static(self._linear_system.has_explicit_operator):
                self._linear_system.finalize_bcoo()
            self._pcg.initialize(self._linear_system, self._solve_plan)
            while qd.graph.do_while(self._pcg.condition):
                self._pcg.iteration(
                    self._linear_system,
                    self._solve_plan,
                    self._config.pcg_tolerance,
                    self._config.max_pcg,
                )

            self._solve_plan.prepare_direction(self._linear_system.solution)
            for _ in range(1):
                self.line_search_alpha[0] = qd.f64(1.0)
                self.line_search_condition[()] = 1
                self.is_line_search_active[()] = qd.i32(self.is_newton_active[()] != 0 and self._pcg.is_failed[()] == 0)
                self.is_line_search_accepted[()] = 0
                self.line_search_trial[0] = 0

            while qd.graph.do_while(self.line_search_condition):
                for _ in range(1):
                    self.energy_delta[()] = qd.f64(0.0)
                self._solve_plan.evaluate_energy_delta(self.line_search_alpha[0], self.energy_delta)
                for _ in range(1):
                    if self.is_line_search_active[()] != 0:
                        if self.energy_delta[()] <= 0.0:
                            self.is_line_search_accepted[()] = 1
                            self.is_line_search_active[()] = 0
                        else:
                            self.line_search_alpha[0] = self.line_search_alpha[0] * 0.5
                            self.line_search_trial[0] = self.line_search_trial[0] + 1
                            self.line_search_iterations[0] = self.line_search_iterations[0] + 1
                            if self.line_search_trial[0] >= self._config.max_line_search:
                                self.is_line_search_active[()] = 0
                    self.line_search_condition[()] = self.is_line_search_active[()]

            for _ in range(1):
                if self.is_line_search_accepted[()] == 0:
                    self.line_search_alpha[0] = qd.f64(0.0)
            self._solve_plan.accept(self.line_search_alpha[0])
            self._linear_system.zero_assembly()
            for _ in range(1):
                self.gradient_squared[()] = qd.f64(0.0)
                if self.is_newton_active[()] != 0 and self.is_line_search_accepted[()] != 0:
                    self.energy[()] = self.energy[()] + self.energy_delta[()]
                    self.newton_iterations[0] = self.newton_iterations[0] + 1
            self._solve_plan.assemble_gradient(self._linear_system, self.gradient_squared)
            for _ in range(1):
                tolerance_squared = self._config.newton_tolerance * self._config.newton_tolerance
                is_converged = self.gradient_squared[()] <= tolerance_squared
                is_exhausted = self.newton_iterations[0] >= self._config.max_newton
                is_rejected = self.is_newton_active[()] != 0 and self.is_line_search_accepted[()] == 0
                is_pcg_failed = self.is_newton_active[()] != 0 and self._pcg.is_failed[()] != 0
                self.is_newton_active[()] = qd.i32(not (is_converged or is_exhausted or is_rejected or is_pcg_failed))
                self.newton_condition[()] = self.is_newton_active[()]
                if (is_exhausted and not is_converged) or is_rejected or is_pcg_failed:
                    self.is_failed[()] = 1
            self._solve_plan.set_newton_active(self.is_newton_active[()])
            self._solve_plan.build_preconditioner(compute_envelope=False)

        self._solve_plan.finalize()

    def step(self) -> None:
        if not self._engine_state.is_initialized:
            raise RuntimeError("SimEngine must be initialized before stepping")
        self._step_kernel()
        if self.failed:
            raise RuntimeError("The global Rigid Newton timestep did not converge")
