from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .sim_engine import SimEngine


T = TypeVar("T", bound="SimSystem")


class _SystemState:
    __slots__ = ("engine", "is_valid")

    def __init__(self) -> None:
        self.engine: SimEngine | None = None
        self.is_valid = True


_SYSTEM_STATE: dict[int, _SystemState] = {}


class SimSystem(ABC):
    """Provide build-time registration and dependency lookup for one numerical system."""

    def __init__(self) -> None:
        key = id(self)
        _SYSTEM_STATE[key] = _SystemState()
        weakref.finalize(self, _SYSTEM_STATE.pop, key, None)

    @property
    def _system_state(self) -> _SystemState:
        state = _SYSTEM_STATE.get(id(self))
        if state is None:
            raise RuntimeError(f"{type(self).__name__} must call SimSystem.__init__()")
        return state

    @property
    def engine(self) -> SimEngine:
        engine = self._system_state.engine
        if engine is None:
            raise RuntimeError(f"{type(self).__name__} is not registered with a SimEngine")
        return engine

    @property
    def is_valid(self) -> bool:
        return self._system_state.is_valid

    def find(self, system_type: type[T]) -> T | None:
        """Return a registered system of the requested type, if one is active."""
        return self.engine.find(system_type)

    def require(self, system_type: type[T]) -> T:
        """Return a registered system of the requested type."""
        system = self.find(system_type)
        if system is None:
            raise RuntimeError(f"Required system {system_type.__name__} is not registered")
        return system

    @abstractmethod
    def build(self) -> None:
        """Resolve dependencies and initialize device-visible runtime data."""

    def _set_engine(self, engine: SimEngine) -> None:
        self._system_state.engine = engine

    def _invalidate(self) -> None:
        self._system_state.is_valid = False
