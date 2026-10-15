"""System of the tool solver, which owns every entity's pose, mesh data, initialization and substep."""

from typing import TYPE_CHECKING

from genesis.engine.core import System, register_system
from genesis.engine.entities.tool_entity.data import (
    ToolSolverConfig,
    ToolSolverData,
    get_tool_entity_data,
    get_tool_mesh_config,
    get_tool_solver_info,
)
from genesis.engine.entities.tool_entity.tool_entity import (
    kernel_advect,
    kernel_set_init_state,
    kernel_update_latest_pos,
)
from genesis.engine.solvers.base_solver import adopt_model
from genesis.engine.states.entities import ToolEntityState

from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


TOOL_MODEL = ("_entities", "boundary", "_substep_dt", "substeps")


class ToolSolverSystem(System):
    """System owning the data and simulation of every tool entity of the scene."""

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.tool_solver

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    def initialize_data(self) -> None:
        """Take over the model from the solver, then allocate and initialize every tool entity."""
        adopt_model(self, TOOL_MODEL)
        self.solver._system = self
        self.info = get_tool_solver_info(self)
        self.config = ToolSolverConfig(boundary=self.boundary)
        entity_data = tuple(get_tool_entity_data(entity) for entity in self._entities)
        self.data = ToolSolverData(entity_data, tuple(get_tool_mesh_config(entity.mesh) for entity in self._entities))
        for entity, data in zip(self._entities, entity_data):
            entity._data = data
            entity.init_state = ToolEntityState(entity, 0)
            kernel_set_init_state(entity._init_pos, entity._init_quat, data)

    @System.action(kind="host")
    def process_input(self):
        return self.solver.process_input

    @System.action(kind="host")
    def pre_coupling(self):
        return self.substep_pre_coupling

    @System.action(kind="host")
    def post_coupling(self):
        return self.substep_post_coupling

    def substep_pre_coupling(self, f: int) -> None:
        """Advect every tool entity through substep f."""
        for entity_data in self.data.entities:
            kernel_advect(f, entity_data, self.info, self.config)

    def substep_post_coupling(self, f: int) -> None:
        """Publish the unbatched render position of every tool entity after substep f."""
        for entity_data in self.data.entities:
            kernel_update_latest_pos(f, entity_data)


@register_system(SimulatorEngine)
def create_tool_solver_system(scene: "Scene") -> ToolSolverSystem | None:
    """Create the System owning the tool data whenever the solver holds an entity."""
    return ToolSolverSystem(scene) if scene.sim.tool_solver.is_active else None
