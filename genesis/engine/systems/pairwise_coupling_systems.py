"""Systems composing the host fallback schedule of PairwiseCoupler."""

from typing import TYPE_CHECKING

import numpy as np

import genesis as gs
from genesis.engine import fem_runtime
from genesis.engine.coupling_data import (
    get_mpm_rigid_coupling_data,
    get_pbd_rigid_coupling_data,
    get_sph_rigid_coupling_data,
)
from genesis.engine.core import Find, Require, System, register_system
from genesis.engine.fem_pairwise_coupling_runtime import kernel_fem_pairwise_couple
from genesis.engine.pairwise_coupling_runtime import (
    kernel_attach_pbd_to_rigid_link,
    kernel_clear_pbd_rigid_attachment,
    kernel_mpm_grid_begin,
    kernel_mpm_grid_end,
    kernel_mpm_pbd_couple,
    kernel_mpm_rigid_couple,
    kernel_mpm_sph_couple,
    kernel_mpm_surface_to_particle,
    kernel_mpm_tool_couple,
    kernel_pbd_rigid_couple,
    kernel_reset_rigid_normals_field,
    kernel_reset_rigid_normals_ndarray,
    kernel_solve_pbd_rigid_attachment,
    kernel_sph_rigid_couple,
)
from genesis.options.solvers import LegacyCouplerOptions

from .coupler_system import CouplerSystem
from .fem_solver_system import FEMSolverSystem
from .mpm_solver_system import MPMSolverSystem
from .pbd_solver_system import PBDSolverSystem
from .rigid_sdf_system import RigidSdfSystem
from .rigid_solver_system import RigidSolverSystem
from .simulator_engine import SimulatorEngine
from .sph_solver_system import SPHSolverSystem
from .tool_solver_system import ToolSolverSystem

if TYPE_CHECKING:
    from genesis.engine.scene import Scene

CLAMPED_INV_DT = 50.0


def _pairwise_options(scene: "Scene"):
    options = scene.options.coupler
    return options if isinstance(options, LegacyCouplerOptions) else None


class MPMRigidCouplingSystem(System):
    """MPM-Rigid grid collision and optional CPIC preprocessing."""

    manager = Require(CouplerSystem)
    mpm = Require(MPMSolverSystem)
    rigid = Require(RigidSolverSystem)
    rigid_sdf = Require(RigidSdfSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.data = None

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="host")
    def preprocess(self):
        return self.run_preprocess

    @System.action(kind="host")
    def reset(self):
        return self.run_reset

    def initialize_data(self) -> None:
        self.rigid_sdf.activate()
        if self.mpm._enable_CPIC:
            self.data = get_mpm_rigid_coupling_data(self.mpm, self.rigid)
            self.cpic_flag = self.data.cpic_flag
            self.rigid_normal = self.data.rigid_normal

    def build(self) -> None:
        if self.mpm.solver.enable_CPIC:
            self.manager.on_pairwise_mpm_rigid_preprocess(action=self.preprocess)
            self.manager.on_pairwise_reset(action=self.reset)
        self.manager.on_pairwise_mpm_rigid_couple(action=self.couple)
        self.manager.on_pairwise_mpm_rigid_grad(action=self.couple_grad)

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    @System.action(kind="host")
    def couple_grad(self):
        return self.run_couple_grad

    def run_preprocess(self, f: int) -> None:
        kernel_mpm_surface_to_particle(
            f,
            self.mpm,
            self.data,
            self.rigid.n_geoms,
            self.rigid.dyn_state.geoms,
            self.rigid.dyn_info.geoms,
            self.rigid.rigid_info,
            self.rigid_sdf.sdf._sdf_info,
            self.rigid.collider_data.collider_config,
        )

    def run_couple(self, f: int) -> None:
        kernel_mpm_rigid_couple(
            f,
            self.mpm,
            self.rigid.dyn_state.geoms,
            self.rigid.dyn_info.geoms,
            self.rigid.dyn_state.links,
            self.rigid.rigid_info,
            self.rigid_sdf.sdf._sdf_info,
            self.rigid.collider_data.collider_config,
            self.rigid.n_geoms,
        )

    def run_couple_grad(self, f: int) -> None:
        kernel_mpm_rigid_couple.grad(
            f,
            self.mpm,
            self.rigid.dyn_state.geoms,
            self.rigid.dyn_info.geoms,
            self.rigid.dyn_state.links,
            self.rigid.rigid_info,
            self.rigid_sdf.sdf._sdf_info,
            self.rigid.collider_data.collider_config,
            self.rigid.n_geoms,
        )

    def run_reset(self, envs_idx) -> None:
        if envs_idx is None:
            self.data.rigid_normal.fill(0)
            return
        kernel = kernel_reset_rigid_normals_ndarray if gs.use_ndarray else kernel_reset_rigid_normals_field
        kernel(self.data.rigid_normal, self.mpm._n_particles, self.rigid.n_geoms, envs_idx)


@register_system(SimulatorEngine)
def create_mpm_rigid_coupling_system(scene: "Scene") -> MPMRigidCouplingSystem | None:
    options = _pairwise_options(scene)
    if (
        options is not None
        and scene.sim.mpm_solver.is_active
        and scene.sim.rigid_solver.is_active
        and options.rigid_mpm
    ):
        return MPMRigidCouplingSystem(scene)
    return None


class MPMGridCouplingSystem(System):
    """Begin and end stages surrounding pairwise MPM grid coupling."""

    manager = Require(CouplerSystem)
    mpm = Require(MPMSolverSystem)

    @System.action(kind="host")
    def begin(self):
        return self.run_begin

    @System.action(kind="host")
    def end(self):
        return self.run_end

    @System.action(kind="host")
    def begin_grad(self):
        return self.run_begin_grad

    @System.action(kind="host")
    def end_grad(self):
        return self.run_end_grad

    def build(self) -> None:
        self.manager.on_pairwise_mpm_grid_begin(action=self.begin)
        self.manager.on_pairwise_mpm_grid_end(action=self.end)
        self.manager.on_pairwise_mpm_grid_begin_grad(action=self.begin_grad)
        self.manager.on_pairwise_mpm_grid_end_grad(action=self.end_grad)

    def run_begin(self, f: int) -> None:
        kernel_mpm_grid_begin(f, self._scene.sim.cur_t, self.mpm)

    def run_end(self, f: int) -> None:
        kernel_mpm_grid_end(f, self.mpm)

    def run_begin_grad(self, f: int) -> None:
        kernel_mpm_grid_begin.grad(f, self._scene.sim.cur_t, self.mpm)

    def run_end_grad(self, f: int) -> None:
        kernel_mpm_grid_end.grad(f, self.mpm)


@register_system(SimulatorEngine)
def create_mpm_grid_coupling_system(scene: "Scene") -> MPMGridCouplingSystem | None:
    return (
        MPMGridCouplingSystem(scene)
        if _pairwise_options(scene) is not None and scene.sim.mpm_solver.is_active
        else None
    )


class MPMToolCouplingSystem(System):
    """Pairwise MPM-Tool collision over the MPM grid."""

    manager = Require(CouplerSystem)
    mpm = Require(MPMSolverSystem)
    tool = Require(ToolSolverSystem)

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    @System.action(kind="host")
    def couple_grad(self):
        return self.run_couple_grad

    def build(self) -> None:
        self.manager.on_pairwise_mpm_tool_couple(action=self.couple)
        self.manager.on_pairwise_mpm_tool_grad(action=self.couple_grad)

    def run_couple(self, f: int) -> None:
        kernel_mpm_tool_couple(f, self.mpm, self.tool)

    def run_couple_grad(self, f: int) -> None:
        kernel_mpm_tool_couple.grad(f, self.mpm, self.tool)


@register_system(SimulatorEngine)
def create_mpm_tool_coupling_system(scene: "Scene") -> MPMToolCouplingSystem | None:
    if _pairwise_options(scene) is not None and scene.sim.mpm_solver.is_active and scene.sim.tool_solver.is_active:
        return MPMToolCouplingSystem(scene)
    return None


class MPMSPHCouplingSystem(System):
    """Pairwise momentum exchange between the MPM grid and SPH particles."""

    manager = Require(CouplerSystem)
    mpm = Require(MPMSolverSystem)
    sph = Require(SPHSolverSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.stencil_size = int(np.floor(scene.sim.mpm_solver.dx / scene.sim.sph_solver.hash_grid_cell_size) + 2)

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    @System.action(kind="host")
    def couple_grad(self):
        return self.run_couple_grad

    def build(self) -> None:
        self.manager.on_pairwise_mpm_sph_couple(action=self.couple)
        self.manager.on_pairwise_mpm_sph_grad(action=self.couple_grad)

    def run_couple(self, f: int) -> None:
        kernel_mpm_sph_couple(f, self.mpm, self.sph, self.stencil_size)

    def run_couple_grad(self, f: int) -> None:
        kernel_mpm_sph_couple.grad(f, self.mpm, self.sph, self.stencil_size)


@register_system(SimulatorEngine)
def create_mpm_sph_coupling_system(scene: "Scene") -> MPMSPHCouplingSystem | None:
    options = _pairwise_options(scene)
    if options is not None and scene.sim.mpm_solver.is_active and scene.sim.sph_solver.is_active and options.mpm_sph:
        return MPMSPHCouplingSystem(scene)
    return None


class MPMPBDCouplingSystem(System):
    """Pairwise momentum exchange between the MPM grid and PBD particles."""

    manager = Require(CouplerSystem)
    mpm = Require(MPMSolverSystem)
    pbd = Require(PBDSolverSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.stencil_size = int(np.floor(scene.sim.mpm_solver.dx / scene.sim.pbd_solver.hash_grid_cell_size) + 2)

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    @System.action(kind="host")
    def couple_grad(self):
        return self.run_couple_grad

    def build(self) -> None:
        self.manager.on_pairwise_mpm_pbd_couple(action=self.couple)
        self.manager.on_pairwise_mpm_pbd_grad(action=self.couple_grad)

    def run_couple(self, f: int) -> None:
        kernel_mpm_pbd_couple(f, self.mpm, self.pbd, self.stencil_size)

    def run_couple_grad(self, f: int) -> None:
        kernel_mpm_pbd_couple.grad(f, self.mpm, self.pbd, self.stencil_size)


@register_system(SimulatorEngine)
def create_mpm_pbd_coupling_system(scene: "Scene") -> MPMPBDCouplingSystem | None:
    options = _pairwise_options(scene)
    if options is not None and scene.sim.mpm_solver.is_active and scene.sim.pbd_solver.is_active and options.mpm_pbd:
        return MPMPBDCouplingSystem(scene)
    return None


class SPHRigidCouplingSystem(System):
    """Two-way Pairwise coupling between SPH particles and rigid geoms."""

    manager = Require(CouplerSystem)
    sph = Require(SPHSolverSystem)
    rigid = Require(RigidSolverSystem)
    rigid_sdf = Require(RigidSdfSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    @System.action(kind="host")
    def reset(self):
        return self.run_reset

    def initialize_data(self) -> None:
        self.rigid_sdf.activate()
        self.data = get_sph_rigid_coupling_data(self.sph, self.rigid)
        self.rigid_normal = self.data.rigid_normal
        self.rigid_normal_reordered = self.data.rigid_normal_reordered

    def build(self) -> None:
        self.manager.on_pairwise_sph_rigid_couple(action=self.couple)
        self.manager.on_pairwise_reset(action=self.reset)

    def run_couple(self, f: int) -> None:
        kernel_sph_rigid_couple(
            f,
            self.sph,
            self,
            self.rigid.n_geoms,
            self.rigid._options.batch_links_info,
            self.rigid.dyn_state.geoms,
            self.rigid.dyn_info.geoms,
            self.rigid.dyn_state.links,
            self.rigid.dyn_info.links,
            self.rigid.rigid_info,
            self.rigid_sdf.sdf._sdf_info,
            self.rigid.collider_data.collider_config,
        )

    def run_reset(self, envs_idx) -> None:
        if envs_idx is None:
            self.data.rigid_normal.fill(0)
            return
        kernel = kernel_reset_rigid_normals_ndarray if gs.use_ndarray else kernel_reset_rigid_normals_field
        kernel(self.data.rigid_normal, self.sph._n_particles, self.rigid.n_geoms, envs_idx)


@register_system(SimulatorEngine)
def create_sph_rigid_coupling_system(scene: "Scene") -> SPHRigidCouplingSystem | None:
    options = _pairwise_options(scene)
    if (
        options is not None
        and scene.sim.sph_solver.is_active
        and scene.sim.rigid_solver.is_active
        and options.rigid_sph
    ):
        return SPHRigidCouplingSystem(scene)
    return None


class PBDRigidCouplingSystem(System):
    """Pairwise collision and link animation between PBD particles and rigid geoms."""

    manager = Require(CouplerSystem)
    pbd = Require(PBDSolverSystem)
    rigid = Require(RigidSolverSystem)
    rigid_sdf = Require(RigidSdfSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    def initialize_data(self) -> None:
        self.rigid_sdf.activate()
        self.data = get_pbd_rigid_coupling_data(self.pbd, self.rigid)
        self.rigid_normal_reordered = self.data.rigid_normal_reordered
        self.attachment = self.data.attachment

    def build(self) -> None:
        self.manager.on_pairwise_pbd_rigid_couple(action=self.couple)

    def run_couple(self, f: int) -> None:
        kernel_pbd_rigid_couple(
            self.pbd,
            self,
            self.rigid.n_geoms,
            self.rigid._substep_dt,
            self.rigid.dyn_state.geoms,
            self.rigid.dyn_info.geoms,
            self.rigid.dyn_state.links,
            self.rigid_sdf.sdf._sdf_info,
            self.rigid.rigid_info,
            self.rigid.collider_data.collider_config,
        )
        clamped_inv_dt = min(1.0 / self.pbd._substep_dt, CLAMPED_INV_DT)
        kernel_solve_pbd_rigid_attachment(self.pbd, self, clamped_inv_dt, self.rigid.dyn_state.links)

    def attach(self, particles_idx, envs_idx, link_idx: int, links_state=None) -> None:
        kernel_attach_pbd_to_rigid_link(
            self.pbd,
            self,
            particles_idx,
            envs_idx,
            link_idx,
            self.rigid.dyn_state.links if links_state is None else links_state,
        )

    def detach(self, particles_idx, envs_idx) -> None:
        kernel_clear_pbd_rigid_attachment(
            self.pbd,
            self,
            particles_idx,
            envs_idx,
        )


@register_system(SimulatorEngine)
def create_pbd_rigid_coupling_system(scene: "Scene") -> PBDRigidCouplingSystem | None:
    options = _pairwise_options(scene)
    if (
        options is not None
        and scene.sim.pbd_solver.is_active
        and scene.sim.rigid_solver.is_active
        and options.rigid_pbd
    ):
        return PBDRigidCouplingSystem(scene)
    return None


class FEMPairwiseCouplingSystem(System):
    """Pairwise FEM interactions with rigid, MPM and SPH participants plus the FEM floor boundary."""

    manager = Require(CouplerSystem)
    fem = Require(FEMSolverSystem)
    rigid = Require(RigidSolverSystem)
    rigid_sdf = Require(RigidSdfSystem)
    mpm = Find(MPMSolverSystem)
    sph = Find(SPHSolverSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="host")
    def couple(self):
        return self.run_couple

    @System.action(kind="host")
    def couple_grad(self):
        return self.run_couple_grad

    def build(self) -> None:
        options = self.manager.pairwise_options
        self.rigid_fem = self.rigid.is_active and options.rigid_fem
        self.fem_mpm = self.mpm is not None and options.fem_mpm
        self.fem_sph = self.sph is not None and options.fem_sph
        self.manager.on_pairwise_fem_couple(action=self.couple)
        self.manager.on_pairwise_fem_grad(action=self.couple_grad)

    def initialize_data(self) -> None:
        if self.rigid_fem:
            self.rigid_sdf.activate()

    def run_couple(self, f: int) -> None:
        kernel_fem_pairwise_couple(
            f,
            self.rigid.dyn_state.geoms,
            self.rigid.dyn_info.geoms,
            self.rigid.dyn_state.links,
            self.rigid.rigid_info,
            self.rigid_sdf.sdf._sdf_info,
            self.fem,
            self.mpm,
            self.sph,
            self.rigid_fem,
            self.fem_mpm,
            self.fem_sph,
            self.rigid.n_geoms,
            self.rigid.collider_data.collider_config,
        )
        if self.fem.solver._constraints_initialized and self.rigid.is_active:
            fem_runtime._kernel_update_linked_vertex_constraints(self.fem, self.rigid.dyn_state.links)

    def run_couple_grad(self, f: int) -> None:
        kernel_fem_pairwise_couple.grad(
            f,
            self.rigid.dyn_state.geoms,
            self.rigid.dyn_info.geoms,
            self.rigid.dyn_state.links,
            self.rigid.rigid_info,
            self.rigid_sdf.sdf._sdf_info,
            self.fem,
            self.mpm,
            self.sph,
            self.rigid_fem,
            self.fem_mpm,
            self.fem_sph,
            self.rigid.n_geoms,
            self.rigid.collider_data.collider_config,
        )


@register_system(SimulatorEngine)
def create_fem_pairwise_coupling_system(scene: "Scene") -> FEMPairwiseCouplingSystem | None:
    return (
        FEMPairwiseCouplingSystem(scene)
        if _pairwise_options(scene) is not None and scene.sim.fem_solver.is_active
        else None
    )
