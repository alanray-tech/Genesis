"""Manager System that runs the phases of the solvers and of the coupler in the order of one simulator substep."""

from typing import TYPE_CHECKING

import quadrants as qd

from genesis.engine.core import Find, Require, System, register_system
from genesis.utils.qd_ext import invoke_stage_actions

from .coupler_system import CouplerSystem
from .constraint_solver_system import ConstraintSolverSystem
from .fem_solver_system import FEMSolverSystem
from .kinematic_solver_system import KinematicSolverSystem
from .mpm_grid_lifecycle_system import MPMGridLifecycleSystem
from .mpm_pbd_coupling_system import MPMPBDCouplingSystem
from .mpm_rigid_coupling_system import MPMRigidCouplingSystem
from .mpm_solver_system import MPMSolverSystem
from .mpm_sph_coupling_system import MPMSPHCouplingSystem
from .mpm_tool_coupling_system import MPMToolCouplingSystem
from .pbd_rigid_coupling_system import PBDRigidCouplingSystem
from .pbd_solver_system import PBDSolverSystem
from .rigid_solver_system import RigidSolverSystem
from .sf_solver_system import SFSolverSystem
from .simulator_engine import SimulatorEngine
from .sph_rigid_coupling_system import SPHRigidCouplingSystem
from .sph_solver_system import SPHSolverSystem
from .tool_solver_system import ToolSolverSystem

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class Substepper(System):
    """Manager of one substep: the preprocess of the coupler, every pre-coupling phase, the coupling, then every
    post-coupling phase, each in the order substep states.

    A solver System takes part when it has entities to advance: the rigid and kinematic Systems own their data whether
    or not their solver holds an entity, the other solver Systems only exist when theirs does.
    """

    tool = Find(ToolSolverSystem)
    rigid = Require(RigidSolverSystem)
    kinematic = Require(KinematicSolverSystem)
    mpm = Find(MPMSolverSystem)
    sph = Find(SPHSolverSystem)
    pbd = Find(PBDSolverSystem)
    fem = Find(FEMSolverSystem)
    sf = Find(SFSolverSystem)
    coupler = Require(CouplerSystem)
    constraint = Find(ConstraintSolverSystem)
    mpm_sph_coupling = Find(MPMSPHCouplingSystem)
    mpm_pbd_coupling = Find(MPMPBDCouplingSystem)
    mpm_tool_coupling = Find(MPMToolCouplingSystem)
    mpm_rigid_coupling = Find(MPMRigidCouplingSystem)
    sph_rigid_coupling = Find(SPHRigidCouplingSystem)
    pbd_rigid_coupling = Find(PBDRigidCouplingSystem)
    mpm_grid_lifecycle = Find(MPMGridLifecycleSystem)

    @System.action_collection(kind="stage", call_args=("f", "t"))
    def graph_preprocess(self):
        """Coupling data required before any solver advances."""

    @System.action_collection(kind="stage", call_args=("f", "t"))
    def graph_rigid_wakeup(self):
        """Wake hibernated rigid islands targeted by coupling forces from the previous substep."""

    @System.action_collection(kind="stage", call_args=("f", "t"))
    def graph_rigid_dynamics(self):
        """Rigid forward dynamics and equality assembly."""

    @System.action_collection(kind="stage", call_args=("f", "t"))
    def graph_rigid_collision(self):
        """Rigid collision detection and inequality assembly."""

    @System.action_collection(kind="stage", call_args=("f", "t"))
    def graph_rigid_solve(self):
        """Frozen rigid constraint solve."""

    @System.action_collection(kind="stage", call_args=("f", "t"))
    def graph_rigid_post(self):
        """Rigid force resolution and integration."""

    @System.action_collection(kind="stage", call_args=("f", "t"))
    def graph_particle_pre(self):
        """Independent Tool and particle solver pre-coupling stages."""

    @System.action_collection(kind="stage", call_args=("f", "t"))
    def graph_mpm_grid_begin(self):
        """Initialize MPM grid velocity."""

    @System.action_collection(kind="stage", call_args=("f", "t"))
    def graph_mpm_tool(self):
        """MPM-Tool pair stage."""

    @System.action_collection(kind="stage", call_args=("f", "t"))
    def graph_mpm_rigid(self):
        """MPM-Rigid pair stage."""

    @System.action_collection(kind="stage", call_args=("f", "t"))
    def graph_mpm_sph(self):
        """MPM-SPH pair stage."""

    @System.action_collection(kind="stage", call_args=("f", "t"))
    def graph_mpm_pbd(self):
        """MPM-PBD pair stage."""

    @System.action_collection(kind="stage", call_args=("f", "t"))
    def graph_mpm_grid_end(self):
        """Apply the MPM grid boundary."""

    @System.action_collection(kind="stage", call_args=("f", "t"))
    def graph_parallel_couplings(self):
        """Pair stages with no producer/consumer relation and race-free shared accumulation."""

    @System.action_collection(kind="stage", call_args=("f", "t"))
    def graph_solver_post(self):
        """Independent solver post-coupling stages."""

    def build(self) -> None:
        # Resolve System admission once; absent Systems contribute no Action to the frozen device schedule.
        if self.mpm_rigid_coupling is not None and self.mpm_rigid_coupling.mpm.solver.enable_CPIC:
            self.graph_preprocess += self.mpm_rigid_coupling.graph_preprocess
        if self.rigid.is_active:
            self.graph_rigid_wakeup += self.rigid.graph_wakeup
            self.graph_rigid_dynamics += self.rigid.graph_dynamics
            self.graph_rigid_collision += self.rigid.graph_collision
            self.graph_rigid_post += self.rigid.graph_post
        if self.constraint is not None:
            self.graph_rigid_solve += self.constraint.graph_solve
        for system in (self.tool, self.mpm, self.sph, self.pbd, self.sf):
            if system is not None:
                self.graph_particle_pre += system.graph_pre_coupling
        if self.mpm_grid_lifecycle is not None:
            self.graph_mpm_grid_begin += self.mpm_grid_lifecycle.graph_begin
            self.graph_mpm_grid_end += self.mpm_grid_lifecycle.graph_end
        if self.mpm_tool_coupling is not None:
            self.graph_mpm_tool += self.mpm_tool_coupling.graph_couple
        if self.mpm_rigid_coupling is not None:
            self.graph_mpm_rigid += self.mpm_rigid_coupling.graph_couple
        if self.mpm_sph_coupling is not None:
            self.graph_mpm_sph += self.mpm_sph_coupling.graph_couple
        if self.mpm_pbd_coupling is not None:
            self.graph_mpm_pbd += self.mpm_pbd_coupling.graph_couple
        if self.sph_rigid_coupling is not None:
            self.graph_parallel_couplings += self.sph_rigid_coupling.graph_couple
        if self.pbd_rigid_coupling is not None:
            self.graph_parallel_couplings += self.pbd_rigid_coupling.graph_couple
        for system in (self.tool, self.mpm, self.sph, self.pbd):
            if system is not None:
                self.graph_solver_post += system.graph_post_coupling
        if self.kinematic.is_active:
            self.graph_solver_post += self.kinematic.graph_post_coupling

    def process_input(self, in_backward: bool) -> None:
        """Apply the external commands of one step to every solver, each of which applies its own."""
        if self.tool is not None:
            self.tool.process_input.invoke((in_backward,))
        if self.rigid.is_active:
            self.rigid.process_input.invoke((in_backward,))
        if self.kinematic.is_active:
            self.kinematic.process_input.invoke((in_backward,))
        if self.mpm is not None:
            self.mpm.process_input.invoke((in_backward,))
        if self.sph is not None:
            self.sph.process_input.invoke((in_backward,))
        if self.pbd is not None:
            self.pbd.process_input.invoke((in_backward,))
        if self.fem is not None:
            self.fem.process_input.invoke((in_backward,))
        if self.sf is not None:
            self.sf.process_input.invoke((in_backward,))

    # The manager stays on the host while implicit FEM, rigid constraint dispatch, SAP and PairwiseCoupler retain host
    # phases. Each migrated participant may still run its own device graph inside these explicit lifecycle boundaries.
    def substep(self, f: int) -> None:
        """Advance every solver by substep f. A scene simulating rigid entities alone advances through
        SimulatorEngine.rigid_substep_pipeline instead."""
        self.coupler.preprocess.invoke((f,))

        if self.tool is not None:
            self.tool.pre_coupling.invoke((f,))
        if self.rigid.is_active:
            self.rigid.pre_coupling.invoke((f,))
        if self.kinematic.is_active:
            self.kinematic.pre_coupling.invoke((f,))
        # The p2g of the MPM solver reads the rigid geoms and links that the rigid substep has just moved
        if self.mpm is not None:
            self.mpm.pre_coupling.invoke((f,))
        if self.sph is not None:
            self.sph.pre_coupling.invoke((f,))
        if self.pbd is not None:
            self.pbd.pre_coupling.invoke((f,))
        if self.fem is not None:
            self.fem.pre_coupling.invoke((f,))
        if self.sf is not None:
            self.sf.pre_coupling.invoke((f,))

        self.coupler.couple.invoke((f,))

        if self.tool is not None:
            self.tool.post_coupling.invoke((f,))
        if self.rigid.is_active:
            self.rigid.post_coupling.invoke((f,))
        if self.kinematic.is_active:
            self.kinematic.post_coupling.invoke((f,))
        # The g2p of the MPM solver reads the rigid links that the rigid post-coupling phase has completed
        if self.mpm is not None:
            self.mpm.post_coupling.invoke((f,))
        if self.sph is not None:
            self.sph.post_coupling.invoke((f,))
        if self.pbd is not None:
            self.pbd.post_coupling.invoke((f,))
        if self.fem is not None:
            self.fem.post_coupling.invoke((f,))
        if self.sf is not None:
            self.sf.post_coupling.invoke((f,))


@register_system(SimulatorEngine)
def create_substepper(scene: "Scene") -> Substepper:
    """Create the manager of the substep, which every scene stepped by the engine has."""
    return Substepper(scene)


@qd.kernel(graph=True)
def kernel_graph_substep(context: qd.template(), schedule: qd.template(), f: qd.i32, t: qd.f32):
    """One graph containing the complete frozen lifecycle schedule."""
    invoke_stage_actions(schedule.graph_preprocess, (f, t))
    invoke_stage_actions(schedule.graph_rigid_wakeup, (f, t))
    invoke_stage_actions(schedule.graph_rigid_dynamics, (f, t))
    invoke_stage_actions(schedule.graph_rigid_collision, (f, t))
    invoke_stage_actions(schedule.graph_rigid_solve, (f, t))
    invoke_stage_actions(schedule.graph_rigid_post, (f, t))
    invoke_stage_actions(schedule.graph_particle_pre, (f, t))
    invoke_stage_actions(schedule.graph_mpm_grid_begin, (f, t))
    invoke_stage_actions(schedule.graph_mpm_tool, (f, t))
    invoke_stage_actions(schedule.graph_mpm_rigid, (f, t))
    invoke_stage_actions(schedule.graph_mpm_sph, (f, t))
    invoke_stage_actions(schedule.graph_mpm_pbd, (f, t))
    # qd_ext documents why this parallel unrolling must remain syntactically in the graph kernel. MPM grid
    # finalization and default pair couplers have no cross-branch reads/writes.
    with qd.graph.parallel_context():
        for action in qd.static(schedule.graph_mpm_grid_end):
            with qd.graph.parallel():
                action.invoke((f, t))
        for action in qd.static(schedule.graph_parallel_couplings):
            with qd.graph.parallel():
                action.invoke((f, t))
    invoke_stage_actions(schedule.graph_solver_post, (f, t))


@qd.kernel(graph=True)
def kernel_graph_rigid_dynamics(context: qd.template(), schedule: qd.template(), f: qd.i32, t: qd.f32):
    invoke_stage_actions(schedule.graph_preprocess, (f, t))
    invoke_stage_actions(schedule.graph_rigid_wakeup, (f, t))
    invoke_stage_actions(schedule.graph_rigid_dynamics, (f, t))


@qd.kernel(graph=True)
def kernel_graph_rigid_collision(context: qd.template(), schedule: qd.template(), f: qd.i32, t: qd.f32):
    invoke_stage_actions(schedule.graph_rigid_collision, (f, t))


@qd.kernel(graph=True)
def kernel_graph_rigid_solve_post(context: qd.template(), schedule: qd.template(), f: qd.i32, t: qd.f32):
    invoke_stage_actions(schedule.graph_rigid_solve, (f, t))
    invoke_stage_actions(schedule.graph_rigid_post, (f, t))


@qd.kernel(graph=True)
def kernel_graph_particle_pre(context: qd.template(), schedule: qd.template(), f: qd.i32, t: qd.f32):
    invoke_stage_actions(schedule.graph_particle_pre, (f, t))


@qd.kernel(graph=True)
def kernel_graph_couple_post(context: qd.template(), schedule: qd.template(), f: qd.i32, t: qd.f32):
    invoke_stage_actions(schedule.graph_mpm_grid_begin, (f, t))
    invoke_stage_actions(schedule.graph_mpm_tool, (f, t))
    invoke_stage_actions(schedule.graph_mpm_rigid, (f, t))
    invoke_stage_actions(schedule.graph_mpm_sph, (f, t))
    invoke_stage_actions(schedule.graph_mpm_pbd, (f, t))
    with qd.graph.parallel_context():
        for action in qd.static(schedule.graph_mpm_grid_end):
            with qd.graph.parallel():
                action.invoke((f, t))
        for action in qd.static(schedule.graph_parallel_couplings):
            with qd.graph.parallel():
                action.invoke((f, t))
    invoke_stage_actions(schedule.graph_solver_post, (f, t))
