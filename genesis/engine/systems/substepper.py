"""Manager System that runs the phases of the solvers and of the coupler in the order of one simulator substep."""

from typing import TYPE_CHECKING

import quadrants as qd

from genesis.engine import mpm_graph_runtime, pbd_runtime, sph_runtime
from genesis.engine.core import Find, Require, System, register_system
from genesis.engine.pairwise_coupling_runtime import (
    func_mpm_grid_begin,
    func_mpm_grid_end,
    func_mpm_pbd_couple,
    func_mpm_sph_couple,
    func_mpm_tool_couple,
)
from genesis.engine.entities.tool_entity.tool_entity import func_advect as func_tool_advect
from genesis.engine.entities.tool_entity.tool_entity import func_update_latest_pos as func_tool_update_latest_pos
from genesis.engine.systems.sf_solver_system import (
    func_advect_and_impulse,
    func_divergence,
    func_pressure_jacobi_odd,
    func_pressure_jacobi_to_current,
    func_pressure_jacobi_to_next,
    func_projection_finish,
    func_projection_init,
    func_subtract_gradient,
)

from .coupler_system import CouplerSystem
from .fem_solver_system import FEMSolverSystem
from .kinematic_solver_system import KinematicSolverSystem
from .mpm_solver_system import MPMSolverSystem
from .pairwise_coupling_systems import MPMPBDCouplingSystem, MPMSPHCouplingSystem, MPMToolCouplingSystem
from .pbd_solver_system import PBDSolverSystem
from .rigid_solver_system import RigidSolverSystem
from .sf_solver_system import SFSolverSystem
from .simulator_engine import SimulatorEngine
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
    mpm_sph_coupling = Find(MPMSPHCouplingSystem)
    mpm_pbd_coupling = Find(MPMPBDCouplingSystem)
    mpm_tool_coupling = Find(MPMToolCouplingSystem)

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
def kernel_particle_graph_substep(
    has_sph: qd.template(),
    sph: qd.template(),
    has_pbd: qd.template(),
    pbd: qd.template(),
    has_sf: qd.template(),
    sf: qd.template(),
    has_mpm: qd.template(),
    mpm: qd.template(),
    rigid: qd.template(),
    rigid_sdf: qd.template(),
    has_mpm_sph: qd.template(),
    mpm_sph: qd.template(),
    has_mpm_pbd: qd.template(),
    mpm_pbd: qd.template(),
    has_mpm_constraints: qd.template(),
    has_tool: qd.template(),
    tool: qd.template(),
    has_mpm_tool: qd.template(),
    f: qd.i32,
    t: qd.f32,
):
    """One whole-scene graph substep for particle-only scenes."""
    if qd.static(has_tool):
        for i_e in qd.static(range(len(tool.data.entities))):
            func_tool_advect(f, tool.data.entities[i_e], tool.info, tool.config)

    if qd.static(has_mpm):
        if qd.static(mpm.material_config.needs_svd):
            mpm_graph_runtime.graph_compute_F_tmp_and_svd(mpm, f)
        else:
            mpm_graph_runtime.graph_compute_F_tmp_only(mpm, f)
        mpm_graph_runtime.graph_p2g(
            mpm,
            f,
            rigid.dyn_state.geoms,
            rigid.dyn_info.geoms,
            rigid.dyn_state.links,
            rigid.rigid_info,
            rigid_sdf.sdf._sdf_info,
            rigid.collider_data.collider_config,
            False,
            None,
            0,
        )

    if qd.static(has_sph):
        sph_runtime.func_reorder_particles(sph, f, False, None, 0)
        sph_runtime.func_compute_rho(sph, f)
        if qd.static(sph._pressure_solver == "WCSPH"):
            sph_runtime.func_compute_non_pressure_forces(sph, f, t)
            sph_runtime.func_compute_pressure_forces(sph, f)
            sph_runtime.func_advect_velocity(sph, f)
        else:
            sph_runtime.func_compute_DFSPH_factor(sph, f)
            sph_runtime.func_divergence_solve(sph, sph.graph_counter)
            sph_runtime.func_advect_velocity(sph, f)
            sph_runtime.func_compute_non_pressure_forces(sph, f, t)
            sph_runtime.func_predict_velocity(sph, f)
            sph_runtime.func_density_solve(sph, sph.graph_counter)

    if qd.static(has_pbd):
        pbd_runtime.func_store_initial_pos(pbd, f)
        pbd_runtime.func_apply_external_force(pbd, f, t)
        if qd.static(pbd._n_edges > 0):
            pbd_runtime.func_solve_stretch(pbd, f)
        if qd.static(pbd._n_inner_edges > 0):
            pbd_runtime.func_solve_bending(pbd, f)
        if qd.static(pbd._n_elems > 0):
            pbd_runtime.func_solve_volume(pbd, f)
        pbd_runtime.func_reorder_particles(pbd, f)
        pbd_runtime.func_solve_density(pbd, f)
        pbd_runtime.func_solve_viscosity(pbd, f)
        pbd_runtime.func_solve_collision(pbd, f)
        pbd_runtime.func_compute_velocity(pbd, f)

    if qd.static(has_sf):
        func_advect_and_impulse(sf.grid_state, sf.grid_info, sf.sf_config)
        func_divergence(sf.grid_state)
        func_projection_init(sf.graph_counter, sf.grid_state, sf.grid_info, sf.projection_state)
        while qd.graph.do_while(sf.graph_counter):
            func_pressure_jacobi_to_next(sf.grid_state, sf.projection_state)
            func_pressure_jacobi_to_current(sf.graph_counter, sf.grid_state, sf.projection_state)
        if qd.static(sf.solver_iters % 2 == 1):
            func_pressure_jacobi_odd(sf.grid_state, sf.projection_state)
        func_projection_finish(sf.grid_state, sf.projection_state, is_odd=sf.solver_iters % 2 == 1)
        func_subtract_gradient(sf.grid_state, sf.grid_info)

    if qd.static(has_mpm):
        func_mpm_grid_begin(f, t, mpm)
        if qd.static(has_mpm_tool):
            func_mpm_tool_couple(f, mpm, tool)
        if qd.static(has_mpm_sph):
            func_mpm_sph_couple(f, mpm, sph, mpm_sph.stencil_size)
        if qd.static(has_mpm_pbd):
            func_mpm_pbd_couple(f, mpm, pbd, mpm_pbd.stencil_size)
        func_mpm_grid_end(f, mpm)
        if qd.static(has_tool):
            for i_e in qd.static(range(len(tool.data.entities))):
                func_tool_update_latest_pos(f, tool.data.entities[i_e])
        mpm_graph_runtime.graph_g2p(
            mpm,
            f,
            rigid.dyn_info.geoms,
            rigid.dyn_state.links,
            rigid.rigid_info,
            False,
            None,
        )
        if qd.static(has_mpm_constraints):
            mpm_graph_runtime.graph_apply_particle_constraints(mpm, f, rigid.dyn_state.links)
        mpm_graph_runtime.graph_reset_dirty_cells(mpm, f)
        mpm_graph_runtime.graph_reset_dirty_count(mpm)

    if qd.static(has_tool and not has_mpm):
        for i_e in qd.static(range(len(tool.data.entities))):
            func_tool_update_latest_pos(f, tool.data.entities[i_e])

    if qd.static(has_sph):
        sph_runtime.func_advect_position(sph, f)
        sph_runtime.func_copy_from_reordered(sph, f, False, None, 0)

    if qd.static(has_pbd):
        pbd_runtime.func_copy_from_reordered(pbd, f)
        pbd_runtime.func_solve_boundary_collision(pbd, f)
