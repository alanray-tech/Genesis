from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from genesis.engine.solvers.rigid.rigid_solver import RigidSolver

from .consistent_ipc_contact import ConsistentIPCContactConstitution
from .contact import CONTACT_CONFIG_DEFAULTS, ContactTabular
from .contact_system import ContactSystem
from .finite_element import (
    FEMBDF1,
    FEMDiagPreconditioner,
    FiniteElement,
    FiniteElementMethod,
    QuadraticBending,
    StrainLimitBaraffWitkinShell2D,
)
from .global_body_manager import GlobalBodyManager
from .global_linear_system import GlobalLinearSystem
from .global_surface_manager import GlobalSurfaceManager
from .global_vertex_manager import GlobalVertexManager
from .lbvh_broad_phase import InfoLBVHBatchedBroadPhaseDop14, LBVHBroadPhase
from .rigid_system import RigidSystem
from .sim_engine import SimEngine
from .standard_pcg_solver import StandardPCGSolver


def build_rigid_engine(rigid_solver: RigidSolver) -> SimEngine:
    """Build the retained Genesis Rigid adapter in the CGQ engine lifecycle."""
    engine = SimEngine()
    engine.add_system(GlobalLinearSystem())
    engine.add_system(StandardPCGSolver())
    engine.add_system(RigidSystem(rigid_solver))
    engine.build_systems()
    engine.wire_solver_params(
        dt=rigid_solver._substep_dt,
        tol=rigid_solver._options.tolerance,
        max_newton_iter=rigid_solver._options.iterations,
        max_pcg_iter=rigid_solver._options.iterations,
        max_ls_iter=rigid_solver._options.ls_iterations,
        pcg_tol_rate=rigid_solver._options.tolerance,
    )
    engine.init()
    return engine


def build_scene_engine(
    scene,
    *,
    contact_config: Mapping[str, object] | None = None,
    contact_tabular: ContactTabular | None = None,
    halfplanes: tuple[np.ndarray, np.ndarray] | None = None,
) -> SimEngine:
    """Build the graph-native Rigid + QCloth CGQ system set."""
    finite_element = FiniteElement()
    has_fem = finite_element.init(scene)
    contact_requested = contact_config is not None
    resolved_contact_config = dict(CONTACT_CONFIG_DEFAULTS)
    if contact_config is not None:
        unknown = set(contact_config) - set(CONTACT_CONFIG_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown CGQ contact config keys: {sorted(unknown)}")
        resolved_contact_config.update(contact_config)
    enable_contact = contact_requested and bool(resolved_contact_config["contact/enable"])
    if enable_contact and not has_fem:
        raise RuntimeError("The current ContactSystem milestone requires FiniteElementMethod")

    engine = SimEngine()
    engine.add_system(GlobalLinearSystem())
    engine.add_system(StandardPCGSolver())
    if scene.rigid_solver.is_active:
        engine.add_system(RigidSystem(scene.rigid_solver))

    fem = None
    bdf1 = None
    membrane = None
    bending = None
    fem_preconditioner = None
    if has_fem:
        global_body_manager = GlobalBodyManager()
        global_vertex_manager = GlobalVertexManager()
        global_surface_manager = GlobalSurfaceManager()
        fem = FiniteElementMethod()
        bdf1 = FEMBDF1()
        membrane = StrainLimitBaraffWitkinShell2D()
        bending = QuadraticBending()
        fem_preconditioner = FEMDiagPreconditioner()
        for system in (
            global_body_manager,
            global_vertex_manager,
            global_surface_manager,
            fem,
            bdf1,
            membrane,
            bending,
            fem_preconditioner,
        ):
            engine.add_system(system)
        if enable_contact:
            contact_system = ContactSystem()
            bvh_type = str(resolved_contact_config["bvh/type"])
            if bvh_type == "info_lbvh_batched_dop14":
                broad_phase_system = InfoLBVHBatchedBroadPhaseDop14(
                    ee_query=str(resolved_contact_config["bvh/ee_query"]),
                    dual_frontier_levels=int(resolved_contact_config["bvh/dual/frontier_levels"]),
                    dual_target_waves=float(resolved_contact_config["bvh/dual/target_waves"]),
                    dual_max_levels=int(resolved_contact_config["bvh/dual/max_levels"]),
                )
            elif bvh_type in ("lbvh", "info_lbvh", "info_lbvh_batched"):
                broad_phase_system = LBVHBroadPhase(
                    bound_type="aabb",
                    ee_query=str(resolved_contact_config["bvh/ee_query"]),
                    dual_frontier_levels=int(resolved_contact_config["bvh/dual/frontier_levels"]),
                    dual_target_waves=float(resolved_contact_config["bvh/dual/target_waves"]),
                    dual_max_levels=int(resolved_contact_config["bvh/dual/max_levels"]),
                )
            else:
                raise NotImplementedError(f"Unsupported CGQ bvh/type {bvh_type!r}")
            contact_constitution = ConsistentIPCContactConstitution()
            for system in (contact_system, broad_phase_system, contact_constitution):
                engine.add_system(system)

    engine.build_systems()

    if has_fem:
        fem.wire_data(finite_element)
        fem.receive_global_vertex_range(0, finite_element.n_verts)
        fem.receive_global_body_range(0, finite_element.n_bodies)

        global_vertex_manager.init(finite_element.n_verts)
        global_vertex_manager.wire_thickness_data(finite_element.thicknesses)
        global_vertex_manager.wire_d_hat_data(
            np.full(
                finite_element.n_verts,
                resolved_contact_config["contact/d_hat"],
                dtype=np.float64,
            )
        )
        global_vertex_manager.wire_is_fixed_data(finite_element.is_fixed)

        global_body_manager.init(finite_element.n_bodies)
        global_body_manager.wire_body_contact_ignorance(
            finite_element.body_contact_ignorance_ranges,
            finite_element.body_contact_ignorance_body_ids,
        )

        global_surface_manager.wire_surface_data(
            finite_element.surf_triangles,
            finite_element.surf_edges,
            finite_element.surf_verts,
        )
        global_surface_manager.wire_vert_dimensions(finite_element.vert_dimensions)
        global_surface_manager.wire_area_weights(
            finite_element.vert_area_weights,
            finite_element.edge_area_weights,
            finite_element.face_area_weights,
        )
        bdf1.wire_data(finite_element.n_verts)
        membrane.wire_data(
            tri_indices=np.arange(finite_element.n_tris, dtype=np.int32),
            mu=finite_element.membrane_mu,
            lambda_param=finite_element.membrane_lambda,
            strain_limit_multiplier=finite_element.strain_limit_multiplier,
        )
        bending.wire_data(
            hinge_indices=finite_element.hinge_indices,
            bending_stiffness=finite_element.hinge_stiffness,
            Q0=finite_element.hinge_Q0,
            vert_bend_k=finite_element.vert_bend_k,
        )

        if enable_contact:
            table = contact_tabular if contact_tabular is not None else ContactTabular()
            default_model = table.at(0, 0)
            constitution = resolved_contact_config["contact/constitution"]
            if constitution == "auto":
                constitution = "consistent_ipc"
            if constitution != "consistent_ipc":
                raise NotImplementedError(
                    f"The cloth contact milestone implements only consistent_ipc, got {constitution!r}"
                )
            contact_system.wire_params(
                d_hat=float(resolved_contact_config["contact/d_hat"]),
                kappa=default_model.resistance,
                init_pair_capacity=int(resolved_contact_config["contact/init_collision_pair_capacity"]),
            )
            contact_system.set_dt_sq(scene.sim.substep_dt * scene.sim.substep_dt)
            contact_system.wire_friction_params(
                mu=default_model.friction_rate,
                eps_v=float(resolved_contact_config["friction/eps_v"]),
            )
            contact_system.wire_contact_tabular(table)
            if halfplanes is None:
                halfplane_positions = np.empty((0, 3), dtype=np.float64)
                halfplane_normals = np.empty((0, 3), dtype=np.float64)
            else:
                halfplane_positions, halfplane_normals = halfplanes
            contact_system.wire_halfplanes(halfplane_positions, halfplane_normals)
            contact_system.set_adaptive_kappa(
                str(resolved_contact_config["contact/adaptive_kappa_mode"]),
                str(resolved_contact_config["contact/adaptive_kappa_tick"]),
                finite_element.n_bodies,
            )
            contact_system.init(finite_element.n_verts)
            broad_phase_system.init_bvh(
                finite_element.n_tris,
                len(finite_element.surf_edges),
                0,
            )

    engine.wire_solver_params(
        dt=scene.sim.substep_dt,
        tol=5e-2 if has_fem else scene.rigid_solver._options.tolerance,
        max_newton_iter=1024 if has_fem else scene.rigid_solver._options.iterations,
        max_pcg_iter=1024 if has_fem else scene.rigid_solver._options.iterations,
        max_ls_iter=12 if has_fem else scene.rigid_solver._options.ls_iterations,
        pcg_tol_rate=1e-4 if has_fem else scene.rigid_solver._options.tolerance,
    )
    engine.init()
    return engine
