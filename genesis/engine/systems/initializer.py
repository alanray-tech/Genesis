"""Manager System that initializes the data of the Systems once the solvers have built their entities."""

from typing import TYPE_CHECKING

from genesis.engine.core import Find, Require, System, register_system

from .constraint_solver_system import ConstraintSolverSystem
from .fem_solver_system import FEMSolverSystem
from .kinematic_solver_system import KinematicSolverSystem
from .pairwise_coupling_systems import (
    FEMPairwiseCouplingSystem,
    MPMRigidCouplingSystem,
    PBDRigidCouplingSystem,
    SPHRigidCouplingSystem,
)
from .mpm_solver_system import MPMSolverSystem
from .pbd_solver_system import PBDSolverSystem
from .rigid_rigid_contact_system import RigidRigidContactSystem
from .rigid_sdf_system import RigidSdfSystem
from .rigid_solver_system import RigidSolverSystem
from .sf_solver_system import SFSolverSystem
from .simulator_engine import SimulatorEngine
from .sph_solver_system import SPHSolverSystem
from .tool_solver_system import ToolSolverSystem

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class Initializer(System):
    """Manager of the initialization, which invokes the initialize Action of every System in the order initialize
    states, once the Engine is built and before the coupler reads the data of the solvers."""

    kinematic = Require(KinematicSolverSystem)
    rigid_sdf = Require(RigidSdfSystem)
    rigid = Require(RigidSolverSystem)
    rigid_rigid_contact = Find(RigidRigidContactSystem)
    constraint = Find(ConstraintSolverSystem)
    mpm = Find(MPMSolverSystem)
    pbd = Find(PBDSolverSystem)
    fem = Find(FEMSolverSystem)
    sf = Find(SFSolverSystem)
    sph = Find(SPHSolverSystem)
    tool = Find(ToolSolverSystem)
    mpm_rigid_coupling = Find(MPMRigidCouplingSystem)
    sph_rigid_coupling = Find(SPHRigidCouplingSystem)
    pbd_rigid_coupling = Find(PBDRigidCouplingSystem)
    fem_pairwise_coupling = Find(FEMPairwiseCouplingSystem)

    def initialize(self) -> None:
        """Initialize the data of every System."""
        # The kinematic data, the stable fluid grid and the tool data depend on no other System
        self.kinematic.initialize.invoke()
        if self.sf is not None:
            self.sf.initialize.invoke()
        if self.sph is not None:
            self.sph.initialize.invoke()
        if self.pbd is not None:
            self.pbd.initialize.invoke()
        if self.fem is not None:
            self.fem.initialize.invoke()
        if self.tool is not None:
            self.tool.initialize.invoke()
        # The signed distance fields of the rigid geoms are sized from the rigid model, and the collision detection
        # that the rigid data initialization constructs embeds them, so they are allocated in between
        self.rigid.initialize_model.invoke()
        self.rigid_sdf.initialize.invoke()
        self.rigid.initialize.invoke()
        # The rigid-rigid contact and the constraint solve record the configuration of the collider and of the
        # constraint solver that the rigid initialization constructs
        if self.rigid_rigid_contact is not None:
            self.rigid_rigid_contact.initialize.invoke()
        if self.constraint is not None:
            self.constraint.initialize.invoke()
        # The MPM solver fills the signed distance fields against the static configuration of the rigid data
        if self.mpm is not None:
            self.mpm.initialize.invoke()
        # Pairwise coupling data is sized from the fully initialized solver Systems.
        if self.mpm_rigid_coupling is not None:
            self.mpm_rigid_coupling.initialize.invoke()
        if self.sph_rigid_coupling is not None:
            self.sph_rigid_coupling.initialize.invoke()
        if self.pbd_rigid_coupling is not None:
            self.pbd_rigid_coupling.initialize.invoke()
        if self.fem_pairwise_coupling is not None:
            self.fem_pairwise_coupling.initialize.invoke()


@register_system(SimulatorEngine)
def create_initializer(scene: "Scene") -> Initializer:
    """Create the manager of the initialization, which every scene has."""
    return Initializer(scene)
