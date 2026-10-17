"""Typed state owned by pairwise Legacy coupling Systems."""

import dataclasses
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.engine.core import Data
from genesis.utils.array_class import V, V_VEC, DataKind


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPMRigidCouplingData(Data):
    """CPIC cell separation and persistent particle-side rigid normals."""

    kind: ClassVar[DataKind] = DataKind.STATE

    cpic_flag: qd.Tensor
    rigid_normal: qd.Tensor


def get_mpm_rigid_coupling_data(mpm, rigid) -> MPMRigidCouplingData:
    return MPMRigidCouplingData(
        cpic_flag=V(dtype=gs.qd_int, shape=(mpm._n_particles, 3, 3, 3, mpm._B)),
        rigid_normal=V_VEC(3, dtype=gs.qd_float, shape=(mpm._n_particles, rigid.n_geoms, mpm._B)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SPHRigidCouplingData(Data):
    """Rigid contact normals in authored and spatially reordered SPH particle order."""

    kind: ClassVar[DataKind] = DataKind.STATE

    rigid_normal: qd.Tensor
    rigid_normal_reordered: qd.Tensor


def get_sph_rigid_coupling_data(sph, rigid) -> SPHRigidCouplingData:
    shape = (sph._n_particles, rigid.n_geoms, sph._B)
    return SPHRigidCouplingData(
        rigid_normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        rigid_normal_reordered=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class PBDRigidAttachmentState:
    kind: ClassVar[DataKind] = DataKind.STATE

    link_idx: qd.Tensor
    local_pos: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class PBDRigidCouplingData(Data):
    """Reordered contact normals and persistent particle-to-link attachments."""

    kind: ClassVar[DataKind] = DataKind.STATE

    rigid_normal_reordered: qd.Tensor
    attachment: PBDRigidAttachmentState


def get_pbd_rigid_coupling_data(pbd, rigid) -> PBDRigidCouplingData:
    shape = (pbd._n_particles, pbd._B)
    attachment = PBDRigidAttachmentState(
        link_idx=V(dtype=gs.qd_int, shape=shape),
        local_pos=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )
    attachment.link_idx.fill(-1)
    attachment.local_pos.fill(0.0)
    return PBDRigidCouplingData(
        rigid_normal_reordered=V_VEC(
            3,
            dtype=gs.qd_float,
            shape=(pbd._n_particles, pbd._B, rigid.n_geoms),
        ),
        attachment=attachment,
    )
