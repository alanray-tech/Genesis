from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from genesis.engine.states.cache import QueriedStates
from genesis.engine.states.entities import ToolEntityState
from genesis.utils.geom import (
    qd_rotvec_to_quat,
    qd_transform_quat_by_quat,
    transform_pos_quat_by_trans_quat,
)
from genesis.utils.misc import qd_zero_grad, to_gs_tensor

from ..base_entity import Entity
from .data import ToolEntityData, ToolSolverConfig, ToolSolverInfo
from .mesh import Mesh


class ToolEntity(Entity):
    # Mesh-based tool body entity
    def __init__(
        self,
        scene,
        idx,
        solver,
        material,
        morph,
        surface,
        name: str | None = None,
    ):
        super().__init__(idx, scene, morph, solver, material, surface, name=name)

        # The morph pose offset (e.g. an up-axis conversion) is composed onto the morph pose.
        self._init_pos, self._init_quat = transform_pos_quat_by_trans_quat(
            np.array(morph.offset_pos, dtype=gs.np_float),
            np.array(morph.offset_quat, dtype=gs.np_float),
            np.array(morph.pos, dtype=gs.np_float),
            np.array(morph.quat, dtype=gs.np_float),
        )

        self.mesh = Mesh(
            entity=self,
            material=material,
            morph=morph,
        )
        self.init_tgt_vars()
        self.init_ckpt()
        self._queried_states = QueriedStates()
        self._data: ToolEntityData | None = None

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        if isinstance(self._morph, gs.morphs.Mesh):
            return Path(self._morph.file).stem
        return "tool"

    # ------------------------------------------------------------------------------------
    # --------------------------------- initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def init_tgt_vars(self):
        # temp variable to store targets for next step
        self._tgt = {
            "pos": None,
            "quat": None,
            "vel": None,
            "ang": None,
        }

        self._tgt_buffer = {
            "pos": list(),
            "quat": list(),
            "vel": list(),
            "ang": list(),
        }

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        qd_zero_grad(self._data.state)

        self._tgt_buffer["pos"].clear()
        self._tgt_buffer["quat"].clear()
        self._tgt_buffer["vel"].clear()
        self._tgt_buffer["ang"].clear()

        self._queried_states.clear()

    def save_ckpt(self, ckpt_name):
        if self._sim.requires_grad:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = {
                    "pos": torch.zeros((self._sim._B, 3), dtype=gs.tc_float, device=gs.device),
                    "quat": torch.zeros((self._sim._B, 4), dtype=gs.tc_float, device=gs.device),
                    "vel": torch.zeros((self._sim._B, 3), dtype=gs.tc_float, device=gs.device),
                    "ang": torch.zeros((self._sim._B, 3), dtype=gs.tc_float, device=gs.device),
                    "_tgt_buffer": dict(),
                }

            kernel_save_ckpt(
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["quat"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["ang"],
                self.data,
            )

            self._ckpt[ckpt_name]["_tgt_buffer"]["pos"] = list(self._tgt_buffer["pos"])
            self._ckpt[ckpt_name]["_tgt_buffer"]["quat"] = list(self._tgt_buffer["quat"])
            self._ckpt[ckpt_name]["_tgt_buffer"]["vel"] = list(self._tgt_buffer["vel"])
            self._ckpt[ckpt_name]["_tgt_buffer"]["ang"] = list(self._tgt_buffer["ang"])

            self._tgt_buffer["pos"].clear()
            self._tgt_buffer["quat"].clear()
            self._tgt_buffer["vel"].clear()
            self._tgt_buffer["ang"].clear()

        # restart from frame 0 in memory
        kernel_copy_frame(self._sim.substeps_local, 0, self.data)

    def load_ckpt(self, ckpt_name):
        kernel_copy_frame(0, self._sim.substeps_local, self.data)
        kernel_copy_grad(0, self._sim.substeps_local, self.data)
        kernel_reset_grad_till_frame(self._sim.substeps_local, self.data)

        kernel_load_ckpt(
            self._ckpt[ckpt_name]["pos"],
            self._ckpt[ckpt_name]["quat"],
            self._ckpt[ckpt_name]["vel"],
            self._ckpt[ckpt_name]["ang"],
            self.data,
        )

        self._tgt_buffer["pos"] = list(self._ckpt[ckpt_name]["_tgt_buffer"]["pos"])
        self._tgt_buffer["quat"] = list(self._ckpt[ckpt_name]["_tgt_buffer"]["quat"])
        self._tgt_buffer["vel"] = list(self._ckpt[ckpt_name]["_tgt_buffer"]["vel"])
        self._tgt_buffer["ang"] = list(self._ckpt[ckpt_name]["_tgt_buffer"]["ang"])

    def substep_pre_coupling(self, f):
        kernel_advect(f, self.data, self.solver.info, self.solver.config)

    def substep_pre_coupling_grad(self, f):
        kernel_advect.grad(f, self.data, self.solver.info, self.solver.config)

    def substep_post_coupling(self, f):
        kernel_update_latest_pos(f, self.data)

    def substep_post_coupling_grad(self, f):
        pass

    def get_state(self, f=None):
        state = ToolEntityState(self, self._sim.cur_step_global)

        if f is None:
            f = self._sim.cur_substep_local
        kernel_get_frame(f, state.pos, state.quat, state.vel, state.ang, self.data)

        # we store all queried states to track gradient flow
        self._queried_states.append(state)

        return state

    def set_state(self, f, state):
        f = self._sim.cur_substep_local
        kernel_set_frame(f, state.pos, state.quat, state.vel, state.ang, self.data)

    def set_velocity(self, vel=None, ang=None):
        if vel is not None:
            vel = to_gs_tensor(vel)
            self._tgt["vel"] = vel

        if ang is not None:
            ang = to_gs_tensor(ang)
            self._tgt["ang"] = ang

    def set_position(self, pos):
        pos = to_gs_tensor(pos)
        self._tgt["pos"] = pos

    def set_quaternion(self, quat):
        quat = to_gs_tensor(quat)
        self._tgt["quat"] = quat

    def process_input(self, in_backward=False):
        if in_backward:
            self._tgt["pos"] = self._tgt_buffer["pos"][self._sim.cur_step_local]
            self._tgt["quat"] = self._tgt_buffer["quat"][self._sim.cur_step_local]
            self._tgt["vel"] = self._tgt_buffer["vel"][self._sim.cur_step_local]
            self._tgt["ang"] = self._tgt_buffer["ang"][self._sim.cur_step_local]
        else:
            self._tgt_buffer["pos"].append(self._tgt["pos"])
            self._tgt_buffer["quat"].append(self._tgt["quat"])
            self._tgt_buffer["vel"].append(self._tgt["vel"])
            self._tgt_buffer["ang"].append(self._tgt["ang"])

        if self._tgt["pos"] is not None:
            self._tgt["pos"].assert_contiguous()
            self._tgt["pos"].assert_sceneless()
            kernel_set_pos(self._sim.cur_step_local, self._tgt["pos"], self.data, self.solver.info)

        if self._tgt["quat"] is not None:
            self._tgt["quat"].assert_contiguous()
            self._tgt["quat"].assert_sceneless()
            kernel_set_quat(self._sim.cur_step_local, self._tgt["quat"], self.data, self.solver.info)

        if self._tgt["vel"] is not None:
            self._tgt["vel"].assert_contiguous()
            self._tgt["vel"].assert_sceneless()
            kernel_set_vel(self._sim.cur_step_local, self._tgt["vel"], self.data, self.solver.info)

        if self._tgt["ang"] is not None:
            self._tgt["ang"].assert_contiguous()
            self._tgt["ang"].assert_sceneless()
            kernel_set_ang(self._sim.cur_step_local, self._tgt["ang"], self.data, self.solver.info)

        self._tgt["pos"] = None
        self._tgt["quat"] = None
        self._tgt["vel"] = None
        self._tgt["ang"] = None

    def process_input_grad(self):
        _tgt_pos = self._tgt_buffer["pos"].pop()
        _tgt_quat = self._tgt_buffer["quat"].pop()
        _tgt_vel = self._tgt_buffer["vel"].pop()
        _tgt_ang = self._tgt_buffer["ang"].pop()

        if _tgt_vel is not None and _tgt_vel.requires_grad:
            _tgt_vel._backward_from_qd(kernel_set_vel_grad, self._sim.cur_step_local, self.data, self.solver.info)

        if _tgt_ang is not None and _tgt_ang.requires_grad:
            _tgt_ang._backward_from_qd(kernel_set_ang_grad, self._sim.cur_step_local, self.data, self.solver.info)

        if _tgt_pos is not None and _tgt_pos.requires_grad:
            _tgt_pos._backward_from_qd(kernel_set_pos_grad, self._sim.cur_step_local, self.data, self.solver.info)

        if _tgt_quat is not None and _tgt_quat.requires_grad:
            _tgt_quat._backward_from_qd(kernel_set_quat_grad, self._sim.cur_step_local, self.data, self.solver.info)

    def collect_output_grads(self):
        """
        Collect gradients from external queried states.
        """
        if self._sim.cur_step_global in self._queried_states:
            # one step could have multiple states
            for state in self._queried_states[self._sim.cur_step_global]:
                self.add_grad_from_state(state)

    def add_grad_from_state(self, state):
        if state.pos.grad is not None:
            state.pos.assert_contiguous()
            kernel_add_frame_grad_pos(self._sim.cur_substep_local, state.pos.grad, self.data)

        if state.quat.grad is not None:
            state.quat.assert_contiguous()
            kernel_add_frame_grad_quat(self._sim.cur_substep_local, state.quat.grad, self.data)

        if state.vel.grad is not None:
            state.vel.assert_contiguous()
            kernel_add_frame_grad_vel(self._sim.cur_substep_local, state.vel.grad, self.data)

        if state.ang.grad is not None:
            state.ang.assert_contiguous()
            kernel_add_frame_grad_ang(self._sim.cur_substep_local, state.ang.grad, self.data)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def data(self) -> ToolEntityData:
        assert self._data is not None
        return self._data

    @property
    def pos(self):
        return self.data.state.pos

    @property
    def quat(self):
        return self.data.state.quat

    @property
    def vel(self):
        return self.data.state.vel

    @property
    def ang(self):
        return self.data.state.ang

    @property
    def latest_pos(self):
        return self.data.state.latest_pos

    @property
    def uid(self):
        return self._uid

    @property
    def idx(self):
        return self._idx

    @property
    def scene(self):
        return self._scene

    @property
    def solver(self):
        return self._solver

    @property
    def material(self):
        return self._material

    @property
    def morph(self):
        return self._morph

    @property
    def surface(self):
        return self._surface

    @property
    def init_pos(self):
        return self._init_pos

    @property
    def init_quat(self):
        return self._init_quat


@qd.kernel
def kernel_save_ckpt(
    pos: qd.types.ndarray(),
    quat: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    ang: qd.types.ndarray(),
    entity_data: ToolEntityData,
):
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(3)):
            pos[i_b, k] = entity_data.state.pos[0, i_b][k]
            vel[i_b, k] = entity_data.state.vel[0, i_b][k]
            ang[i_b, k] = entity_data.state.ang[0, i_b][k]
        for k in qd.static(range(4)):
            quat[i_b, k] = entity_data.state.quat[0, i_b][k]


@qd.kernel
def kernel_load_ckpt(
    pos: qd.types.ndarray(),
    quat: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    ang: qd.types.ndarray(),
    entity_data: ToolEntityData,
):
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(3)):
            entity_data.state.pos[0, i_b][k] = pos[i_b, k]
            entity_data.state.vel[0, i_b][k] = vel[i_b, k]
            entity_data.state.ang[0, i_b][k] = ang[i_b, k]
        for k in qd.static(range(4)):
            entity_data.state.quat[0, i_b][k] = quat[i_b, k]


@qd.kernel
def kernel_update_latest_pos(f: int, entity_data: ToolEntityData):
    entity_data.state.latest_pos[0] = entity_data.state.pos[f, 0]


@qd.func(requires_top_level=True)
def func_update_latest_pos(f: int, entity_data: ToolEntityData):
    for _ in range(1):
        entity_data.state.latest_pos[0] = entity_data.state.pos[f, 0]


@qd.kernel
def kernel_advect(
    f: int,
    entity_data: ToolEntityData,
    solver_info: ToolSolverInfo,
    solver_config: qd.template(),
):
    for i_b in range(entity_data.state.pos.shape[1]):
        entity_data.state.pos[f + 1, i_b] = solver_config.boundary.impose_pos(
            entity_data.state.pos[f, i_b] + entity_data.state.vel[f, i_b] * solver_info.substep_dt[None]
        )
        # Rotate in world coordinates about the entity origin
        entity_data.state.quat[f + 1, i_b] = qd_transform_quat_by_quat(
            entity_data.state.quat[f, i_b],
            qd_rotvec_to_quat(entity_data.state.ang[f, i_b] * solver_info.substep_dt[None], gs.EPS),
        )
        entity_data.state.vel[f + 1, i_b] = entity_data.state.vel[f, i_b]
        entity_data.state.ang[f + 1, i_b] = entity_data.state.ang[f, i_b]


@qd.func(requires_top_level=True)
def func_advect(
    f: int,
    entity_data: ToolEntityData,
    solver_info: ToolSolverInfo,
    solver_config: qd.template(),
):
    for i_b in range(entity_data.state.pos.shape[1]):
        entity_data.state.pos[f + 1, i_b] = solver_config.boundary.impose_pos(
            entity_data.state.pos[f, i_b] + entity_data.state.vel[f, i_b] * solver_info.substep_dt[None]
        )
        entity_data.state.quat[f + 1, i_b] = qd_transform_quat_by_quat(
            entity_data.state.quat[f, i_b],
            qd_rotvec_to_quat(entity_data.state.ang[f, i_b] * solver_info.substep_dt[None], gs.EPS),
        )
        entity_data.state.vel[f + 1, i_b] = entity_data.state.vel[f, i_b]
        entity_data.state.ang[f + 1, i_b] = entity_data.state.ang[f, i_b]


@qd.kernel
def kernel_copy_frame(source: int, target: int, entity_data: ToolEntityData):
    for i_b in range(entity_data.state.pos.shape[1]):
        entity_data.state.pos[target, i_b] = entity_data.state.pos[source, i_b]
        entity_data.state.quat[target, i_b] = entity_data.state.quat[source, i_b]
        entity_data.state.vel[target, i_b] = entity_data.state.vel[source, i_b]
        entity_data.state.ang[target, i_b] = entity_data.state.ang[source, i_b]


@qd.kernel
def kernel_copy_grad(source: int, target: int, entity_data: ToolEntityData):
    for i_b in range(entity_data.state.pos.shape[1]):
        entity_data.state.pos.grad[target, i_b] = entity_data.state.pos.grad[source, i_b]
        entity_data.state.quat.grad[target, i_b] = entity_data.state.quat.grad[source, i_b]
        entity_data.state.vel.grad[target, i_b] = entity_data.state.vel.grad[source, i_b]
        entity_data.state.ang.grad[target, i_b] = entity_data.state.ang.grad[source, i_b]


@qd.kernel
def kernel_reset_grad_till_frame(f: int, entity_data: ToolEntityData):
    for i_b in range(entity_data.state.pos.shape[1]):
        for i_f in range(f):
            entity_data.state.pos.grad[i_f, i_b].fill(0)
            entity_data.state.quat.grad[i_f, i_b].fill(0)
            entity_data.state.vel.grad[i_f, i_b].fill(0)
            entity_data.state.ang.grad[i_f, i_b].fill(0)


@qd.kernel
def kernel_get_frame(
    f: int,
    pos: qd.types.ndarray(),
    quat: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    ang: qd.types.ndarray(),
    entity_data: ToolEntityData,
):
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(3)):
            pos[i_b, k] = entity_data.state.pos[f, i_b][k]
            vel[i_b, k] = entity_data.state.vel[f, i_b][k]
            ang[i_b, k] = entity_data.state.ang[f, i_b][k]
        for k in qd.static(range(4)):
            quat[i_b, k] = entity_data.state.quat[f, i_b][k]


@qd.kernel
def kernel_set_frame(
    f: int,
    pos: qd.types.ndarray(),
    quat: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    ang: qd.types.ndarray(),
    entity_data: ToolEntityData,
):
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(3)):
            entity_data.state.pos[f, i_b][k] = pos[i_b, k]
            entity_data.state.vel[f, i_b][k] = vel[i_b, k]
            entity_data.state.ang[f, i_b][k] = ang[i_b, k]
        for k in qd.static(range(4)):
            entity_data.state.quat[f, i_b][k] = quat[i_b, k]


@qd.kernel
def kernel_add_frame_grad_pos(f: int, pos_grad: qd.types.ndarray(), entity_data: ToolEntityData):
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(3)):
            entity_data.state.pos.grad[f, i_b][k] += pos_grad[i_b, k]


@qd.kernel
def kernel_add_frame_grad_quat(f: int, quat_grad: qd.types.ndarray(), entity_data: ToolEntityData):
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(4)):
            entity_data.state.quat.grad[f, i_b][k] += quat_grad[i_b, k]


@qd.kernel
def kernel_add_frame_grad_vel(f: int, vel_grad: qd.types.ndarray(), entity_data: ToolEntityData):
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(3)):
            entity_data.state.vel.grad[f, i_b][k] += vel_grad[i_b, k]


@qd.kernel
def kernel_add_frame_grad_ang(f: int, ang_grad: qd.types.ndarray(), entity_data: ToolEntityData):
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(3)):
            entity_data.state.ang.grad[f, i_b][k] += ang_grad[i_b, k]


@qd.kernel
def kernel_set_init_state(pos: qd.types.ndarray(), quat: qd.types.ndarray(), entity_data: ToolEntityData):
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(3)):
            entity_data.state.pos[0, i_b][k] = pos[k]
        for k in qd.static(range(4)):
            entity_data.state.quat[0, i_b][k] = quat[k]


@qd.kernel
def kernel_set_vel(s: int, vel: qd.types.ndarray(), entity_data: ToolEntityData, solver_info: ToolSolverInfo):
    f = s * solver_info.substeps[None]
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(3)):
            entity_data.state.vel[f, i_b][k] = vel[i_b, k]


@qd.kernel
def kernel_set_vel_grad(
    s: int,
    entity_data: ToolEntityData,
    solver_info: ToolSolverInfo,
    vel_grad: qd.types.ndarray(),
):
    f = s * solver_info.substeps[None]
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(3)):
            vel_grad[i_b, k] += entity_data.state.vel.grad[f, i_b][k]


@qd.kernel
def kernel_set_ang(s: int, ang: qd.types.ndarray(), entity_data: ToolEntityData, solver_info: ToolSolverInfo):
    f = s * solver_info.substeps[None]
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(3)):
            entity_data.state.ang[f, i_b][k] = ang[i_b, k]


@qd.kernel
def kernel_set_ang_grad(
    s: int,
    entity_data: ToolEntityData,
    solver_info: ToolSolverInfo,
    ang_grad: qd.types.ndarray(),
):
    f = s * solver_info.substeps[None]
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(3)):
            ang_grad[i_b, k] += entity_data.state.ang.grad[f, i_b][k]


@qd.kernel
def kernel_set_pos(s: int, pos: qd.types.ndarray(), entity_data: ToolEntityData, solver_info: ToolSolverInfo):
    f = s * solver_info.substeps[None]
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(3)):
            entity_data.state.pos[f, i_b][k] = pos[i_b, k]


@qd.kernel
def kernel_set_pos_grad(
    s: int,
    entity_data: ToolEntityData,
    solver_info: ToolSolverInfo,
    pos_grad: qd.types.ndarray(),
):
    f = s * solver_info.substeps[None]
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(3)):
            pos_grad[i_b, k] += entity_data.state.pos.grad[f, i_b][k]


@qd.kernel
def kernel_set_quat(s: int, quat: qd.types.ndarray(), entity_data: ToolEntityData, solver_info: ToolSolverInfo):
    f = s * solver_info.substeps[None]
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(4)):
            entity_data.state.quat[f, i_b][k] = quat[i_b, k]


@qd.kernel
def kernel_set_quat_grad(
    s: int,
    entity_data: ToolEntityData,
    solver_info: ToolSolverInfo,
    quat_grad: qd.types.ndarray(),
):
    f = s * solver_info.substeps[None]
    for i_b in range(entity_data.state.pos.shape[1]):
        for k in qd.static(range(4)):
            quat_grad[i_b, k] += entity_data.state.quat.grad[f, i_b][k]
