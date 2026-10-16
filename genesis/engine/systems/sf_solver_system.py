"""System of the stable fluid solver, which owns the grid and advances it once per substep."""

import dataclasses
from typing import TYPE_CHECKING, ClassVar

import quadrants as qd

import genesis as gs
from genesis.engine.core import System, register_system
from genesis.engine.solvers.base_solver import adopt_model
from genesis.utils.array_class import V, V_SCALAR_FROM, V_VEC, AutoInitMeta, DataKind, of_kind
from genesis.utils.misc import qd_to_numpy

from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


# The model of the stable fluid solver, which the user-facing solver resolves from its options and its jets
SF_MODEL = ("res", "dx", "solver_iters", "decay", "inlet_s", "jets", "_substep_dt")


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SFGridState:
    """State of every cell of the grid: its velocity, its pressure, and the smoke concentration of every jet."""

    kind: ClassVar[DataKind] = DataKind.STATE

    v: qd.Tensor
    v_tmp: qd.Tensor = of_kind(DataKind.SCRATCH)
    div: qd.Tensor = of_kind(DataKind.SCRATCH)
    # The pressure of the previous substep starts the Jacobi iterations of the next one
    p: qd.Tensor
    q: qd.Tensor
    t: qd.Tensor


def get_sf_grid_state(system):
    return SFGridState(
        v=V_VEC(3, dtype=gs.qd_float, shape=system.res),
        v_tmp=V_VEC(3, dtype=gs.qd_float, shape=system.res),
        div=V(dtype=gs.qd_float, shape=system.res),
        p=V(dtype=gs.qd_float, shape=system.res),
        q=V_VEC(len(system.jets), dtype=gs.qd_float, shape=system.res),
        # The legacy host clock accumulates as a Python double and reaches the jet funcs as qd.f32
        t=V_SCALAR_FROM(dtype=qd.f64, value=0.0),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SFGridInfo:
    """Cell size of the grid, and the substep, the inlet speed and the smoke decay rate that advance it."""

    kind: ClassVar[DataKind] = DataKind.CONSTANT

    dx: qd.Tensor
    substep_dt: qd.Tensor
    inlet_s: qd.Tensor
    decay: qd.Tensor
    solver_iters: qd.Tensor
    clock_dt: qd.Tensor


def get_sf_grid_info(system):
    return SFGridInfo(
        dx=V_SCALAR_FROM(dtype=gs.qd_float, value=system.dx),
        substep_dt=V_SCALAR_FROM(dtype=gs.qd_float, value=system._substep_dt),
        inlet_s=V_SCALAR_FROM(dtype=gs.qd_float, value=system.inlet_s),
        decay=V_SCALAR_FROM(dtype=gs.qd_float, value=system.decay),
        solver_iters=V_SCALAR_FROM(dtype=gs.qd_int, value=system.solver_iters),
        clock_dt=V_SCALAR_FROM(dtype=qd.f64, value=system._substep_dt),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SFProjectionState:
    """Scratch buffers that alternate as the source and destination of the pressure Jacobi iteration."""

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    p: qd.Tensor
    p_next: qd.Tensor


def get_sf_projection_state(system):
    return SFProjectionState(
        p=V(dtype=gs.qd_float, shape=system.res),
        p_next=V(dtype=gs.qd_float, shape=system.res),
    )


@qd.data_oriented
class SFStaticConfig(metaclass=AutoInitMeta):
    # The jets of the user, whose funcs every kernel advecting the grid inlines
    jets: tuple


class SFSolverSystem(System):
    """System owning the grid of the stable fluid solver of the scene, which reaches it through forwarding attributes.

    The Initializer has it allocate and initialize the grid from the jets of the solver. A substep advects the grid by
    its own velocity, injects the momentum and the smoke of the jets, and projects the velocity onto a divergence-free
    field by Jacobi iterations of the pressure Poisson equation.
    """

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.sf_solver

    @System.data
    def state_data(self):
        return self.grid_state

    @System.data
    def info_data(self):
        return self.grid_info

    @System.data
    def projection_data(self):
        return self.projection_state

    @System.data
    def static_config(self):
        return self.sf_config

    @System.data
    def graph_condition(self):
        return self.graph_counter

    @property
    def t(self) -> float:
        """The simulated duration of the grid, in seconds."""
        return float(qd_to_numpy(self.grid_state.t, transpose=True)[()])

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    def initialize_data(self) -> None:
        """Take over the model from the solver, then allocate and initialize the grid, which the solver forwards to
        from here on."""
        adopt_model(self, SF_MODEL)
        self.solver._system = self
        self.grid_state = get_sf_grid_state(self)
        self.grid_info = get_sf_grid_info(self)
        self.projection_state = get_sf_projection_state(self)
        self.sf_config = SFStaticConfig(jets=self.jets)
        # graph.do_while reads its condition from the same physical ndarray at every iteration
        self.graph_counter = qd.ndarray(qd.i32, shape=())
        kernel_init_grid(self.grid_state, self.sf_config)

    @System.action(kind="host")
    def process_input(self):
        return self.solver.process_input

    @System.action(kind="host")
    def pre_coupling(self):
        return self.substep_pre_coupling

    @System.action(kind="host")
    def post_coupling(self):
        return self.substep_post_coupling

    @System.action(kind="stage")
    def graph_pre_coupling(self):
        return (func_graph_sf_odd if self.solver_iters % 2 else func_graph_sf_even), self

    def substep_pre_coupling(self, f: int) -> None:
        """Advect the grid by one substep, inject the jets, then project the velocity onto a divergence-free field."""
        if self.solver_iters % 2 == 0:
            kernel_substep_even(
                self.graph_counter,
                self.grid_state,
                self.grid_info,
                self.projection_state,
                self.sf_config,
            )
        else:
            kernel_substep_odd(
                self.graph_counter,
                self.grid_state,
                self.grid_info,
                self.projection_state,
                self.sf_config,
            )

    def substep_post_coupling(self, f: int) -> None:
        """The stable fluid solver advances nothing after the coupling."""


@register_system(SimulatorEngine)
def create_sf_solver_system(scene: "Scene") -> SFSolverSystem | None:
    """Create the System owning the grid of the stable fluid solver whenever it has a jet to simulate."""
    return SFSolverSystem(scene) if scene.sim.sf_solver.is_active else None


@qd.kernel
def kernel_init_grid(grid_state: SFGridState, sf_config: qd.template()):
    for I in qd.grouped(qd.ndrange(grid_state.q.shape[0], grid_state.q.shape[1], grid_state.q.shape[2])):
        for q in qd.static(range(len(sf_config.jets))):
            grid_state.q[I][q] = 0.0


@qd.kernel(graph=True)
def kernel_substep_even(
    graph_counter: qd.types.ndarray(qd.i32, ndim=0),
    grid_state: SFGridState,
    grid_info: SFGridInfo,
    projection_state: SFProjectionState,
    sf_config: qd.template(),
):
    func_advect_and_impulse(grid_state, grid_info, sf_config)
    func_divergence(grid_state)
    func_projection_init(graph_counter, grid_state, grid_info, projection_state)
    while qd.graph.do_while(graph_counter):
        func_pressure_jacobi_to_next(grid_state, projection_state)
        func_pressure_jacobi_to_current(graph_counter, grid_state, projection_state)
    func_projection_finish(grid_state, projection_state, is_odd=False)
    func_subtract_gradient(grid_state, grid_info)


@qd.kernel(graph=True)
def kernel_substep_odd(
    graph_counter: qd.types.ndarray(qd.i32, ndim=0),
    grid_state: SFGridState,
    grid_info: SFGridInfo,
    projection_state: SFProjectionState,
    sf_config: qd.template(),
):
    func_advect_and_impulse(grid_state, grid_info, sf_config)
    func_divergence(grid_state)
    func_projection_init(graph_counter, grid_state, grid_info, projection_state)
    while qd.graph.do_while(graph_counter):
        func_pressure_jacobi_to_next(grid_state, projection_state)
        func_pressure_jacobi_to_current(graph_counter, grid_state, projection_state)
    func_pressure_jacobi_odd(grid_state, projection_state)
    func_projection_finish(grid_state, projection_state, is_odd=True)
    func_subtract_gradient(grid_state, grid_info)


@qd.func(requires_top_level=True)
def func_graph_sf_even(system: qd.template(), f: int, t: float):
    func_advect_and_impulse(system.grid_state, system.grid_info, system.sf_config)
    func_divergence(system.grid_state)
    func_projection_init(system.graph_counter, system.grid_state, system.grid_info, system.projection_state)
    while qd.graph.do_while(system.graph_counter):
        func_pressure_jacobi_to_next(system.grid_state, system.projection_state)
        func_pressure_jacobi_to_current(system.graph_counter, system.grid_state, system.projection_state)
    func_projection_finish(system.grid_state, system.projection_state, is_odd=False)
    func_subtract_gradient(system.grid_state, system.grid_info)


@qd.func(requires_top_level=True)
def func_graph_sf_odd(system: qd.template(), f: int, t: float):
    func_advect_and_impulse(system.grid_state, system.grid_info, system.sf_config)
    func_divergence(system.grid_state)
    func_projection_init(system.graph_counter, system.grid_state, system.grid_info, system.projection_state)
    while qd.graph.do_while(system.graph_counter):
        func_pressure_jacobi_to_next(system.grid_state, system.projection_state)
        func_pressure_jacobi_to_current(system.graph_counter, system.grid_state, system.projection_state)
    func_pressure_jacobi_odd(system.grid_state, system.projection_state)
    func_projection_finish(system.grid_state, system.projection_state, is_odd=True)
    func_subtract_gradient(system.grid_state, system.grid_info)


@qd.func(requires_top_level=True)
def func_advect_and_impulse(grid_state: SFGridState, grid_info: SFGridInfo, sf_config: qd.template()):
    n_grid = grid_state.v.shape[0]
    substep_dt = grid_info.substep_dt[None]
    t = qd.cast(grid_state.t[None], qd.f32)
    for i, j, k in qd.ndrange(n_grid, n_grid, n_grid):
        p = qd.Vector([i, j, k], dt=gs.qd_float) + 0.5
        p = func_backtrace(grid_state.v, p, substep_dt, n_grid)
        v_tmp = func_trilerp(grid_state.v, p, n_grid)

        for q in qd.static(range(len(sf_config.jets))):
            q_f = func_trilerp_scalar(grid_state.q, p, q, n_grid)

            imp_dir = sf_config.jets[q].get_tan_dir(t)
            factor = sf_config.jets[q].get_factor(i, j, k, grid_info.dx[None], t)
            momentum = (imp_dir * grid_info.inlet_s[None] * factor) * substep_dt

            v_tmp += momentum

            grid_state.q[i, j, k][q] = (1 - factor) * q_f + factor
            grid_state.q[i, j, k][q] -= grid_info.decay[None] * substep_dt
            grid_state.q[i, j, k][q] = max(0.0, grid_state.q[i, j, k][q])

        grid_state.v_tmp[i, j, k] = v_tmp


@qd.func(requires_top_level=True)
def func_divergence(grid_state: SFGridState):
    n_grid = grid_state.v_tmp.shape[0]
    for u, v, w in qd.ndrange(n_grid, n_grid, n_grid):
        vl = grid_state.v_tmp[func_compute_location(u, v, w, -1, 0, 0, n_grid)]
        vr = grid_state.v_tmp[func_compute_location(u, v, w, 1, 0, 0, n_grid)]
        vb = grid_state.v_tmp[func_compute_location(u, v, w, 0, -1, 0, n_grid)]
        vt = grid_state.v_tmp[func_compute_location(u, v, w, 0, 1, 0, n_grid)]
        vp = grid_state.v_tmp[func_compute_location(u, v, w, 0, 0, -1, n_grid)]
        vq = grid_state.v_tmp[func_compute_location(u, v, w, 0, 0, 1, n_grid)]
        vc = grid_state.v_tmp[func_compute_location(u, v, w, 0, 0, 0, n_grid)]

        if not func_is_free(u, v, w, -1, 0, 0, n_grid):
            vl.x = -vc.x
        if not func_is_free(u, v, w, 1, 0, 0, n_grid):
            vr.x = -vc.x
        if not func_is_free(u, v, w, 0, -1, 0, n_grid):
            vb.y = -vc.y
        if not func_is_free(u, v, w, 0, 1, 0, n_grid):
            vt.y = -vc.y
        if not func_is_free(u, v, w, 0, 0, -1, n_grid):
            vp.z = -vc.z
        if not func_is_free(u, v, w, 0, 0, 1, n_grid):
            vq.z = -vc.z

        grid_state.div[u, v, w] = 0.5 * (vr.x - vl.x + vt.y - vb.y + vq.z - vp.z)


@qd.func(requires_top_level=True)
def func_projection_init(
    graph_counter: qd.types.ndarray(qd.i32, ndim=0),
    grid_state: SFGridState,
    grid_info: SFGridInfo,
    projection_state: SFProjectionState,
):
    for I in qd.grouped(qd.ndrange(grid_state.p.shape[0], grid_state.p.shape[1], grid_state.p.shape[2])):
        projection_state.p[I] = grid_state.p[I]
        projection_state.p_next[I] = 0.0
        if I[0] == 0 and I[1] == 0 and I[2] == 0:
            graph_counter[()] = grid_info.solver_iters[None] // 2


@qd.func
def func_pressure_jacobi_cell(u: int, v: int, w: int, p: qd.template(), grid_state: SFGridState):
    n_grid = grid_state.p.shape[0]
    p_left = p[func_compute_location(u, v, w, -1, 0, 0, n_grid)]
    p_right = p[func_compute_location(u, v, w, 1, 0, 0, n_grid)]
    p_bottom = p[func_compute_location(u, v, w, 0, -1, 0, n_grid)]
    p_top = p[func_compute_location(u, v, w, 0, 1, 0, n_grid)]
    p_back = p[func_compute_location(u, v, w, 0, 0, -1, n_grid)]
    p_front = p[func_compute_location(u, v, w, 0, 0, 1, n_grid)]
    return (p_left + p_right + p_bottom + p_top + p_back + p_front - grid_state.div[u, v, w]) / 6.0


@qd.func(requires_top_level=True)
def func_pressure_jacobi_to_next(grid_state: SFGridState, projection_state: SFProjectionState):
    n_grid = grid_state.p.shape[0]
    for u, v, w in qd.ndrange(n_grid, n_grid, n_grid):
        projection_state.p_next[u, v, w] = func_pressure_jacobi_cell(u, v, w, projection_state.p, grid_state)


@qd.func(requires_top_level=True)
def func_pressure_jacobi_to_current(
    graph_counter: qd.types.ndarray(qd.i32, ndim=0),
    grid_state: SFGridState,
    projection_state: SFProjectionState,
):
    n_grid = grid_state.p.shape[0]
    for u, v, w in qd.ndrange(n_grid, n_grid, n_grid):
        projection_state.p[u, v, w] = func_pressure_jacobi_cell(u, v, w, projection_state.p_next, grid_state)
        if u == 0 and v == 0 and w == 0:
            graph_counter[()] = graph_counter[()] - 1


@qd.func(requires_top_level=True)
def func_pressure_jacobi_odd(
    grid_state: SFGridState,
    projection_state: SFProjectionState,
):
    n_grid = grid_state.p.shape[0]
    for u, v, w in qd.ndrange(n_grid, n_grid, n_grid):
        projection_state.p_next[u, v, w] = func_pressure_jacobi_cell(u, v, w, projection_state.p, grid_state)


@qd.func(requires_top_level=True)
def func_projection_finish(
    grid_state: SFGridState,
    projection_state: SFProjectionState,
    is_odd: qd.template(),
):
    for I in qd.grouped(qd.ndrange(grid_state.p.shape[0], grid_state.p.shape[1], grid_state.p.shape[2])):
        if qd.static(is_odd):
            grid_state.p[I] = projection_state.p_next[I]
        else:
            grid_state.p[I] = projection_state.p[I]
        projection_state.p[I] = 0.0
        projection_state.p_next[I] = 0.0


@qd.func(requires_top_level=True)
def func_subtract_gradient(grid_state: SFGridState, grid_info: SFGridInfo):
    n_grid = grid_state.p.shape[0]
    for I in qd.grouped(qd.ndrange(n_grid, n_grid, n_grid)):
        u, v, w = I
        pl = grid_state.p[func_compute_location(u, v, w, -1, 0, 0, n_grid)]
        pr = grid_state.p[func_compute_location(u, v, w, 1, 0, 0, n_grid)]
        pb = grid_state.p[func_compute_location(u, v, w, 0, -1, 0, n_grid)]
        pt = grid_state.p[func_compute_location(u, v, w, 0, 1, 0, n_grid)]
        pp = grid_state.p[func_compute_location(u, v, w, 0, 0, -1, n_grid)]
        pq = grid_state.p[func_compute_location(u, v, w, 0, 0, 1, n_grid)]

        grid_state.v[I] = grid_state.v_tmp[I] - 0.5 * qd.Vector([pr - pl, pt - pb, pq - pp], dt=gs.qd_float)
        if I[0] == 0 and I[1] == 0 and I[2] == 0:
            grid_state.t[None] += grid_info.clock_dt[None]


@qd.func
def func_compute_location(u: int, v: int, w: int, du: int, dv: int, dw: int, n_grid: int):
    I = qd.Vector([u + du, v + dv, w + dw], dt=gs.qd_int)
    return qd.math.clamp(I, 0, n_grid - 1)


@qd.func
def func_is_free(u: int, v: int, w: int, du: int, dv: int, dw: int, n_grid: int):
    I = qd.Vector([u + du, v + dv, w + dw], dt=gs.qd_int)
    return gs.qd_bool((0 <= I).all() and (I < n_grid).all())


@qd.func
def func_trilerp_scalar(qf: qd.template(), p: qd.types.vector(3), qf_idx: qd.template(), n_grid: int):
    """Interpolate one component of a vector field trilinearly at a position given in cell units.

    The weights of the cells outside of the grid are dropped, and the remaining ones renormalized.
    """
    # convert position to grid index
    base_I = qd.floor(p - 0.5, gs.qd_int)
    p_I = p - 0.5

    q = 0.0
    w_total = 0.0
    for offset in qd.static(qd.grouped(qd.ndrange(2, 2, 2))):
        grid_I = base_I + offset
        w_xyz = 1 - qd.abs(p_I - grid_I)
        w = w_xyz[0] * w_xyz[1] * w_xyz[2]
        grid_I_ = func_compute_location(grid_I[0], grid_I[1], grid_I[2], 0, 0, 0, n_grid)
        q += w * qf[grid_I_][qf_idx]
        w_total += w
    # w_total is less then one when at boundary
    q /= w_total
    return q


@qd.func
def func_trilerp(qf: qd.template(), p: qd.types.vector(3), n_grid: int):
    """Interpolate a 3D vector field trilinearly at a position given in cell units.

    The weights of the cells outside of the grid are dropped, and the remaining ones renormalized.
    """
    # convert position to grid index
    base_I = qd.floor(p - 0.5, gs.qd_int)
    p_I = p - 0.5

    q = qd.Vector([0.0, 0.0, 0.0], dt=gs.qd_float)
    w_total = 0.0
    for offset in qd.static(qd.grouped(qd.ndrange(2, 2, 2))):
        grid_I = base_I + offset
        w_xyz = 1 - qd.abs(p_I - grid_I)
        w = w_xyz[0] * w_xyz[1] * w_xyz[2]
        grid_I_ = func_compute_location(grid_I[0], grid_I[1], grid_I[2], 0, 0, 0, n_grid)
        q += w * qf[grid_I_]
        w_total += w
    # w_total is less then one when at boundary
    q /= w_total
    return q


@qd.func
def func_backtrace(vf: qd.template(), p: qd.types.vector(3), dt: float, n_grid: int):
    """Trace a position given in cell units back by one substep through a velocity field, by third-order Runge-Kutta
    (RK3)."""
    v1 = func_trilerp(vf, p, n_grid)
    p1 = p - 0.5 * dt * v1
    v2 = func_trilerp(vf, p1, n_grid)
    p2 = p - 0.75 * dt * v2
    v3 = func_trilerp(vf, p2, n_grid)
    p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p
