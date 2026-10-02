import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.systems import ComponentPartitioner, GlobalLinearSystem
from genesis.utils.misc import qd_to_numpy


@qd.kernel(graph=True, fastcache=True)
def kernel_initialize_partition(partitioner: qd.template()):
    partitioner.compute_static_labels()


@qd.kernel(graph=True, fastcache=True)
def kernel_partition(partitioner: qd.template()):
    partitioner.partition()


def make_partitioner(
    n_block_rows: int,
    static_edges: np.ndarray,
    *,
    max_sv_iter: int = 64,
) -> tuple[ComponentPartitioner, GlobalLinearSystem]:
    linear_system = GlobalLinearSystem()
    linear_system.do_build()
    linear_system.init(
        n_block_rows=n_block_rows,
        n_elastic_triplets=0,
        max_contact_body_triplets=4,
        dof_block_base=0,
        pcg_tol_rate=1e-4,
    )
    partitioner = ComponentPartitioner()
    partitioner.global_linear_system = linear_system
    partitioner.init(
        n_block_rows,
        max_sv_iter,
        static_edges,
    )
    return partitioner, linear_system


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_component_partitioner_static_and_live_edges():
    partitioner, linear_system = make_partitioner(
        5,
        np.array([[0, 1], [1, 2]], dtype=np.int32),
    )
    linear_system.bcoo_nnz.from_numpy(np.array(1, dtype=np.int32))
    linear_system.bcoo_row.from_numpy(np.array([3, 0, 0, 0], dtype=np.int32))
    linear_system.bcoo_col.from_numpy(np.array([4, 0, 0, 0], dtype=np.int32))

    kernel_initialize_partition(partitioner)
    np.testing.assert_array_equal(
        qd_to_numpy(partitioner.comp_label)[:5],
        np.array([0, 0, 0, 1, 2], dtype=np.int32),
    )
    assert int(qd_to_numpy(partitioner.K)) == 3

    kernel_partition(partitioner)
    np.testing.assert_array_equal(
        qd_to_numpy(partitioner.comp_label)[:5],
        np.array([0, 0, 0, 1, 1], dtype=np.int32),
    )
    assert int(qd_to_numpy(partitioner.K)) == 2


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_component_partitioner_iteration_cap_falls_back_global():
    partitioner, _ = make_partitioner(
        4,
        np.array([[0, 1]], dtype=np.int32),
        max_sv_iter=1,
    )

    kernel_initialize_partition(partitioner)

    np.testing.assert_array_equal(
        qd_to_numpy(partitioner.comp_label)[:4],
        np.zeros(4, dtype=np.int32),
    )
    assert int(qd_to_numpy(partitioner.K)) == 1
