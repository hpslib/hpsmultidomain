import numpy as np
import scipy.sparse as sp

from hpsmultidomain.sparse_utils import (
    CSRBuilder,
    SparseSolver,
    dense_lu_inverse_operator,
    estimate_condition_number,
    estimate_effective_condition_number,
    estimate_operator_2norm,
)


def test_condition_estimator_matches_complex_nonnormal_matrix():
    A = np.array(
        [
            [5.0 + 0.5j, 8.0 - 1.0j, 0.0],
            [0.0, 2.0 - 1.0j, 3.0 + 0.5j],
            [0.0, 0.0, 1.0 + 2.0j],
        ],
        dtype=np.complex128,
    )

    op_norm = estimate_operator_2norm(A, nit=20, seed=11)
    exact_norm = np.linalg.norm(A, 2)
    assert abs(op_norm - exact_norm) / exact_norm < 1e-10

    inv_op = dense_lu_inverse_operator(A)
    inv_norm = estimate_operator_2norm(inv_op, nit=20, seed=23)
    exact_inv_norm = np.linalg.norm(np.linalg.inv(A), 2)
    assert abs(inv_norm - exact_inv_norm) / exact_inv_norm < 1e-10

    _, _, cond_est = estimate_condition_number(A, inv_op, nit=20, seed=11)
    exact_cond = np.linalg.cond(A, 2)
    assert abs(cond_est - exact_cond) / exact_cond < 1e-10

    rhs = np.array([1.0 - 0.5j, 2.0, -1.0 + 1.0j], dtype=np.complex128)
    exact_eff_cond = exact_norm * np.linalg.norm(np.linalg.solve(A, rhs)) / np.linalg.norm(rhs)
    eff_cond = estimate_effective_condition_number(A, rhs, solve_op=inv_op, op_norm=op_norm)
    assert abs(eff_cond - exact_eff_cond) / exact_eff_cond < 1e-10


def test_dense_lu_inverse_operator_uses_adjoint_solve():
    A = np.array(
        [
            [3.0, 2.0 - 1.0j],
            [0.5 + 0.25j, -4.0],
        ],
        dtype=np.complex128,
    )
    rhs = np.array([1.0 + 2.0j, -3.0], dtype=np.complex128)
    rhs_mat = np.column_stack([rhs, rhs.conj()])

    inv_op = dense_lu_inverse_operator(A)

    np.testing.assert_allclose(inv_op.matvec(rhs), np.linalg.solve(A, rhs))
    np.testing.assert_allclose(inv_op.rmatvec(rhs), np.linalg.solve(A.conj().T, rhs))
    np.testing.assert_allclose(inv_op.matmat(rhs_mat), np.linalg.solve(A, rhs_mat))
    np.testing.assert_allclose(inv_op.rmatmat(rhs_mat), np.linalg.solve(A.conj().T, rhs_mat))


def test_sparse_solver_preserves_complex_values():
    A = sp.csr_matrix(
        np.array(
            [
                [4.0 + 1.0j, 1.0 - 0.5j],
                [0.25 + 0.75j, 3.0 - 2.0j],
            ],
            dtype=np.complex128,
        )
    )
    rhs = np.array([1.0 + 2.0j, -0.5 + 0.25j], dtype=np.complex128)

    solver = SparseSolver(A)
    np.testing.assert_allclose(solver.solve_op.matvec(rhs), np.linalg.solve(A.toarray(), rhs))
    if solver.backend == "petsc":
        assert solver.petsc_has_complex
    if solver.backend == "mumps":
        assert solver.mumps_has_complex


def test_csr_builder_preserves_complex_values():
    builder = CSRBuilder(2, 2, 4, dtype=np.complex128)
    builder.add_data(
        sp.coo_matrix(
            (
                np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex128),
                (np.array([0, 1]), np.array([1, 0])),
            ),
            shape=(2, 2),
        )
    )

    A = builder.tocsr()
    assert np.iscomplexobj(A.data)
    np.testing.assert_allclose(A.toarray(), np.array([[0.0, 1.0 + 2.0j], [3.0 - 4.0j, 0.0]]))
