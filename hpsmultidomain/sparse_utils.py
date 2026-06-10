import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, aslinearoperator, splu
from scipy.sparse import csr_matrix, coo_matrix
from time import time

# python-mumps builds against MPI MUMPS on some systems; importing mpi4py first
# initializes MPI and avoids low-level MPICH errors when creating a Context.
try:
    from mpi4py import MPI as _MPI  # noqa: F401
except ImportError:
    _MPI = None

# Attempting to import the python-mumps library for parallel computation, handling failure gracefully
try:
    import mumps
    mumps_imported = True
except ImportError:
    mumps_imported = False
    print("python-mumps not available")

# Try to import PETSc; if successful, we can use its KSP solvers.
try:
    from petsc4py import PETSc
    petsc_imported = True
    print("PETSC IMPORTED")
except ImportError:
    print("PETSC NOT IMPORTED")
    petsc_imported = False


def petsc_supports_complex():
    """
    Return True when the imported PETSc build uses a complex scalar type.
    """
    if not petsc_imported:
        return False
    return np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating)


_mumps_supports_complex = None


def mumps_supports_complex():
    """
    Return True when the imported python-mumps binding accepts complex matrices.
    """
    global _mumps_supports_complex
    if _mumps_supports_complex is not None:
        return _mumps_supports_complex
    if not mumps_imported:
        _mumps_supports_complex = False
        return _mumps_supports_complex

    try:
        A = csr_matrix(
            np.array(
                [[2.0 + 1.0j, 0.5 - 0.25j], [1.0j, 3.0 - 0.5j]],
                dtype=np.complex128,
            )
        )
        inst = mumps.Context()
        inst.analyze(A, ordering='metis')
        inst.factor(A)
        _ = inst.solve(np.array([1.0 + 0.0j, 0.25 - 0.5j], dtype=np.complex128))
        _mumps_supports_complex = True
    except Exception:
        _mumps_supports_complex = False
    return _mumps_supports_complex


def _random_unit_vector(n, dtype, seed=0):
    """
    Draw a random unit vector of length n with the requested dtype.

    Parameters:
    - n: int, vector length
    - dtype: NumPy dtype
    - seed: int, random seed

    Returns:
    - v: 1D NumPy array with ||v||_2 = 1
    """
    rng = np.random.default_rng(seed)
    real_dtype = np.float64 if np.issubdtype(np.dtype(dtype), np.complexfloating) else np.float64
    v = rng.standard_normal(n).astype(real_dtype, copy=False)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        v = v.astype(np.complex128, copy=False) + 1j * rng.standard_normal(n)
    nv = np.linalg.norm(v)
    if nv == 0:
        raise ValueError("random unit vector generation failed")
    return v / nv


def estimate_operator_2norm(op, nit=20, seed=0):
    """
    Estimate the spectral norm ||A||_2 using power iteration on A^* A.

    This is appropriate for possibly nonnormal operators, since it estimates the
    largest singular value rather than the spectral radius.

    Parameters:
    - op: matrix-like object or scipy.sparse.linalg.LinearOperator
    - nit: int, number of power iterations
    - seed: int, random seed for the starting vector

    Returns:
    - float: estimate of ||A||_2
    """
    op = aslinearoperator(op)
    dtype = op.dtype if op.dtype is not None else np.float64
    v = _random_unit_vector(op.shape[1], dtype=dtype, seed=seed)

    for _ in range(nit):
        w = op.matvec(v)
        z = op.rmatvec(w)
        nz = np.linalg.norm(z)
        if nz == 0:
            return 0.0
        v = z / nz

    return float(np.linalg.norm(op.matvec(v)))


def estimate_condition_number(op, op_inv, nit=20, seed=0):
    """
    Estimate the 2-norm condition number kappa_2(A) = ||A||_2 ||A^{-1}||_2.

    Parameters:
    - op: matrix-like object or LinearOperator representing A
    - op_inv: matrix-like object or LinearOperator representing A^{-1}
    - nit: int, number of power iterations for each norm estimate
    - seed: int, random seed for the forward operator

    Returns:
    - op_norm: float, estimate of ||A||_2
    - inv_norm: float, estimate of ||A^{-1}||_2
    - cond_est: float, product op_norm * inv_norm
    """
    op_norm = estimate_operator_2norm(op, nit=nit, seed=seed)
    inv_norm = estimate_operator_2norm(op_inv, nit=nit, seed=seed + 7919)
    return op_norm, inv_norm, op_norm * inv_norm


def estimate_effective_condition_number(op, rhs, solution=None, solve_op=None, op_norm=None, nit=20, seed=0):
    """
    Estimate the RHS-dependent effective condition number ||A||_2 ||A^{-1}b||_2 / ||b||_2.

    Parameters:
    - op: matrix-like object or LinearOperator representing A
    - rhs: right-hand side b
    - solution: optional precomputed A^{-1}b
    - solve_op: optional LinearOperator representing A^{-1}, used if solution is not supplied
    - op_norm: optional precomputed estimate of ||A||_2
    - nit: int, number of power iterations if op_norm is not supplied
    - seed: int, random seed if op_norm is not supplied

    Returns:
    - float: estimate of ||A||_2 ||A^{-1}b||_2 / ||b||_2
    """
    rhs = np.asarray(rhs)
    rhs_norm = np.linalg.norm(rhs)
    if rhs_norm == 0:
        return 0.0

    if solution is None:
        if solve_op is None:
            raise ValueError("Either solution or solve_op must be supplied.")
        solve_op = aslinearoperator(solve_op)
        solution = solve_op.matvec(rhs) if rhs.ndim == 1 else solve_op.matmat(rhs)

    if op_norm is None:
        op_norm = estimate_operator_2norm(op, nit=nit, seed=seed)

    return float(op_norm * np.linalg.norm(solution) / rhs_norm)


def dense_lu_inverse_operator(A):
    """
    Factor a dense matrix once and wrap triangular inverse solves in a LinearOperator.

    This is useful for condition-estimation loops, where A^{-1} and A^{-*} are
    applied many times to the same dense matrix.

    Parameters:
    - A: 2D NumPy array

    Returns:
    - LinearOperator representing A^{-1}
    """
    A = np.asarray(A)
    dtype = np.complex128 if np.iscomplexobj(A) else np.float64
    lu = lu_factor(A)
    lu_adj = lu_factor(A.conj().T)
    return LinearOperator(
        shape=A.shape,
        dtype=dtype,
        matvec=lambda x: lu_solve(lu, x),
        rmatvec=lambda x: lu_solve(lu_adj, x),
        matmat=lambda X: lu_solve(lu, X),
        rmatmat=lambda X: lu_solve(lu_adj, X),
    )


def petscdense_to_nparray(pM):
    """
    Convert a PETSc dense matrix pM into a NumPy ndarray.

    Parameters:
    - pM: PETSc.Mat (dense) object

    Returns:
    - M: 2D NumPy array with the same entries as pM
    """
    # Get all values from PETSc matrix by specifying rows and columns
    M = pM.getValues(range(0, pM.getSize()[0]),
                     range(0, pM.getSize()[1]))
    return M


def setup_ksp(A, use_approx=False):
    """
    Set up a PETSc KSP (Krylov solver) or direct solver on sparse matrix A.

    Parameters:
    - A: scipy.sparse matrix (CSR or convertible to CSR)
    - use_approx: bool
        If False, use direct LU factorization (MUMPS) via PC “preonly”.
        If True, use GMRES with a Hypre preconditioner for an approximate solve.

    Returns:
    - ksp: PETSc KSP object that can solve linear systems with A
    """
    # Ensure A is in CSR format for building PETSc AIJ matrix
    A = A.tocsr()

    # Create a new KSP solver
    ksp = PETSc.KSP().create()
    # Wrap A in a PETSc AIJ (sparse) matrix
    pA = PETSc.Mat().createAIJ(A.shape, csr=(A.indptr, A.indices, A.data))
    ksp.setOperators(pA)

    if not use_approx:
        # Direct solve: no Krylov iterations, just factor and solve
        ksp.setType('preonly')
        ksp.getPC().setType('lu')
        ksp.getPC().setFactorSolverType('mumps')
    else:
        # Iterative solve: GMRES + Hypre preconditioner
        ksp.setType('gmres')
        ksp.getPC().setType('hypre')
        # Tight relative tolerance for convergence
        ksp.setTolerances(rtol=5e-14)
        ksp.setConvergenceHistory()

    # Finalize KSP setup
    ksp.setUp()
    return ksp

def setup_mumps(A):
    """
    Set up a PETSc KSP (Krylov solver) or direct solver on sparse matrix A.

    Parameters:
    - A: scipy.sparse matrix (CSR or convertible to CSR)
    - use_approx: bool
        If False, use direct LU factorization (MUMPS) via PC “preonly”.
        If True, use GMRES with a Hypre preconditioner for an approximate solve.

    Returns:
    - ksp: PETSc KSP object that can solve linear systems with A
    """
    # Ensure A is in CSR format for building PETSc AIJ matrix
    A = A.tocsr()

    inst = mumps.Context()
    inst.analyze(A, ordering='metis')
    inst.factor(A)

    return inst


def get_vecsolve(ksp):
    """
    Given a PETSc KSP solver, return a function that solves A x = b for a vector b.

    Parameters:
    - ksp: PETSc KSP object already set up with matrix A

    Returns:
    - vecsolve: function taking a NumPy vector b and returning the solution x
    """
    def vecsolve(b):
        """
        Solve A x = b for a single right-hand side.

        Parameters:
        - b: 1D NumPy array of length N

        Returns:
        - result: 1D NumPy array, solution x
        """
        # Create PETSc vectors from NumPy arrays
        pb = PETSc.Vec().createWithArray(b)
        px = PETSc.Vec().createWithArray(np.zeros(b.shape, dtype=b.dtype))
        # Solve in-place: px = A^{-1} pb
        ksp.solve(pb, px)

        # Extract solution array and clean up PETSc objects
        result = px.getArray()
        px.destroy()
        pb.destroy()
        return result

    return vecsolve


def get_matsolve(ksp, use_approx):
    """
    Given a PETSc KSP solver, return a function that solves A X = B for a dense matrix B.

    If use_approx is False, call the PETSc KSP's matSolve for multiple RHS at once.
    If use_approx is True, fall back on sequential vector solves.

    Parameters:
    - ksp: PETSc KSP object
    - use_approx: bool
        If False, use ksp.matSolve (dense multiple-RHS solve).
        If True, solve each column of B by repeated vec solves.

    Returns:
    - matsolve: function taking a 2D NumPy array B (shape N×M) and returning solution X (N×M)
    """
    def matsolve(B):
        """
        Solve A X = B in a single call using PETSc's matSolve.

        Parameters:
        - B: 2D NumPy array of shape (N, M), right-hand sides

        Returns:
        - result: 2D NumPy array of shape (N, M), solution
        """
        # Wrap B in a PETSc dense matrix
        pB = PETSc.Mat().createDense([B.shape[0], B.shape[1]], None, B)
        # Create an empty PETSc dense matrix for solution
        pX = PETSc.Mat().createDense([B.shape[0], B.shape[1]])
        pX.zeroEntries()

        # Solve: pX = A^{-1} * pB
        ksp.matSolve(pB, pX)
        # Convert PETSc dense matrix back to NumPy array
        result = petscdense_to_nparray(pX)
        # Clean up PETSc matrices
        pX.destroy()
        pB.destroy()
        return result

    def seq_vec_solve(B):
        """
        Solve each column of B separately via the vector solver.

        Parameters:
        - B: 2D NumPy array of shape (N, M)

        Returns:
        - res: 2D NumPy array of same shape, solutions for each column
        """
        vec_solve = get_vecsolve(ksp)
        res = np.zeros(B.shape, dtype=B.dtype)
        for j in range(B.shape[-1]):
            res[:, j] = vec_solve(B[:, j])
        return res

    # Choose which implementation to return based on use_approx
    return matsolve if not use_approx else seq_vec_solve


# ------------------------------------------------------------------------------------
# SparseSolver: wraps either PETSc KSP solvers or SciPy LU for sparse systems
# ------------------------------------------------------------------------------------
class SparseSolver:
    def __init__(self, A, use_approx=False):
        """
        Initialize a sparse solver for matrix A.

        If PETSc is available, build a KSP solver; otherwise, use SciPy's splu for direct solves.

        Parameters:
        - A: scipy.sparse matrix (preferably CSR or convertible to CSR)
        - use_approx: bool
            If using PETSc, whether to use iterative approximate solves (GMRES+Hypre).
        """
        # The LinearOperator rmatvec convention is an adjoint action.  For real
        # matrices this is the transpose, while complex matrices need A^*.
        v = np.random.rand(A.shape[0],).astype(A.dtype, copy=False)
        A_adj = A.getH() if np.iscomplexobj(A.data) else A.T
        self.is_self_adjoint = np.linalg.norm(A @ v - A_adj @ v) < 1e-12
        self.N = A.shape[0]
        self.dtype = A.dtype
        self.is_complex = np.iscomplexobj(A.data)

        self.mumps_has_complex = mumps_supports_complex()
        self.use_mumps = mumps_imported and ((not self.is_complex) or self.mumps_has_complex)
        self.use_petsc = petsc_imported
        self.petsc_has_complex = petsc_supports_complex()
        self.use_approx = use_approx
        self.backend = None
        self.ksp_adj = None
        self.storage_bytes = None

        tic = time()
        if self.use_mumps:
            # Build solver on A
            self.backend = 'mumps'
            self.ksp = setup_mumps(A)
            if not self.is_self_adjoint:
                self.ksp_adj = setup_mumps(A.getH() if self.is_complex else A.T)
        elif self.use_petsc and ((not self.is_complex) or self.petsc_has_complex):
            # Build KSP solver on A
            self.backend = 'petsc'
            self.ksp = setup_ksp(A, use_approx)
            if not self.is_self_adjoint:
                self.ksp_adj = setup_ksp(A.getH() if self.is_complex else A.T, use_approx)
        else:
            # Fallback to SciPy’s LU decomposition (CSC format for efficiency)
            self.backend = 'superlu'
            self.ksp = splu(A.tocsc())
            self.storage_bytes = (
                self.ksp.L.data.nbytes + self.ksp.L.indices.nbytes + self.ksp.L.indptr.nbytes
                + self.ksp.U.data.nbytes + self.ksp.U.indices.nbytes + self.ksp.U.indptr.nbytes
            )
        self.build_time = time() - tic

        # Perform a test solve to gauge performance (solve A x = A v)
        rhs = A @ v
        tic = time()
        _ = self.solve_op.matvec(rhs)
        toc = time() - tic

        if self.backend == 'petsc':
            niter = self.ksp.getIterationNumber()
            # Optionally, print or log timing and iteration counts here
            # print(f"use_approx={use_approx}, solve time={toc:.2e}, relerr={np.linalg.norm(res-v):.2e}, iter={niter}")

    def _solve_mumps(self, solver, rhs):
        rhs = np.asarray(rhs)
        if rhs.ndim == 1:
            return np.asarray(solver.solve(rhs.copy()))

        sol = np.zeros(rhs.shape, dtype=np.result_type(rhs.dtype, self.dtype))
        for j in range(rhs.shape[-1]):
            sol[:, j] = np.asarray(solver.solve(rhs[:, j].copy()))
        return sol

    def _solve_backend(self, rhs, adjoint=False):
        rhs = np.asarray(rhs)
        if (not self.is_complex) and np.iscomplexobj(rhs):
            return self._solve_backend(rhs.real, adjoint=adjoint) + 1j * self._solve_backend(rhs.imag, adjoint=adjoint)

        if self.backend == 'mumps':
            solver = self.ksp_adj if adjoint and self.ksp_adj is not None else self.ksp
            return self._solve_mumps(solver, rhs)

        if self.backend == 'petsc':
            solver = self.ksp_adj if adjoint and self.ksp_adj is not None else self.ksp
            if rhs.ndim == 1:
                return get_vecsolve(solver)(rhs)
            return get_matsolve(solver, self.use_approx)(rhs)

        trans = 'H' if (adjoint and self.is_complex) else ('T' if adjoint else 'N')
        return self.ksp.solve(rhs, trans=trans)

    @property
    def solve_op(self):
        """
        Return a scipy.sparse.linalg.LinearOperator that applies A^{-1} (and its transpose if needed).

        Based on whether PETSc is used and whether A is symmetric, provide appropriate
        matvec, rmatvec (for A^T), matmat (for dense multi-RHS), and rmatmat.

        Returns:
        - LinearOperator of shape (N, N)
        """
        return LinearOperator(
            shape=(self.N, self.N),
            dtype=self.dtype,
            matvec=lambda x: self._solve_backend(x, adjoint=False),
            rmatvec=lambda x: self._solve_backend(x, adjoint=not self.is_self_adjoint),
            matmat=lambda X: self._solve_backend(X, adjoint=False),
            rmatmat=lambda X: self._solve_backend(X, adjoint=not self.is_self_adjoint),
        )


# ------------------------------------------------------------------------------------
# CSRBuilder: helper to build a CSR matrix by accumulating COO data
# ------------------------------------------------------------------------------------
class CSRBuilder:
    def __init__(self, M, N, nnz, dtype=float):
        """
        Initialize a builder for an M×N sparse matrix with at most nnz nonzeros.

        Parameters:
        - M: int, number of rows
        - N: int, number of columns
        - nnz: int, maximum number of nonzero entries expected
        - dtype: NumPy dtype for the stored values
        """
        self.M = M
        self.N = N
        # Preallocate arrays to hold row indices, column indices, and values
        self.row = np.zeros(nnz, dtype=int)
        self.col = np.zeros(nnz, dtype=int)
        self.data = np.zeros(nnz, dtype=dtype)
        # Accumulator points to the next free position in the arrays
        self.acc = 0

    def add_data(self, mat):
        """
        Add nonzero entries of a scipy.sparse matrix mat to this builder.

        Parameters:
        - mat: scipy.sparse matrix (any format); will be converted to COO internally
        """
        # Convert to COO format for easy access to row, col, data arrays
        mat = mat.tocoo()
        ndata = mat.row.shape[0]
        # Ensure we don't overflow the preallocated arrays
        assert self.acc + ndata <= self.data.shape[0]

        # Copy row indices, col indices, and values into the builder arrays
        self.row[self.acc : self.acc + ndata] = mat.row
        self.col[self.acc : self.acc + ndata] = mat.col
        self.data[self.acc : self.acc + ndata] = mat.data
        self.acc += ndata

    def tocsr(self):
        """
        Finalize the builder and return a scipy.sparse.csr_matrix.

        Only the first self.acc entries of row, col, data are used.

        Returns:
        - csr: scipy.sparse.csr_matrix of shape (M, N)
        """
        coo = coo_matrix((self.data[:self.acc],
                          (self.row[:self.acc], self.col[:self.acc])),
                         shape=(self.M, self.N))
        return coo.tocsr()
