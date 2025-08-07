import numpy as np
from scipy.sparse.linalg import LinearOperator, splu
from scipy.sparse import csr_matrix, coo_matrix
from time import time

# Try to import PETSc; if successful, we can use its KSP solvers.
try:
    from petsc4py import PETSc
    petsc_imported = True
    print("PETSC IMPORTED")
except ImportError:
    print("PETSC NOT IMPORTED")
    petsc_imported = False


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
        px = PETSc.Vec().createWithArray(np.zeros(b.shape))
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
        res = np.zeros(B.shape)
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
        # Test if A is symmetric by checking A v ?= A^T v for a random vector v
        v = np.random.rand(A.shape[0],)
        self.is_symmetric = np.linalg.norm(A @ v - A.T @ v) < 1e-12
        self.N = A.shape[0]

        self.use_petsc = petsc_imported
        self.use_approx = use_approx

        if self.use_petsc:
            # Build KSP solver on A
            self.ksp = setup_ksp(A, use_approx)
            if not self.is_symmetric:
                # If A is nonsymmetric, build a separate KSP on A^T for transpose solves
                self.ksp_adj = setup_ksp(A.T, use_approx)
        else:
            # Fallback to SciPy’s LU decomposition (CSC format for efficiency)
            self.ksp = splu(A.tocsc())

        # Perform a test solve to gauge performance (solve A x = A v)
        rhs = A @ v
        tic = time()
        _ = self.solve_op.matvec(rhs)
        toc = time() - tic

        if self.use_petsc:
            niter = self.ksp.getIterationNumber()
            # Optionally, print or log timing and iteration counts here
            # print(f"use_approx={use_approx}, solve time={toc:.2e}, relerr={np.linalg.norm(res-v):.2e}, iter={niter}")

    @property
    def solve_op(self):
        """
        Return a scipy.sparse.linalg.LinearOperator that applies A^{-1} (and its transpose if needed).

        Based on whether PETSc is used and whether A is symmetric, provide appropriate
        matvec, rmatvec (for A^T), matmat (for dense multi-RHS), and rmatmat.

        Returns:
        - LinearOperator of shape (N, N)
        """
        if self.use_petsc and self.is_symmetric:
            # Symmetric case: matvec and rmatvec are same
            return LinearOperator(
                shape=(self.N, self.N),
                matvec=get_vecsolve(self.ksp),
                rmatvec=get_vecsolve(self.ksp),
                matmat=get_matsolve(self.ksp, self.use_approx),
                rmatmat=get_matsolve(self.ksp, self.use_approx)
            )

        elif self.use_petsc and not self.is_symmetric:
            # Nonsymmetric: use ksp for matvec, ksp_adj for rmatvec
            return LinearOperator(
                shape=(self.N, self.N),
                matvec=get_vecsolve(self.ksp),
                rmatvec=get_vecsolve(self.ksp_adj),
                matmat=get_matsolve(self.ksp, self.use_approx),
                rmatmat=get_matsolve(self.ksp_adj, self.use_approx)
            )

        elif not self.use_petsc and self.is_symmetric:
            # Use SciPy LU for both matvec and rmatvec (direct symmetric solve)
            return LinearOperator(
                shape=(self.N, self.N),
                matvec=lambda x: self.ksp.solve(x),
                rmatvec=lambda x: self.ksp.solve(x),
                matmat=lambda X: self.ksp.solve(X),
                rmatmat=lambda X: self.ksp.solve(X)
            )

        else:
            # SciPy LU for nonsymmetric: use transpose solve for rmatvec
            return LinearOperator(
                shape=(self.N, self.N),
                matvec=lambda x: self.ksp.solve(x),
                rmatvec=lambda x: self.ksp.solve(x, trans='T'),
                matmat=lambda X: self.ksp.solve(X),
                rmatmat=lambda X: self.ksp.solve(X, trans='T')
            )


# ------------------------------------------------------------------------------------
# CSRBuilder: helper to build a CSR matrix by accumulating COO data
# ------------------------------------------------------------------------------------
class CSRBuilder:
    def __init__(self, M, N, nnz):
        """
        Initialize a builder for an M×N sparse matrix with at most nnz nonzeros.

        Parameters:
        - M: int, number of rows
        - N: int, number of columns
        - nnz: int, maximum number of nonzero entries expected
        """
        self.M = M
        self.N = N
        # Preallocate arrays to hold row indices, column indices, and values
        self.row = np.zeros(nnz, dtype=int)
        self.col = np.zeros(nnz, dtype=int)
        self.data = np.zeros(nnz)
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
        assert self.acc + ndata < self.data.shape[0]

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
