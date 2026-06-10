import torch  # Used for tensor operations
import numpy as np  # For numerical operations, especially those not directly supported by PyTorch
import sys  # System-specific parameters and functions
import scipy.sparse as sp

# Importing parent class:
from hpsmultidomain.abstract_hps_solver import AbstractHPSSolver
import hpsmultidomain.pdo as pdo

# Importing necessary components for sparse matrix operations
from scipy.sparse import kron, diags, block_diag, eye as speye, hstack as sp_hstack, vstack as sp_vstack
import scipy.sparse.linalg as spla  # For sparse linear algebra operations
from time import time  # For timing operations
torch.set_default_dtype(torch.double)  # Setting default tensor type to double for precision

# Import custom modules for handling multidomain discretization and partial differential operators
import hpsmultidomain.hps_multidomain_disc
import hpsmultidomain.pdo
from functools import reduce  # For performing cumulative operations
import scipy.sparse.linalg as sla  # For sparse linear algebra operations, alternative variable
from hpsmultidomain.built_in_funcs import uu_dir_func_greens
from hpsmultidomain.sparse_utils import SparseSolver

# Attempting to import the python-mumps library for parallel computation, handling failure gracefully
try:
    import mumps
    mumps_available = True
except ImportError:
    mumps_available = False
    print("python-mumps not available")

# Attempting to import the PETSc library for parallel computation, handling failure gracefully
try:
    from petsc4py import PETSc
    # Set use of BLR compression and the precision of the dropping parameter.
    # Looks like these are better left unused which makes sense, the DtNs aren't low rank.
    #PETSc.Options()['mat_mumps_icntl_35'] = 3
    #PETSc.Options()['mat_mumps_cntl_7']   = 1e-6

    # Set relative threshold for numerical pivoting:
    #PETSc.Options()['mat_mumps_cntl_1'] = 0.0
    # Set matrix permutation. 7 is automatically decided, 1-6 are different choices.
    #PETSc.Options()['mat_mumps_icntl_6']  = 7
    #PETSc.Options()['mat_mumps_icntl_8']  = 77 # Scaling strategy, set to be automatically picked
    #PETSc.Options()['mat_mumps_icntl_10'] = 0 # No iterative refinement
    #PETSc.Options()['mat_mumps_icntl_12'] = 1 # Ordering strategy with icntl 6
    #PETSc.Options()['mat_mumps_icntl_13'] = 0 # Parallel factorization of root node
    petsc_available = True
except ImportError:
    petsc_available = False
    print("petsc not available")
    
def torch_setdiff1d(vec1,vec2):
    device = vec1.device
    return torch.tensor(np.setdiff1d(vec1.numpy(), vec2.numpy()),dtype=int,device=device)

def to_torch_csr(A, device=torch.device('cpu')):
    """
    Converts a SciPy sparse matrix to a PyTorch sparse CSR tensor.
    
    Parameters:
    - A: SciPy sparse matrix.
    - device: The torch device where the tensor will be allocated.
    
    Returns:
    - A PyTorch sparse CSR tensor representation of A.
    """
    A.sort_indices()
    return torch.sparse_csr_tensor(A.indptr, A.indices, A.data, size=(A.shape[0], A.shape[1]), device=device)

def apply_sparse_lowmem(A, I, J, v, transpose=False):
    """
    Applies a sparse matrix to a vector or batch of vectors with minimal memory usage,
    using only specified indices for non-zero entries.
    
    Parameters:
    - A: The sparse matrix.
    - I: Indices to extract from the resulting vector after applying A.
    - J: Indices of non-zero entries in the input vector(s) v.
    - v: The vector(s) to be multiplied.
    - transpose: If True, apply the transpose of A instead.
    
    Returns:
    - The resulting vector after applying A (or A^T if transpose=True) to v, extracting indices I.
    """
    if isinstance(v, torch.Tensor):
        v_np = v.detach().cpu().numpy()
    else:
        v_np = np.asarray(v)

    if v_np.ndim == 1:
        v_np = v_np[:, None]

    vec_full = np.zeros((A.shape[1], v_np.shape[-1]), dtype=v_np.dtype)
    vec_full[J] = v_np
    vec_full = A.T @ vec_full if transpose else A @ vec_full
    return torch.from_numpy(np.asarray(vec_full[I]))

# Domain_Driver class for setting up and solving the discretized PDE
class Domain_Driver(AbstractHPSSolver):
    def __init__(
        self,
        box_geom,
        pdo_op,
        kh,
        a,
        p=12,
        d=2,
        periodic_bc=False,
        statically_condense=True,
        use_iti_maps=False,
        impedance_eta=None,
    ):
        """
        Initializes the domain and discretization for solving a PDE.
        
        Parameters:
        - box_geom: Geometry of the computational domain.
        - pdo_op: The partial differential operator to be solved.
        - kh: Wave number or parameter in the differential equation.
        - a: Characteristic length for HPS.
        - p: Polynomial degree for HPS discretization (ignored for FD).
        - d: dimension of domain for HPS
        - periodic_bc: Boolean indicating if periodic boundary conditions are applied.
        - statically_condense: If True, eliminate leaf interiors before assembling the reduced system.
        - use_iti_maps: If True, assemble Helmholtz leaf maps as incoming-to-outgoing impedance maps.
        - impedance_eta: Constant impedance parameter eta; defaults to kh for ItI maps.
        """
        self.d = d
        self.kh = kh
        self.periodic_bc  = periodic_bc
        self.statically_condense = statically_condense
        self.use_iti_maps = use_iti_maps
        self.impedance_eta = kh if impedance_eta is None else impedance_eta
        self.box_geometry = box_geom # The full BoxGeometry object
        self.box_geom     = self.box_geometry.bounds.T # The array itself
        self.hps_disc(self.box_geom,a,p,d,pdo_op,periodic_bc)

    ################### Required functions for parent class AbstractHPSSolver #####################

    #################################################
    # Abstract properties defining essential data
    #################################################

    @property
    def geom(self):
        """
        Geometry object containing domain bounds and, optionally, a parameterization map.
        Must have attribute `bounds` of shape (2, ndim).
        """
        return self.box_geometry

    # TODO: one or both of these might have to be I_unique indexed:

    @property
    def XX(self):
        """
        Flattened array of coordinates for all boundary (exterior) nodes:
        """
        return self.XX_active

    @property
    def XXfull(self):
        """
        Flattened array of coordinates for all discretization (including interiors) nodes:
        """
        return self._XXfull
       
    @property 
    def p(self):
        """
        Polynomial degree used in each patch (number of Chebyshev nodes per direction).
        """
        return self.hps.p

    @property
    def Ji(self):
        """
        Index array for interior (duplicated interface) points in the global boundary ordering.
        """
        return self._Ji # aka self.I_Ctot

    @property
    def Jx(self):
        """
        Index array for unique exterior (non-duplicated) boundary points.
        """
        return self._Jx # aka self.I_Xtot

    @property
    def npoints_dim(self):
        """
        Total number of Chebyshev points per dimension (npan_dim * p for each dimension).
        """
        return self.hps.n * self.p

    @property
    def npan_dim(self):
        return self.hps.n

    #################################################
    # Abstract properties defining Schur complement blocks
    #################################################

    @property
    def Aii(self):
        """
        Sparse matrix block coupling interior‐interior (duplicated interface) degrees of freedom.
        """
        return self.A_CC

    @property
    def Aix(self):
        """
        Sparse matrix block coupling interior (duplicated interface) to unique exterior DOFs.
        """
        return self.A_CX

    @property
    def Axx(self):
        """
        Sparse matrix block coupling unique exterior DOFs to themselves.
        """
        return self.A_XX

    @property
    def Axi(self):
        """
        Sparse matrix block coupling unique exterior DOFs to interior (duplicated interface) DOFs.
        """
        return self.A_XC

    #################################################
    # Utilities to solve the PDE
    #################################################

    # This needs to be interior only for n_int on ff_body... figure out how to reduce it
    # Also need sol_tot to be just interiors (no global bdry or no box boundary?)
    def solve_dir_full(self, uu_dir, ff_body=None):
        uu_dir_func = uu_dir if callable(uu_dir) else (lambda xx: uu_dir)
        uu_dir_vec = None if callable(uu_dir) else uu_dir
        ff_body_func = ff_body if callable(ff_body) else None
        ff_body_vec = None if callable(ff_body) else ff_body

        sol_tot, _, _, _, _, _, _, _ = self.solve(
            uu_dir_func,
            uu_dir_vec=uu_dir_vec,
            ff_body_func=ff_body_func,
            ff_body_vec=ff_body_vec,
        )

        return sol_tot

    def verify_discretization(self, kh):
        if self.use_iti_maps:
            raise NotImplementedError("verify_discretization currently uses the DtN interface solve.")
        if not self.statically_condense:
            raise NotImplementedError("verify_discretization currently uses the condensed interface solve.")
        # 1) Possibly map XX through a parameterization, if geometry defines one
        if hasattr(self.geom, 'parameter_map'):
            XX_mapped = self.geom.parameter_map(self.XX)
        else:
            XX_mapped = self.XX

        # 2) Evaluate known Green's function at boundary points, with a source placed far outside domain
        uu = uu_dir_func_greens(self.d, XX_mapped, kh, center=self.geom.bounds[1] + 12)

        # 3) Extract boundary values (unique exterior) and interior true values
        #uu_sol = self.solve_dir_full(uu[self.Jx()])
        uu_sol, _, ff_body = self.solve_helper_blackbox(uu[self.Jx],uu_dir_vec=uu[self.Jx])
        uu_true = uu[self.Ji]

        # 4) Compute error and relative error
        err = np.linalg.norm(uu_sol - uu_true, ord=2)
        relerr = err / np.linalg.norm(uu_true, ord=2)
        return relerr

    #################################################
    # Setup and retrieve a solver for the Aii block
    #################################################

    def setup_solver_Aii(self, solve_op=None, use_approx=False):
        """
        Initialize a linear solver for the Aii block. If solve_op is provided,
        use it directly. Otherwise, create a SparseSolver on Aii.

        Parameters:
        - solve_op:   Optionally, a pre‐constructed LinearOperator for Aii^{-1}.
        - use_approx: If True and PETSc is available, use an approximate iterative solver.
        """
        if solve_op is None:
            self.sparse_solver = SparseSolver(self.Aii, use_approx=use_approx)
            self.solve_op = self.sparse_solver.solve_op
        else:
            self.sparse_solver = None
            self.solve_op = solve_op

        self.mumps_LU = None
        self.petsc_LU = None
        self.superLU = None
        if self.sparse_solver is not None:
            if self.sparse_solver.backend == 'mumps':
                self.mumps_LU = self.sparse_solver.ksp
            elif self.sparse_solver.backend == 'petsc':
                self.petsc_LU = self.sparse_solver.ksp
            else:
                self.superLU = self.sparse_solver.ksp

    @property
    def solver_Aii(self):
        """
        Return the LinearOperator that applies Aii^{-1}. If not yet set up,
        call setup_solver_Aii with default parameters.
        """
        if not hasattr(self, 'solve_op'):
            self.setup_solver_Aii()
        return self.solve_op

            
    ############################### HPS discretization and panel split #####################
    def hps_disc(self,box_geom,a,p,d,pdo_op,periodic_bc):

        if isinstance(a, (int, float)):
            a = np.array([a] * d)

        if isinstance(p, (int)):
            p = [p] * d

        p = np.array(p)

        assert p.all() > 0

        HPS_multi = hpsmultidomain.hps_multidomain_disc.HPS_Multidomain(pdo_op,box_geom,a,p,d, periodic_bc=periodic_bc)

        self.hps = HPS_multi

        self.XX_active  = self.hps.xx_active
        self.ntot       = self.XX_active.shape[0]
        tol             = 0.01 * self.hps.hmin # Adding a tolerance to avoid potential numerical error

        if d==2:
            if periodic_bc:
                I_dir = torch.where((self.XX_active[:,1] < self.box_geom[1,0] + tol)
                                | (self.XX_active[:,1] > self.box_geom[1,1] - tol))[0]

                I_dir2 = torch.where((self.XX_active[:,0] > self.box_geom[0,1] - tol)
                                | (self.XX_active[:,1] < self.box_geom[1,0] + tol)
                                | (self.XX_active[:,1] > self.box_geom[1,1] - tol))[0]

                self.I_Xtot = I_dir
                self.I_Ctot = torch.sort(torch_setdiff1d(torch.arange(self.ntot), I_dir2))[0]
            else:
                I_dir = torch.where((self.XX_active[:,0] < self.box_geom[0,0] + tol)
                                | (self.XX_active[:,0] > self.box_geom[0,1] - tol)
                                | (self.XX_active[:,1] < self.box_geom[1,0] + tol)
                                | (self.XX_active[:,1] > self.box_geom[1,1] - tol))[0]
            
                self.I_Xtot = I_dir
                self.I_Ctot = torch.sort(torch_setdiff1d(torch.arange(self.ntot), self.I_Xtot))[0]

        else: # d==3
            if periodic_bc:
                I_dir = torch.where((self.XX_active[:,1] < self.box_geom[1,0] + tol)
                                | (self.XX_active[:,1] > self.box_geom[1,1] - tol)
                                | (self.XX_active[:,2] < self.box_geom[2,0] + tol)
                                | (self.XX_active[:,2] > self.box_geom[2,1] - tol))[0]

                I_dir2 = torch.where((self.XX_active[:,0] > self.box_geom[0,1] - tol)
                                | (self.XX_active[:,1] < self.box_geom[1,0] + tol)
                                | (self.XX_active[:,1] > self.box_geom[1,1] - tol)
                                | (self.XX_active[:,2] < self.box_geom[2,0] + tol)
                                | (self.XX_active[:,2] > self.box_geom[2,1] - tol))[0]

                self.I_Xtot = I_dir
                self.I_Ctot = torch.sort(torch_setdiff1d(torch.arange(self.ntot), I_dir2))[0]
            else:
                I_dir = torch.where((self.XX_active[:,0] < self.box_geom[0,0] + tol)
                                | (self.XX_active[:,0] > self.box_geom[0,1] - tol)
                                | (self.XX_active[:,1] < self.box_geom[1,0] + tol)
                                | (self.XX_active[:,1] > self.box_geom[1,1] - tol)
                                | (self.XX_active[:,2] < self.box_geom[2,0] + tol)
                                | (self.XX_active[:,2] > self.box_geom[2,1] - tol))[0]
            
                self.I_Xtot = I_dir
                self.I_Ctot = torch.sort(torch_setdiff1d(torch.arange(self.ntot), self.I_Xtot))[0]

        self.I_Xtot_in_unique = self.hps.I_unique[self.I_Xtot]
        # Note that I_Xtot and I_Ctot are both out of all XX, not just the unique
        # boundaries.

        self._XXfull = torch.reshape(self.hps.grid_xx, (self.hps.grid_xx.shape[0] * self.hps.grid_xx.shape[1], -1))

        self._Ji = self.I_Ctot
        self._Jx = self.I_Xtot
            
    
    def build_superLU(self,verbose):
        """
        Constructs the sparse system using superLU from scipy.sparse. Used if PETSc is not avaialble.
        """
        info_dict = dict()
        try:
            tic = time()
            # SuperLU expects CSC input; convert once here to avoid repeated warnings.
            LU = sla.splu(self.A_CC.tocsc())          
            toc_superLU = time() - tic
            if (verbose):
                print("SUPER LU BUILD SUMMARY")
            mem_superLU  = LU.L.data.nbytes + LU.L.indices.nbytes + LU.L.indptr.nbytes
            mem_superLU += LU.U.data.nbytes + LU.U.indices.nbytes + LU.U.indptr.nbytes
            stor_superLU = mem_superLU/1e9
            if (verbose):
                print("\t--time superLU build = %5.2f seconds"\
                      % (toc_superLU))
                print("\t--total memory = %5.2f GB"\
                      % (stor_superLU))

            self.superLU    = LU
            #self.solver_Aii = self.superLU

            info_dict['toc_build_superLU'] = toc_superLU
            info_dict['toc_build_blackbox']= toc_superLU
            info_dict['solver_type']       = 'scipy_superLU'
            
            info_dict['mem_build_superLU'] = stor_superLU
        except:
            print("SuperLU had an error.")
        return info_dict
    
    
    def build_petsc(self,solvertype,verbose):
        """
        Constructs the sparse system using MUMPS from petsc4py. Preferred.
        """
        #
        # Here we will try setting different MUMPS parameters to improve speed
        #

        # Set the blocksize using icntl(15) to the size of a face (q**2)
        #if self.d==3:
        #    PETSc.Options()['mat_mumps_icntl_15'] = -self.hps.q**2

        info_dict = dict()
        tmp = self.A_CC
        pA = PETSc.Mat().createAIJ(tmp.shape, csr=(tmp.indptr,tmp.indices,tmp.data),comm=PETSc.COMM_WORLD)
        
        ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
        ksp.setOperators(pA)
        ksp.setType('preonly')
        
        ksp.getPC().setType('lu')
        ksp.getPC().setFactorSolverType(solvertype)
        
        px = PETSc.Vec().createWithArray(np.ones(tmp.shape[0]),comm=PETSc.COMM_WORLD)
        pb = PETSc.Vec().createWithArray(np.ones(tmp.shape[0]),comm=PETSc.COMM_WORLD)

        #print("Initiated the ksp operator, still need to do a solve to break it in")        
        tic = time()
        ksp.solve(pb, px)
        toc_build = time() - tic

        if (verbose):
            print("\t--time for %s build through petsc = %5.2f seconds"\
                  % (solvertype,toc_build))
               
        info_dict['toc_build_blackbox']   = toc_build
        info_dict['solver_type']          = "petsc_"+solvertype
        
        self.petsc_LU   = ksp
        #self.solver_Aii = self.petsc_LU

        return info_dict

    def build_mumps(self,verbose):
        """
        Constructs the sparse system using mumps from python-mumps. Referred to access MUMPS.
        """
        info_dict = dict()
        try:
            tic = time()
            inst = mumps.Context()
            inst.analyze(self.A_CC, ordering='pord')
            mumps_LU = inst.factor(self.A_CC)
            toc_mumps_LU = time() - tic
            if (verbose):
                print("\t--time superLU build = %5.2f seconds"\
                      % (toc_mumps_LU))

            self.mumps_LU    = mumps_LU
            #self.solver_Aii = self.superLU

            info_dict['toc_build_superLU'] = toc_mumps_LU
            info_dict['toc_build_blackbox']= toc_mumps_LU
            info_dict['solver_type']       = 'python-mumps'
        except:
            print("python-mumps had an error.")
        return info_dict

    def _require_uncondensed_supported(self):
        if self.hps.interpolate:
            raise NotImplementedError(
                "statically_condense=False is currently supported only for non-interpolating square/cube cases."
            )

    def _get_uncondensed_indices(self):
        self._require_uncondensed_supported()

        nboxes = int(self.hps.nboxes)
        nint_leaf = int(np.prod(self.hps.p - 2))
        size_ext = len(self.hps.H.JJ.Jx)
        block_size = nint_leaf + size_ext

        I_int = np.concatenate(
            [np.arange(box * block_size, box * block_size + nint_leaf) for box in range(nboxes)]
        )

        def lift_surface_inds(surface_inds):
            surface_inds = np.asarray(surface_inds, dtype=int)
            box = surface_inds // size_ext
            local = surface_inds % size_ext
            return box * block_size + nint_leaf + local

        I_copy1 = lift_surface_inds(self.hps.I_copy1.detach().cpu().numpy())
        I_copy2 = lift_surface_inds(self.hps.I_copy2.detach().cpu().numpy())
        I_ext = lift_surface_inds(self.I_Xtot_in_unique.detach().cpu().numpy())
        return I_int, I_copy1, I_copy2, I_ext, nint_leaf, size_ext

    def _require_iti_supported(self):
        if not self.use_iti_maps:
            return
        if not self.statically_condense:
            raise NotImplementedError("ItI maps are currently supported only with statically_condense=True.")
        if self.kh == 0:
            raise NotImplementedError("ItI maps are currently supported only for Helmholtz problems.")
        if hasattr(self.geom, 'parameter_map'):
            raise NotImplementedError("ItI maps are currently supported only for non-mapped square/cube geometries.")
        if self.periodic_bc:
            raise NotImplementedError("ItI maps are currently unsupported with periodic boundary conditions.")
        if self.hps.interpolate:
            raise NotImplementedError("ItI maps are currently supported only for non-interpolating square/cube cases.")

    def _get_surface_device(self):
        if self.sparse_assembly == 'reduced_gpu':
            return torch.device('cuda')
        return torch.device('cpu')

    def _setup_iti_partitions(self):
        if hasattr(self, 'iti_leaf_data'):
            return

        self._require_iti_supported()

        size_ext = len(self.hps.H.JJ.Jx)
        nboxes = int(self.hps.nboxes)
        total_surface = nboxes * size_ext
        tol = 0.01 * self.hps.hmin
        xx_ext = self.hps.xx_ext

        if self.d == 2:
            is_exterior = (
                (xx_ext[:, 0] < self.box_geom[0, 0] + tol)
                | (xx_ext[:, 0] > self.box_geom[0, 1] - tol)
                | (xx_ext[:, 1] < self.box_geom[1, 0] + tol)
                | (xx_ext[:, 1] > self.box_geom[1, 1] - tol)
            )
        else:
            is_exterior = (
                (xx_ext[:, 0] < self.box_geom[0, 0] + tol)
                | (xx_ext[:, 0] > self.box_geom[0, 1] - tol)
                | (xx_ext[:, 1] < self.box_geom[1, 0] + tol)
                | (xx_ext[:, 1] > self.box_geom[1, 1] - tol)
                | (xx_ext[:, 2] < self.box_geom[2, 0] + tol)
                | (xx_ext[:, 2] > self.box_geom[2, 1] - tol)
            )

        I_Xtot_dup = torch.where(is_exterior)[0]
        I_Ctot_dup = torch.where(~is_exterior)[0]

        expected_exterior = self.I_Xtot_in_unique.detach().cpu()
        if not torch.equal(torch.sort(I_Xtot_dup).values, torch.sort(expected_exterior).values):
            raise ValueError("Failed to match duplicated exterior surface nodes for ItI assembly.")

        self.I_Xtot_dup = I_Xtot_dup
        self.I_Ctot_dup = I_Ctot_dup

        dup_to_int = -np.ones(total_surface, dtype=int)
        dup_to_int[I_Ctot_dup.detach().cpu().numpy()] = np.arange(len(I_Ctot_dup))
        dup_to_ext = -np.ones(total_surface, dtype=int)
        dup_to_ext[self.I_Xtot_in_unique.detach().cpu().numpy()] = np.arange(len(self.I_Xtot))

        local_surface = np.arange(total_surface, dtype=int).reshape(nboxes, size_ext)
        self.iti_leaf_data = []
        for box in range(nboxes):
            dup_inds = local_surface[box]
            loc_int_mask = dup_to_int[dup_inds] >= 0
            loc_int = np.flatnonzero(loc_int_mask)
            loc_ext = np.flatnonzero(~loc_int_mask)
            self.iti_leaf_data.append(
                dict(
                    dup_inds=dup_inds,
                    local_int=loc_int,
                    local_ext=loc_ext,
                    global_int=dup_to_int[dup_inds[loc_int]],
                    global_ext=dup_to_ext[dup_inds[loc_ext]],
                )
            )

        copy1 = self.hps.I_copy1.detach().cpu().numpy()
        copy2 = self.hps.I_copy2.detach().cpu().numpy()
        self.iti_copy1 = dup_to_int[copy1]
        self.iti_copy2 = dup_to_int[copy2]
        if np.any(self.iti_copy1 < 0) or np.any(self.iti_copy2 < 0):
            raise ValueError("Failed to identify internal copy indices for ItI assembly.")

        nint = len(I_Ctot_dup)
        rows = np.concatenate((self.iti_copy1, self.iti_copy2))
        cols = np.concatenate((self.iti_copy2, self.iti_copy1))
        data = np.ones(rows.shape[0], dtype=np.complex128)
        self.iti_swap = sp.csr_matrix((data, (rows, cols)), shape=(nint, nint))
        self.iti_ndirected = nint
        self.iti_nphys = len(self.iti_copy1)
        self.iti_nint = self.iti_ndirected
        self.iti_size_ext = size_ext

    def _compute_iti_body_flux(self, ff_body_func=None, ff_body_vec=None):
        if (ff_body_func is None) and (ff_body_vec is None):
            return None

        device = self._get_surface_device()
        body_flux = self.hps.get_DtNs(device, mode='reduce_body', ff_body_func=ff_body_func, ff_body_vec=ff_body_vec)
        body_flux = -body_flux.detach().cpu().numpy().astype(np.complex128)
        return body_flux

    def _build_iti_system(self, device, verbose):
        self._setup_iti_partitions()
        eta = self.impedance_eta

        tic = time()
        DtN_loc = self.hps.get_DtNs(device, 'build').detach().cpu().numpy().astype(np.complex128)
        toc_DtN = time() - tic

        dir_blocks = []
        rows_dir = []
        cols_dir = []
        vals_dir = []

        for box, leaf_data in enumerate(self.iti_leaf_data):
            loc_int = leaf_data['local_int']
            loc_ext = leaf_data['local_ext']
            glob_int = leaf_data['global_int']
            glob_ext = leaf_data['global_ext']

            T = DtN_loc[box]
            Tcc = T[np.ix_(loc_int, loc_int)]
            Tcx = T[np.ix_(loc_int, loc_ext)]

            Icc = np.eye(len(loc_int), dtype=np.complex128)
            # Local ItI convention: for flux = Tcc u_int + Tcx u_ext,
            # incoming impedance is g = i eta u_int - flux and outgoing
            # impedance is h = i eta u_int + flux.  Thus R_dir and B_dir
            # give h = R_dir g + B_dir u_ext on each leaf, and the global
            # Schur complement enforces incoming data on one side to equal
            # outgoing data from the neighboring side.
            M_dir_inv = np.linalg.inv(1j * eta * Icc - Tcc)
            R_dir = (Tcc + 1j * eta * Icc) @ M_dir_inv
            B_dir = 2j * eta * (M_dir_inv @ Tcx)

            leaf_data['dir_M_inv'] = M_dir_inv
            leaf_data['dir_T_cc'] = Tcc
            leaf_data['dir_T_cx'] = Tcx
            leaf_data['dir_R'] = R_dir
            leaf_data['dir_B'] = B_dir
            leaf_data['dir_body_map'] = 2j * eta * M_dir_inv

            dir_blocks.append(sp.csr_matrix(R_dir))

            if B_dir.size > 0:
                row_idx, col_idx = np.meshgrid(glob_int, glob_ext, indexing='ij')
                rows_dir.extend(row_idx.ravel().tolist())
                cols_dir.extend(col_idx.ravel().tolist())
                vals_dir.extend(B_dir.ravel().tolist())

        tic = time()
        ndirected = self.iti_ndirected
        next_bdry = len(self.I_Xtot)
        R_dir = sp.block_diag(dir_blocks, format='csr')
        B_dir = sp.csr_matrix((vals_dir, (rows_dir, cols_dir)), shape=(ndirected, next_bdry), dtype=np.complex128)

        # Directed ItI equations are g = P(R g + B u_ext), where P swaps the
        # two leaf-side traces on each physical interface.  This is the doubled
        # sparse HPS formulation: each leaf keeps its own incoming trace copy.
        self.iti_R_dir = R_dir
        self.iti_B_dir = B_dir

        Idirected = speye(ndirected, format='csr', dtype=np.complex128)
        self.A_CC = (Idirected - self.iti_swap @ R_dir).tocsr()
        self.A_CX = -(self.iti_swap @ B_dir).tocsr()
        self.A_XC = sp.csr_matrix((next_bdry, ndirected), dtype=np.complex128)
        self.A_XX = sp.csr_matrix((next_bdry, next_bdry), dtype=np.complex128)
        toc_sparse = time() - tic

        t_dict = dict()
        t_dict['toc_DtN'] = toc_DtN
        t_dict['toc_sparse'] = toc_sparse
        return t_dict

    def _get_iti_body_local(self, ff_body_func=None, ff_body_vec=None):
        body_flux = self._compute_iti_body_flux(ff_body_func=ff_body_func, ff_body_vec=ff_body_vec)
        if body_flux is None:
            return None

        rhs_local = np.zeros((self.iti_ndirected, body_flux.shape[-1]), dtype=np.complex128)
        for box, leaf_data in enumerate(self.iti_leaf_data):
            rows = leaf_data['global_int']
            loc_int = leaf_data['local_int']
            rhs_local[rows] = leaf_data['dir_body_map'] @ body_flux[box, loc_int]
        return rhs_local

    def _get_iti_body_rhs(self, ff_body_func=None, ff_body_vec=None):
        rhs_local = self._get_iti_body_local(ff_body_func=ff_body_func, ff_body_vec=ff_body_vec)
        if rhs_local is None:
            return None
        return self.iti_swap @ rhs_local

    def _get_rhs_iti(self, uu_dir_func, uu_dir_vec=None, ff_body_func=None, ff_body_vec=None):
        if uu_dir_vec is None:
            boundary_data = uu_dir_func(self.XX_active[self.I_Xtot, :])
        else:
            boundary_data = uu_dir_vec

        if isinstance(boundary_data, torch.Tensor):
            boundary_data_np = boundary_data.detach().cpu().numpy()
        else:
            boundary_data_np = np.asarray(boundary_data)

        if boundary_data_np.ndim == 1:
            boundary_data_np = boundary_data_np[:, None]
        boundary_data_np = boundary_data_np.astype(np.complex128, copy=False)

        ff_body = -self.A_CX @ boundary_data_np
        body_rhs = self._get_iti_body_rhs(ff_body_func=ff_body_func, ff_body_vec=ff_body_vec)
        if body_rhs is not None:
            ff_body += body_rhs

        return ff_body, torch.from_numpy(boundary_data_np)

    def _expand_iti_solution(self, boundary_np, sol_np, body_flux=None):
        return sol_np

    def _reconstruct_iti_boundary(self, boundary_data, sol, ff_body_func=None, ff_body_vec=None):
        boundary_np = boundary_data.detach().cpu().numpy().astype(np.complex128, copy=False)
        sol_np = sol.detach().cpu().numpy().astype(np.complex128, copy=False)
        body_flux = self._compute_iti_body_flux(ff_body_func=ff_body_func, ff_body_vec=ff_body_vec)
        sol_dir = self._expand_iti_solution(boundary_np, sol_np, body_flux=body_flux)

        total_surface = int(self.hps.nboxes) * self.iti_size_ext
        surface_dup = np.zeros((total_surface, boundary_np.shape[-1]), dtype=np.complex128)

        for box, leaf_data in enumerate(self.iti_leaf_data):
            rows = leaf_data['global_int']
            cols = leaf_data['global_ext']
            dup_inds = leaf_data['dup_inds']
            loc_int = leaf_data['local_int']
            loc_ext = leaf_data['local_ext']

            x_int = sol_dir[rows]
            boundary_leaf = boundary_np[cols]
            body_leaf = None if body_flux is None else body_flux[box]

            rhs = x_int + leaf_data['dir_T_cx'] @ boundary_leaf
            if body_leaf is not None:
                rhs += body_leaf[loc_int]
            # u_int is the recovered Dirichlet trace on the leaf interfaces.
            # The ItI system solves for incoming impedance data x_int; this
            # converts it back to boundary values for the existing leaf solve.
            u_int = leaf_data['dir_M_inv'] @ rhs
            u_full = np.zeros((self.iti_size_ext, boundary_np.shape[-1]), dtype=np.complex128)
            u_full[loc_int] = u_int
            u_full[loc_ext] = boundary_leaf

            surface_dup[dup_inds] = u_full

        surface_unique = surface_dup[self.hps.I_unique.detach().cpu().numpy()]
        return torch.from_numpy(surface_unique)

    def _solve_helper_iti(self, uu_dir_func, uu_dir_vec=None, ff_body_func=None, ff_body_vec=None):
        tic = time()
        ff_body, boundary_data = self._get_rhs_iti(
            uu_dir_func,
            uu_dir_vec=uu_dir_vec,
            ff_body_func=ff_body_func,
            ff_body_vec=ff_body_vec,
        )

        sol = self._solve_factorized_system(ff_body)
        res = self.A_CC @ sol - ff_body
        if ff_body.size == 0:
            relerr = 0.0
        else:
            ff_norm = np.linalg.norm(ff_body, ord=2)
            relerr = np.linalg.norm(res, ord=2) / ff_norm if ff_norm > 0 else np.linalg.norm(res, ord=2)
        print("NORM OF RESIDUAL for solver %5.2e" % relerr)

        toc_solve = time() - tic
        return torch.from_numpy(np.asarray(sol)), toc_solve, torch.from_numpy(np.asarray(ff_body)), boundary_data

    # Builds the sparse matrix that encodes the solutions to boundary points.
    def build_blackboxsolver(self,solvertype,verbose):
        if self.use_iti_maps:
            return dict()
        if not self.statically_condense:
            return self.build_blackboxsolver_uncondensed(solvertype, verbose)

        info_dict = dict()
        #print("Made it to build_blackboxsolver")
        I_copy1 = self.hps.I_copy1.detach().cpu().numpy()
        I_copy2 = self.hps.I_copy2.detach().cpu().numpy()
        I_ext   = self.I_Xtot_in_unique.detach().cpu().numpy()
        A_CC = self.A[I_copy1] + self.A[I_copy2]
        A_CC = A_CC[:,I_copy1] + A_CC[:,I_copy2]
        A_CX = self.A[I_copy1] + self.A[I_copy2]
        A_CX = A_CX[:,I_ext]
        A_XC = self.A[I_ext]
        A_XC = A_XC[:,I_copy1] + A_XC[:,I_copy2]
        A_XX = self.A[I_ext][:,I_ext]

        self.A_CC = A_CC
        self.A_CX = A_CX
        self.A_XC = A_XC
        self.A_XX = A_XX
        """
        print("Trimmed the unnecessary parts to make A_CC, now assembly with PETSc (or maybe SuperLU)")
        if (not petsc_available):
            info_dict = self.build_superLU(verbose)
        else:
            info_dict = self.build_petsc(solvertype,verbose)
        """
        return info_dict

    def build_blackboxsolver_uncondensed(self, solvertype, verbose):
        info_dict = dict()
        I_int, I_copy1, I_copy2, I_ext, nint_leaf, size_ext = self._get_uncondensed_indices()

        tmp_int = self.A[I_copy1] + self.A[I_copy2]
        A_UU = self.A[I_int][:, I_int]
        A_US = self.A[I_int][:, I_copy1] + self.A[I_int][:, I_copy2]
        A_UX = self.A[I_int][:, I_ext]
        A_SU = tmp_int[:, I_int]
        A_SS = tmp_int[:, I_copy1] + tmp_int[:, I_copy2]
        A_SX = tmp_int[:, I_ext]
        A_XU = self.A[I_ext][:, I_int]
        A_XS = self.A[I_ext][:, I_copy1] + self.A[I_ext][:, I_copy2]
        A_XX = self.A[I_ext][:, I_ext]

        self.A_CC = sp_vstack((sp_hstack((A_UU, A_US)), sp_hstack((A_SU, A_SS))), format='csr')
        self.A_CX = sp_vstack((A_UX, A_SX), format='csr')
        self.A_XC = sp_hstack((A_XU, A_XS), format='csr')
        self.A_XX = A_XX.tocsr()

        self.uncondensed_nint_leaf = nint_leaf
        self.uncondensed_size_ext = size_ext
        self.uncondensed_nint_total = len(I_int)
        return info_dict

    def build_factorize(self,solvertype,verbose):
        info_dict = dict()
        if verbose:
            print("Trimmed the unnecessary parts to make A_CC, now assembly with PETSc (or maybe SuperLU)")

        use_approx = solvertype not in ('superLU', 'mumps')
        self.setup_solver_Aii(use_approx=use_approx)
        backend = self.sparse_solver.backend

        info_dict['toc_build_blackbox'] = self.sparse_solver.build_time
        info_dict['solver_type'] = backend
        if backend in ('superlu', 'mumps'):
            info_dict['toc_build_superLU'] = self.sparse_solver.build_time
        if self.sparse_solver.storage_bytes is not None:
            info_dict['mem_build_superLU'] = self.sparse_solver.storage_bytes / 1e9

        return info_dict


    def build(self,sparse_assembly, solver_type,verbose=True):
        """
        Assembles the sparse system, then factorizes it using build_blackboxsolver
        """
        if self.use_iti_maps:
            self._require_iti_supported()

        self.sparse_assembly = sparse_assembly
        self.solver_type     = solver_type
        ########## sparse assembly ##########
        if (sparse_assembly == 'reduced_cpu'):
            device = torch.device('cpu')
        elif (sparse_assembly == 'reduced_gpu'):
            device = torch.device('cuda')

        #print("About to build sparse matrix")
        tic = time()
        if self.use_iti_maps:
            assembly_time_dict = self._build_iti_system(device,verbose)
            self.A = self.A_CC
        elif self.statically_condense:
            self.A,assembly_time_dict = self.hps.sparse_mat(device,verbose)
        else:
            self._require_uncondensed_supported()
            self.A,assembly_time_dict = self.hps.sparse_mat_uncondensed(device,verbose)
        toc_assembly_tot = time() - tic

        #print("Built sparse matrix A")
        csr_stor  = self.A.data.nbytes
        csr_stor += self.A.indices.nbytes + self.A.indptr.nbytes
        csr_stor /= 1e9
        if (verbose):
            print("SPARSE ASSEMBLY")
            print("\t--time for (sparse assembly) (%5.2f) s"\
                  % (toc_assembly_tot))
            print("\t--memory for (A sparse) (%5.2f) GB"\
                  % (csr_stor))
        
        
        ########## sparse slab operations ##########
        info_dict = dict()
        if (solver_type == 'slabLU'):
            raise ValueError("not included in this version")
        else:
            info_dict = self.build_blackboxsolver(solver_type,verbose)
            if ('toc_build_blackbox' in info_dict):
                info_dict['toc_build_blackbox'] += toc_assembly_tot
               
        info_dict['toc_assembly'] = assembly_time_dict['toc_DtN']

        return info_dict

    def _solve_factorized_system(self, ff_body):
        ff_body = np.asarray(ff_body)
        solver_Aii = self.solver_Aii
        if ff_body.ndim == 1:
            return np.asarray(solver_Aii.matvec(ff_body))
        return np.asarray(solver_Aii.matmat(ff_body))
                
    def get_rhs(self,uu_dir_func,uu_dir_vec=None,ff_body_func=None,ff_body_vec=None,sum_body_load=True):
        """
        Obtains the right-hand-side of a solve based on body loads and Dirichlet BCs.
        """
        I_Ctot   = self.I_Ctot
        I_Xtot   = self.I_Xtot
        nrhs = 1
            
        # Dirichlet data
        # Note for 3D self.XX_active is already converted to Gaussian nodes and features unique entries, so
        # this is right
        if uu_dir_vec is None:
            uu_dir = uu_dir_func(self.XX_active[I_Xtot,:])
        else:
            uu_dir = uu_dir_vec

        # body load on I_Ctot
        I_copy1  = self.hps.I_copy1
        I_copy2  = self.hps.I_copy2

        ff_body  = -apply_sparse_lowmem(self.A,I_copy1,self.I_Xtot_in_unique,uu_dir)
        ff_body  = ff_body - apply_sparse_lowmem(self.A,I_copy2,self.I_Xtot_in_unique,uu_dir)

        if (ff_body_func is not None) or (ff_body_vec is not None):    # THIS NEEDS TO CHANGE FOR C-N

            if (self.sparse_assembly == 'reduced_gpu'):
                device = torch.device('cuda')
            elif (self.sparse_assembly == 'reduced_cpu'):
                device = torch.device('cpu')
            
            if self.d==2:
                ff_body += self.hps.reduce_body(device,ff_body_func,ff_body_vec)[I_Ctot]
            elif self.d==3:
                ff_body += self.hps.reduce_body(device,ff_body_func,ff_body_vec)[I_Ctot]
        
        return ff_body
    
    def solve_residual_calc(self,sol,ff_body):
        """
        This takes our computed solution sol and plugs it into A to get the difference, A (sol) - f
        """
        I_copy1 = self.hps.I_copy1
        I_copy2 = self.hps.I_copy2
        res = apply_sparse_lowmem(self.A,I_copy1,I_copy1,sol)
        res = res + apply_sparse_lowmem(self.A,I_copy1,I_copy2,sol)
        res = res + apply_sparse_lowmem(self.A,I_copy2,I_copy1,sol)
        res = res + apply_sparse_lowmem(self.A,I_copy2,I_copy2,sol)
        res = res - ff_body

        return res
    
    def solve_helper_blackbox(self,uu_dir_func,uu_dir_vec=None,ff_body_func=None,ff_body_vec=None):
        """
        This solves for the box boundaries using either superLU or PETSC.
        """
        
        tic = time()
        ff_body = self.get_rhs(uu_dir_func,uu_dir_vec=uu_dir_vec,ff_body_func=ff_body_func,ff_body_vec=ff_body_vec)
        ff_body = np.array(ff_body)

        sol = self._solve_factorized_system(ff_body)

        res     = self.A_CC @ sol - ff_body
        if ff_body.size == 0:
            relerr = 0.0
        else:
            ff_norm = np.linalg.norm(ff_body,ord=2)
            relerr  = np.linalg.norm(res,ord=2)/ff_norm if ff_norm > 0 else np.linalg.norm(res,ord=2)
        print("NORM OF RESIDUAL for solver %5.2e" % relerr)

        sol       = torch.tensor(sol); ff_body = torch.tensor(ff_body)
        toc_solve = time() - tic

        return sol,toc_solve, ff_body

    def solve_helper_uncondensed(self, uu_dir_func, uu_dir_vec=None, ff_body_func=None, ff_body_vec=None):
        if (ff_body_func is not None) or (ff_body_vec is not None):
            raise NotImplementedError("statically_condense=False currently supports no body load.")

        tic = time()
        if uu_dir_vec is None:
            uu_dir = uu_dir_func(self.XX_active[self.I_Xtot,:])
        else:
            uu_dir = uu_dir_vec

        uu_dir_np = np.array(uu_dir)
        ff_body = -self.A_CX @ uu_dir_np
        sol = self._solve_factorized_system(ff_body)

        res = self.A_CC @ sol - ff_body
        if ff_body.size == 0:
            relerr = 0.0
        else:
            ff_norm = np.linalg.norm(ff_body, ord=2)
            relerr = np.linalg.norm(res, ord=2) / ff_norm if ff_norm > 0 else np.linalg.norm(res, ord=2)
        print("NORM OF RESIDUAL for solver %5.2e" % relerr)

        toc_solve = time() - tic
        return torch.tensor(sol), toc_solve, torch.tensor(ff_body), uu_dir

    def reconstruct_uncondensed_solution(self, uu_dir_func, sol, uu_dir_vec=None):
        if self.sparse_assembly == 'reduced_gpu':
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        nboxes = int(self.hps.nboxes)
        nrhs = sol.shape[-1]
        Jc = torch.tensor(self.hps.H.JJ.Jc).to(device)

        surface_sol = torch.zeros((len(self.hps.I_unique), nrhs), device=device, dtype=sol.dtype)
        surface_sol[self.I_Ctot] = sol[self.uncondensed_nint_total:].to(device)
        if uu_dir_vec is None:
            surface_sol[self.I_Xtot] = uu_dir_func(self.hps.xx_active[self.I_Xtot]).to(device)
        else:
            surface_sol[self.I_Xtot] = uu_dir_vec.to(device)

        uu_sol_bnd = self.hps.expand_boundary_data(device, surface_sol)
        interior_sol = sol[:self.uncondensed_nint_total].to(device).reshape(nboxes, self.uncondensed_nint_leaf, nrhs)

        uu_sol_tot = torch.zeros(nboxes, np.prod(self.hps.p), nrhs, device=device)
        uu_sol_tot[:, Jc, :] = interior_sol
        uu_sol_tot = self.hps.fill_missing_boundary_values(device, uu_sol_tot, uu_sol_bnd)
        return uu_sol_tot.flatten(start_dim=0, end_dim=-2).cpu()
        

    def solve(self,uu_dir_func,uu_dir_vec=None,ff_body_func=None,ff_body_vec=None,known_sol=False):
        """
        The main function that solves the sparse system and leaf interiors.
        """
        if (self.solver_type == 'slabLU'):
            raise ValueError("not included in this version")
        elif self.use_iti_maps:
            sol,toc_system_solve, ff_body, boundary_data = self._solve_helper_iti(
                uu_dir_func,
                uu_dir_vec=uu_dir_vec,
                ff_body_func=ff_body_func,
                ff_body_vec=ff_body_vec,
            )
        elif not self.statically_condense:
            sol,toc_system_solve, ff_body, uu_dir = self.solve_helper_uncondensed(
                uu_dir_func, uu_dir_vec=uu_dir_vec, ff_body_func=ff_body_func, ff_body_vec=ff_body_vec
            )
        else:
            sol,toc_system_solve, ff_body = self.solve_helper_blackbox(uu_dir_func,uu_dir_vec=uu_dir_vec,ff_body_func=ff_body_func,ff_body_vec=ff_body_vec)

        if self.use_iti_maps:
            rel_err = float('nan')
            forward_bdry_error = float('nan')
            reverse_bdry_error = float('nan')
            device = self._get_surface_device()

            surface_sol = self._reconstruct_iti_boundary(
                boundary_data,
                sol,
                ff_body_func=ff_body_func,
                ff_body_vec=ff_body_vec,
            ).to(device)

            tic = time()
            sol_tot,resloc_hps = self.hps.solve(device,surface_sol,ff_body_func=ff_body_func,ff_body_vec=ff_body_vec,uu_true=None)
            toc_leaf_solve = time() - tic
            sol_tot = sol_tot.cpu()
        elif not self.statically_condense:
            interior_coords = self.hps.grid_xx[:, self.hps.H.JJ.Jc, :].reshape(-1, self.d)
            if uu_dir_vec is None:
                true_int_sol = uu_dir_func(interior_coords)
                true_c_sol = torch.vstack((true_int_sol, uu_dir_func(self.hps.xx_active[self.I_Ctot])))
            else:
                print("We don't have a function for subdomain boundaries, so we're just assessing stability")
                true_c_sol = sol

            res = np.linalg.norm(self.A_CC @ true_c_sol.cpu().detach().numpy() - ff_body.cpu().detach().numpy()) / torch.linalg.norm(ff_body)
            forward_bdry_error = res
            reverse_bdry_error = torch.linalg.norm(sol - true_c_sol) / torch.linalg.norm(true_c_sol)
            reverse_bdry_error = reverse_bdry_error.item()

            rel_err = forward_bdry_error
            if known_sol:
                print("Relative error when applying the sparse system as a FORWARD operator on the true solution, i.e. ||A u_true - b||: %5.2e" % forward_bdry_error)
                print("Relative error when using the factorized sparse system to solve, i.e. ||A^-1 b - u_true||: %5.2e" % reverse_bdry_error)

            sol_tot = self.reconstruct_uncondensed_solution(uu_dir_func, sol, uu_dir_vec=uu_dir_vec)
            resloc_hps = torch.tensor([float('nan')])
            toc_leaf_solve = 0.0
        else:
            # We set the solution on the subdomain boundaries to the result of our sparse system.
            sol_tot = torch.zeros((len(self.hps.I_unique), sol.shape[-1]), dtype=sol.dtype)
            sol_tot[self.I_Ctot] = sol

            # Here we set the true exterior to the given data:
            if uu_dir_vec is None:
                true_c_sol = uu_dir_func(self.hps.xx_active[self.I_Ctot])
                sol_tot[self.I_Xtot] = uu_dir_func(self.hps.xx_active[self.I_Xtot])
            else:
                print("We don't have a function for subdomain boundaries, so we're just assessing stability")
                true_c_sol = sol
                sol_tot[self.I_Xtot] = uu_dir_vec

            res = np.linalg.norm(self.A_CC @ true_c_sol.cpu().detach().numpy() - ff_body.cpu().detach().numpy()) / torch.linalg.norm(ff_body)
            forward_bdry_error = res
            reverse_bdry_error = torch.linalg.norm(sol - true_c_sol) / torch.linalg.norm(true_c_sol)
            reverse_bdry_error = reverse_bdry_error.item()

            rel_err = forward_bdry_error

            if known_sol:
                print("Relative error when applying the sparse system as a FORWARD operator on the true solution, i.e. ||A u_true - b||: %5.2e" % forward_bdry_error)
                print("Relative error when using the factorized sparse system to solve, i.e. ||A^-1 b - u_true||: %5.2e" % reverse_bdry_error)

            resloc_hps = torch.tensor([float('nan')])
            if (self.sparse_assembly == 'reduced_gpu'):
                device=torch.device('cuda')
            else:
                device = torch.device('cpu')

            # Creating the true solution for comparison's sake.
            uu_true = None
            if known_sol and uu_dir_vec is not None:
                GridX   = self.hps.grid_xx.clone()
                uu_true = torch.zeros((GridX.shape[0], GridX.shape[1],1), device=device)
                for i in range(GridX.shape[0]):
                    uu_true[i] = uu_dir_func(GridX[i])
            
            tic = time()
            sol_tot,resloc_hps = self.hps.solve(device,sol_tot,ff_body_func=ff_body_func,ff_body_vec=ff_body_vec,uu_true=uu_true)
            toc_leaf_solve = time() - tic
            sol_tot = sol_tot.cpu()

        true_err = torch.tensor([float('nan')])
        if (known_sol):
            sol_boxes = torch.reshape(sol_tot, (self.hps.nboxes,np.prod(self.hps.p)))
            XX       = self.hps.xx_tot
            uu_true  = uu_dir_func(XX.clone())
            uu_true  = torch.reshape(uu_true, (self.hps.nboxes,np.prod(self.hps.p)))
            Jx       = torch.tensor(self.hps.H.JJ.Jx)#.to(device)
            Jc       = torch.tensor(self.hps.H.JJ.Jc)#.to(device)
            Jtot     = torch.hstack((Jc,Jx))
            true_err = torch.linalg.norm(sol_boxes[:,Jtot]-uu_true[:,Jtot]) / torch.linalg.norm(uu_true[:,Jtot])

            del uu_true
            true_err = true_err.item()

        return sol_tot,rel_err,true_err,resloc_hps,toc_system_solve,toc_leaf_solve,forward_bdry_error, reverse_bdry_error
