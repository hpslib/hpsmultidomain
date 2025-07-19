import torch  # Used for tensor operations
import numpy as np  # For numerical operations, especially those not directly supported by PyTorch
import sys  # System-specific parameters and functions

import sys
sys.path.append("/Users/jkump/Desktop/hps-multidomain-disc/hps")

# Importing parent class:
from abstract_hps_solver import AbstractHPSSolver

# Importing necessary components for sparse matrix operations
from scipy.sparse import kron, diags, block_diag, eye as speye, hstack as sp_hstack
import scipy.sparse.linalg as spla  # For sparse linear algebra operations
from time import time  # For timing operations
torch.set_default_dtype(torch.double)  # Setting default tensor type to double for precision

# Import custom modules for handling multidomain discretization and partial differential operators
import hps_multidomain_disc
import pdo
from functools import reduce  # For performing cumulative operations
import scipy.sparse.linalg as sla  # For sparse linear algebra operations, alternative variable
from built_in_funcs import uu_dir_func_greens

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
    PETSc.Options()['mat_mumps_icntl_6']  = 7
    PETSc.Options()['mat_mumps_icntl_8']  = 77 # Scaling strategy, set to be automatically picked
    PETSc.Options()['mat_mumps_icntl_10'] = 0 # No iterative refinement
    PETSc.Options()['mat_mumps_icntl_12'] = 1 # Ordering strategy with icntl 6
    PETSc.Options()['mat_mumps_icntl_13'] = 0 # Parallel factorization of root node
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

    vec_full = torch.zeros(A.shape[1], v.shape[-1])
    vec_full[J] = v
    vec_full = A.T @ vec_full if transpose else A @ vec_full
    return torch.tensor(vec_full[I])

# Domain_Driver class for setting up and solving the discretized PDE
class Domain_Driver(AbstractHPSSolver):
    def __init__(self, box_geom, pdo_op, kh, a, p=12, d=2, periodic_bc=False):
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
        """
        self.d = d
        self.kh = kh
        self.periodic_bc  = periodic_bc
        self.box_geometry = box_geom # The full BoxGeometry object
        self.box_geom     = self.box_geometry.bounds.T # The array itself
        assert p > 0
        self.hps_disc(self.box_geom,a,p,d,pdo_op,periodic_bc)

        print("n is:", self.hps.n)

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
        return self.XXfull_reshape
       
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
        return self.I_Ctot #self.hps.I_copy1

    @property
    def Jx(self):
        """
        Index array for unique exterior (non-duplicated) boundary points.
        """
        return self.I_Xtot

    @property
    def npoints_dim(self):
        """
        Total number of Chebyshev points per dimension (npan_dim * p for each dimension).
        """
        return self.hps.n * self.p

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

        nrhs = uu_dir.shape[-1]

        sol_tot, _, _, _, _, _, _, _ = self.solve(uu_dir,uu_dir_vec=uu_dir,ff_body_vec=ff_body)

        print(sol_tot.shape)

        return sol_tot

    def verify_discretization(self, kh):
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

            
    ############################### HPS discretization and panel split #####################
    def hps_disc(self,box_geom,a,p,d,pdo_op,periodic_bc):

        if isinstance(a, (int, float)):
            a = np.array([a] * d)

        HPS_multi = hps_multidomain_disc.HPS_Multidomain(pdo_op,box_geom,a,p,d, periodic_bc=periodic_bc)

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

        self.XXfull_reshape = torch.reshape(self.hps.grid_xx, (self.hps.grid_xx.shape[0] * self.hps.grid_xx.shape[1], -1))
            
    
    def build_superLU(self,verbose):
        """
        Constructs the sparse system using superLU from scipy.sparse. Used if PETSc is not avaialble.
        """
        info_dict = dict()
        try:
            tic = time()
            LU = sla.splu(self.A_CC)          
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
            self.superLU = LU

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
        if self.d==3:
            PETSc.Options()['mat_mumps_icntl_15'] = -self.hps.q**2

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
        
        self.petsc_LU = ksp

        return info_dict
    
    # Builds the sparse matrix that encodes the solutions to boundary points.
    def build_blackboxsolver(self,solvertype,verbose):

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
        print("Trimmed the unnecessary parts to make A_CC, now assembly with PETSc (or maybe SuperLU)")
        if (not petsc_available):
            info_dict = self.build_superLU(verbose)
        else:
            info_dict = self.build_petsc(solvertype,verbose)
        return info_dict

    def build(self,sparse_assembly, solver_type,verbose=True):
        """
        Assembles the sparse system, then factorizes it using build_blackboxsolver
        """
        
        self.sparse_assembly = sparse_assembly
        self.solver_type     = solver_type
        ########## sparse assembly ##########
        if (sparse_assembly == 'reduced_cpu'):
            device = torch.device('cpu')
        elif (sparse_assembly == 'reduced_gpu'):
            device = torch.device('cuda')

        #print("About to build sparse matrix")
        tic = time()
        self.A,assembly_time_dict = self.hps.sparse_mat(device,verbose)
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

        if (not petsc_available):
            sol = self.superLU.solve(ff_body)
        else:
            psol = PETSc.Vec().createWithArray(np.ones(ff_body.shape))
            pb   = PETSc.Vec().createWithArray(ff_body.copy())
            self.petsc_LU.solve(pb,psol)
            sol  = psol.getArray().reshape(ff_body.shape)

        res     = self.A_CC @ sol - ff_body
        relerr  = np.linalg.norm(res,ord=2)/np.linalg.norm(ff_body,ord=2)
        print("NORM OF RESIDUAL for solver %5.2e" % relerr)

        sol       = torch.tensor(sol); ff_body = torch.tensor(ff_body)
        toc_solve = time() - tic

        return sol,toc_solve, ff_body
        

    def solve(self,uu_dir_func,uu_dir_vec=None,ff_body_func=None,ff_body_vec=None,known_sol=False):
        """
        The main function that solves the sparse system and leaf interiors.
        """
        if (self.solver_type == 'slabLU'):
            raise ValueError("not included in this version")
        else:
            sol,toc_system_solve, ff_body = self.solve_helper_blackbox(uu_dir_func,uu_dir_vec=uu_dir_vec,ff_body_func=ff_body_func,ff_body_vec=ff_body_vec)

        # We set the solution on the subdomain boundaries to the result of our sparse system.
        sol_tot = torch.zeros(len(self.hps.I_unique),1)
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
            sol_boxes = torch.reshape(sol_tot, (self.hps.nboxes,self.hps.p**self.d))
            XX       = self.hps.xx_tot
            uu_true  = uu_dir_func(XX.clone())
            uu_true  = torch.reshape(uu_true, (self.hps.nboxes,self.hps.p**self.d))
            Jx       = torch.tensor(self.hps.H.JJ.Jx)#.to(device)
            Jc       = torch.tensor(self.hps.H.JJ.Jc)#.to(device)
            Jtot     = torch.hstack((Jc,Jx))
            true_err = torch.linalg.norm(sol_boxes[:,Jtot]-uu_true[:,Jtot]) / torch.linalg.norm(uu_true[:,Jtot])

            del uu_true
            true_err = true_err.item()

        return sol_tot,rel_err,true_err,resloc_hps,toc_system_solve,toc_leaf_solve,forward_bdry_error, reverse_bdry_error
