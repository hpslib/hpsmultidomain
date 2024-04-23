import torch  # Used for tensor operations
import numpy as np  # For numerical operations, especially those not directly supported by PyTorch
import sys  # System-specific parameters and functions

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

from fd_disc import *  # Importing all from the finite difference discretization module

# Attempting to import the PETSc library for parallel computation, handling failure gracefully
try:
    from petsc4py import PETSc
    petsc_available = True
except ImportError:
    petsc_available = False
    print("petsc not available")

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

def get_nearest_div(n, x):
    """
    Finds the divisor of n that is nearest to x. In case of a tie, chooses the larger divisor.
    
    Parameters:
    - n: The number to find divisors of.
    - x: The target value to approximate through divisors of n.
    
    Returns:
    - The divisor of n nearest to x.
    """
    factors = list(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
    factors = torch.tensor(factors)
    nearest_div, dist = n, np.abs(n - x)
    for f in factors:
        dist_f = np.abs(f - x)
        if dist_f < dist or (dist_f == dist and f > nearest_div):
            nearest_div, dist = f, dist_f
    return nearest_div.item()

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
    vec_full = torch.zeros(A.shape[0], v.shape[-1])
    vec_full[J] = v
    vec_full = A.T @ vec_full if transpose else A @ vec_full
    return torch.tensor(vec_full[I])

# Domain_Driver class for setting up and solving the discretized PDE
class Domain_Driver:
    def __init__(self, box_geom, pdo_op, kh, a, p=12, buf_constant=0.5, periodic_bc=False):
        """
        Initializes the domain and discretization for solving a PDE.
        
        Parameters:
        - box_geom: Geometry of the computational domain.
        - pdo_op: The partial differential operator to be solved.
        - kh: Wave number or parameter in the differential equation.
        - a: Characteristic length for HPS.
        - p: Polynomial degree for HPS discretization (ignored for FD).
        - buf_constant: Buffer size constant for dividing the domain in HPS.
        - periodic_bc: Boolean indicating if periodic boundary conditions are applied.
        """
        self.kh = kh;
        self.periodic_bc = periodic_bc
        if (periodic_bc):
            assert p > 0
        
        ## buffer size is chosen as buf_constant * n^{2/3}
        self.buf_constant = buf_constant
        
        # interpret h as parameter a
        self.hps_disc(box_geom,a,p,pdo_op)
        self.hps_panel_split()
        self.ntot = self.hps.xx_active.shape[0]

        # local inds for each slab
        I_L = self.I_L; I_R = self.I_R; I_U = self.I_U; I_D = self.I_D
        I_C = self.I_C; Npan = self.Npan

        # all internal nodes for slab
        I_slabC = self.inds_pans[:,I_C].flatten(); slab_Cshape = I_C.shape[0]
        
        # interfaces between slabs
        if (periodic_bc):
            I_slabX = torch.cat((self.inds_pans[:,I_L].flatten(),self.inds_pans[Npan-1,I_R])); 
        else:
            I_slabX = self.inds_pans[1:,I_L].flatten()
        slab_Xshape = I_L.shape[0]
        self.I_slabX = I_slabX
        I_Ctot   = torch.hstack((I_slabC,I_slabX))
        
        if (periodic_bc):
            slab_bnd_size = self.I_L.shape[0]; slab_int_size = self.I_C.shape[0]
            slab_interior_offset = slab_int_size * self.Npan
            self.I_Ctot_unique = torch.arange(slab_interior_offset + slab_bnd_size * (self.Npan))
            self.I_Ctot_copy1  = torch.arange(slab_bnd_size) + slab_interior_offset
            self.I_Ctot_copy2  = torch.arange(slab_bnd_size) + slab_interior_offset + slab_bnd_size * self.Npan

        # dirichlet data for entire domain
        I_Ldir = self.inds_pans[0,I_L]
        I_Rdir = self.inds_pans[Npan-1,I_R]
        I_Ddir = self.inds_pans[:,I_D].flatten();
        I_Udir = self.inds_pans[:,I_U].flatten();
        
        self.I_slabX = I_slabX; self.I_slabC = I_slabC
        self.I_Ctot  = I_Ctot;
        if (periodic_bc):
            self.I_Xtot  = torch.hstack((I_Ddir,I_Udir))
        else:
            self.I_Xtot  = torch.hstack((I_Ldir,I_Rdir,I_Ddir,I_Udir))
            
            
    ############################### HPS discretiation and panel split #####################
    def hps_disc(self,box_geom,a,p,pdo_op):

        HPS_multi = hps_multidomain_disc.HPS_Multidomain(pdo_op,box_geom,a,p)

        # find buf
        size_face = HPS_multi.p-2; n0,n1 = HPS_multi.n
        n0 = n0.item(); n1 = n1.item()
        npan_max = torch.max(HPS_multi.n).item()
        n_tmp = (npan_max) * size_face - 1; n_tmp = n_tmp
        
        # set constant to 0.5,1.0 works on ladyzhen
        buf_points = int(n_tmp**(2/3)*self.buf_constant); buf_points = np.min([400,buf_points])
            
        buf = np.max([int(buf_points/size_face)+1,2]); 
        buf = get_nearest_div(n0,buf);

        Npan = int(n0/buf); 
        print("HPS discretization a=%5.2e,p=%d"%(a,p))
        print("\t--params(n0,n1,buf) (%d,%d,%d)"%(n0,n1,buf))

        nfaces_pan  = (2*n1+1)*buf + n1
        inds_pans   = torch.zeros(Npan,nfaces_pan*size_face).long()

        for j in range(Npan):
            npan_offset  = (2*n1+1)*buf * j
            inds_pans[j] = torch.arange(nfaces_pan*size_face) + npan_offset*size_face

        self.Npan      = Npan;
        self.Npan_loc  = n1;
        self.buf_pans  = buf
        self.inds_pans = inds_pans
        
        self.elim_nblocks = buf-1;      
        self.elim_bs = size_face
        self.rem_nblocks  = self.Npan_loc-1; 
        self.rem_bs  = buf*size_face
        self.hps = HPS_multi
        
    def hps_panel_split(self):
        
        size_face = self.hps.p-2; n0,n1 = self.hps.n; buf = self.buf_pans
        Npan_loc  = self.Npan_loc
        
        elim_nblocks = self.elim_nblocks;      elim_bs = self.elim_bs
        rem_nblocks  = self.rem_nblocks;       rem_bs  = self.rem_bs
        
        self.I_L = torch.arange(n1*size_face)
        self.I_R = torch.arange(n1*size_face) + (2*n1+1)*buf*size_face

        I_elim = torch.zeros(Npan_loc,elim_nblocks,elim_bs).long()
        I_rem  = torch.zeros(rem_nblocks,buf,size_face).long()

        I_D    = torch.zeros(buf,size_face).long()
        I_U    = torch.zeros(buf,size_face).long()

        for b in range(buf):

            buf_offset = (2*n1+1)*b

            # exterior down index
            I_D[b] = torch.arange(size_face) + (buf_offset+n1)*size_face
            # exterior up index
            I_U[b] = torch.arange(size_face) + (buf_offset+2*n1)*size_face
            # rem index
            for box_j in range(1,n1):
                I_rem[box_j-1,b] = torch.arange(size_face) + (buf_offset+n1+box_j) * size_face
            if (b > 0):
                for box_j in range(n1):
                    I_elim[box_j,b-1] = torch.arange(size_face) + (buf_offset+box_j) * size_face

        I_rem  = I_rem.flatten(start_dim=1,end_dim=-1)
        I_elim = I_elim.flatten(start_dim=1,end_dim=-1)
        self.I_D = I_D.flatten()
        self.I_U = I_U.flatten()

        self.I_C = torch.hstack((I_elim.flatten(),I_rem.flatten()))
    
    def build_superLU(self,verbose):
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
        info_dict = dict()
        tmp = self.A_CC.tocsr()
        pA = PETSc.Mat().createAIJ(tmp.shape, csr=(tmp.indptr,tmp.indices,tmp.data),comm=PETSc.COMM_WORLD)
        
        ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
        ksp.setOperators(pA)
        ksp.setType('preonly')
        
        ksp.getPC().setType('lu')
        ksp.getPC().setFactorSolverType(solvertype)
        
        px = PETSc.Vec().createWithArray(np.ones(tmp.shape[0]),comm=PETSc.COMM_WORLD)
        pb = PETSc.Vec().createWithArray(np.ones(tmp.shape[0]),comm=PETSc.COMM_WORLD)
        
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
    
    def build_blackboxsolver(self,solvertype,verbose):
        info_dict = dict()
        
        if (not self.periodic_bc):
            A_CC = self.A[self.I_Ctot][:,self.I_Ctot].tocsc()
            self.A_CC = A_CC
        else:
            A_copy = self.A[np.ix_(self.I_Ctot,self.I_Ctot)].tolil()
            A_copy[np.ix_(self.I_Ctot_unique,self.I_Ctot_copy1)] += \
            A_copy[np.ix_(self.I_Ctot_unique,self.I_Ctot_copy2)]
        
            A_copy[np.ix_(self.I_Ctot_copy1,self.I_Ctot_unique)] += \
            A_copy[np.ix_(self.I_Ctot_copy2,self.I_Ctot_unique)] 
        
            A_copy[np.ix_(self.I_Ctot_copy1,self.I_Ctot_copy1)] += \
            A_copy[np.ix_(self.I_Ctot_copy2,self.I_Ctot_copy2)]
            
            A_CC = A_copy[np.ix_(self.I_Ctot_unique,self.I_Ctot_unique)].tocsc()
            self.A_CC = A_CC
        if (not petsc_available):
            info_dict = self.build_superLU(verbose)
        else:
            info_dict = self.build_petsc(solvertype,verbose)
        return info_dict
        
    def build(self,sparse_assembly,
              solver_type,verbose=True):
        
        self.sparse_assembly = sparse_assembly
        self.solver_type     = solver_type
        ########## sparse assembly ##########
        if (sparse_assembly == 'reduced_cpu'):
            device = torch.device('cpu')
            tic = time()
            self.A,assembly_time_dict    = self.hps.sparse_mat(device,verbose)
            toc_assembly_tot = time() - tic;
        elif (sparse_assembly == 'reduced_gpu'):
            device = torch.device('cuda')
            tic = time()
            self.A,assembly_time_dict    = self.hps.sparse_mat(device,verbose)
            toc_assembly_tot = time() - tic;

        csr_stor  = self.A.data.nbytes
        csr_stor += self.A.indices.nbytes + self.A.indptr.nbytes
        csr_stor /= 1e9
        if (verbose):
            print("SPARSE ASSEMBLY")
            print("\t--time for (sparse assembly) (%5.2f) s"\
                  % (toc_assembly_tot))
            print("\t--memory for (A sparse) (%5.2f) GB"\
              % (csr_stor))
        
        assert self.ntot == self.A.shape[0]
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
                
    def get_rhs(self,uu_dir_func,ff_body_func=None,sum_body_load=True):
        I_slabX = self.I_slabX; I_slabC = self.I_slabC
        I_Ctot  = self.I_Ctot;  I_Xtot  = self.I_Xtot; 
        
        slab_Cshape = self.I_C.shape[0]; slab_Xshape = self.I_L.shape[0]
        Npan = self.Npan
        nrhs = 1
        
        ## assume that XX has size npoints, 2
        XX = self.hps.xx_active
            
        # Dirichlet data
        uu_dir = uu_dir_func(XX[I_Xtot,:])

        # body load on I_Ctot
        ff_body = -apply_sparse_lowmem(self.A,I_Ctot,I_Xtot,uu_dir)
        if (ff_body_func is not None):
            
                
            if (self.sparse_assembly == 'reduced_gpu'):
                device = torch.device('cuda')
                ff_body += self.hps.reduce_body(device,ff_body_func)[I_Ctot]
            elif (self.sparse_assembly == 'reduced_cpu'):
                device = torch.device('cpu')
                ff_body += self.hps.reduce_body(device,ff_body_func)[I_Ctot]
        
        # adjust to sum body load on left and right boundaries
        if (self.periodic_bc and sum_body_load):
            ff_body[self.I_Ctot_copy1] += ff_body[self.I_Ctot_copy2]
            ff_body = ff_body[self.I_Ctot_unique]
        
        return ff_body
    
    def solve_residual_calc(self,sol,ff_body):
        if (not self.periodic_bc):
            res = apply_sparse_lowmem(self.A,self.I_Ctot,self.I_Ctot,sol) - ff_body
        else:
            res  = - ff_body
            res += apply_sparse_lowmem(self.A,self.I_Ctot[self.I_Ctot_unique],\
                                       self.I_Ctot[self.I_Ctot_unique], sol)
            res += apply_sparse_lowmem(self.A,self.I_Ctot[self.I_Ctot_unique],\
                                       self.I_Ctot[self.I_Ctot_copy2], sol[self.I_Ctot_copy1])
            res[self.I_Ctot_copy1] += apply_sparse_lowmem(self.A,self.I_Ctot[self.I_Ctot_copy2],\
                                       self.I_Ctot[self.I_Ctot_unique], sol)
            res[self.I_Ctot_copy1] += apply_sparse_lowmem(self.A,self.I_Ctot[self.I_Ctot_copy2],\
                                       self.I_Ctot[self.I_Ctot_copy2], sol[self.I_Ctot_copy1])
        return res
    
    def solve_helper_blackbox(self,uu_dir_func,ff_body_func=None):
        
        tic = time()
        ff_body = self.get_rhs(uu_dir_func,ff_body_func); ff_body = np.array(ff_body)
        try:
            if (not petsc_available):
                sol = self.superLU.solve(ff_body)
            else:
                psol = PETSc.Vec().createWithArray(np.ones(ff_body.shape))
                pb   = PETSc.Vec().createWithArray(ff_body.copy())
                self.petsc_LU.solve(pb,psol)
                sol  = psol.getArray().reshape(ff_body.shape)
        except:
            return 0,0,0
        sol = torch.tensor(sol); ff_body = torch.tensor(ff_body)
        toc_solve = time() - tic
        res = self.solve_residual_calc(sol,ff_body)
        
        rel_err = torch.linalg.norm(res) / torch.linalg.norm(ff_body)
        return sol,rel_err,toc_solve
        
        
    def solve(self,uu_dir_func,ff_body_func=None,known_sol=False):
        
        if (self.solver_type == 'slabLU'):
            raise ValueError("not included in this version")
        else:
            sol,rel_err,toc_solve = self.solve_helper_blackbox(uu_dir_func,ff_body_func)
        
        sol_tot = torch.zeros(self.A.shape[0],1)
        
        if (not self.periodic_bc):
            sol_tot[self.I_Ctot] = sol
        else:            
            sol_tot[self.I_Ctot[self.I_Ctot_unique]] = sol
            sol_tot[self.I_Ctot[self.I_Ctot_copy2]]  = sol[self.I_Ctot_copy1]
        
        sol_tot[self.I_Xtot] = uu_dir_func(self.hps.xx_active[self.I_Xtot])
        del sol
        
        resloc_hps = torch.tensor([float('nan')])
        if (self.sparse_assembly == 'reduced_gpu'):
            device=torch.device('cuda')
        else:
            device = torch.device('cpu')
        tic = time()
        sol_tot,resloc_hps = self.hps.solve(device,sol_tot,ff_body_func=ff_body_func)
        toc_solve += time() - tic
        sol_tot = sol_tot.cpu()

        true_err = torch.tensor([float('nan')])
        if (known_sol):
            XX = self.hps.xx_tot
            uu_true = uu_dir_func(XX.clone())
            true_err = torch.linalg.norm(sol_tot-uu_true) / torch.linalg.norm(uu_true)
            del uu_true
            true_err = true_err.item()

        return sol_tot,rel_err,true_err,resloc_hps,toc_solve