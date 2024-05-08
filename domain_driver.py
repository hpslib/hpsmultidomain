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

# Attempting to import the PETSc library for parallel computation, handling failure gracefully
try:
    from petsc4py import PETSc
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
        self.periodic_bc = periodic_bc
        self.box_geom    = box_geom
        assert p > 0
        self.hps_disc(box_geom,a,p,d,pdo_op,periodic_bc)
            
            
    ############################### HPS discretization and panel split #####################
    def hps_disc(self,box_geom,a,p,d,pdo_op,periodic_bc):

        HPS_multi = hps_multidomain_disc.HPS_Multidomain(pdo_op,box_geom,a,p,d)

        self.hps = HPS_multi

        if d==2:
            self.XX  = self.hps.xx_active
            
            self.ntot = self.XX.shape[0]
            
            I_Ldir = torch.where(self.XX[:,0] < self.box_geom[0,0] + 0.5 * self.hps.hmin)[0]
            I_Rdir = torch.where(self.XX[:,0] > self.box_geom[0,1] - 0.5 * self.hps.hmin)[0]
            I_Ddir = torch.where(self.XX[:,1] < self.box_geom[1,0] + 0.5 * self.hps.hmin)[0]
            I_Udir = torch.where(self.XX[:,1] > self.box_geom[1,1] - 0.5 * self.hps.hmin)[0]
            
            if (periodic_bc):
                self.I_Xtot  = torch.hstack((I_Ddir,I_Udir))
            else:
                self.I_Xtot  = torch.hstack((I_Ldir,I_Rdir,I_Ddir,I_Udir))
            
            self.I_Ctot = torch.sort(torch_setdiff1d( torch.arange(self.ntot), self.I_Xtot))[0]

            if (periodic_bc):
                
                tot_C = self.I_Ctot.shape[0]
                n_LR  = I_Rdir.shape[0]
                tot_unique = tot_C - n_LR
                
                self.I_Ctot_unique = torch.arange(tot_unique)
                self.I_Ctot_copy1  = torch.arange(n_LR)
                self.I_Ctot_copy2  = torch.arange(tot_unique, tot_C)
        else: # d==3
            self.XX  = self.hps.xx_active
            
            self.ntot = self.XX.shape[0]
            
            tol = 0.01 * self.hps.hmin # Adding a tolerance to avoid potential numerical error
            I_Ldir = torch.where(self.XX[:,0] < self.box_geom[0,0] + tol)[0]
            I_Rdir = torch.where(self.XX[:,0] > self.box_geom[0,1] - tol)[0]
            I_Ddir = torch.where(self.XX[:,1] < self.box_geom[1,0] + tol)[0]
            I_Udir = torch.where(self.XX[:,1] > self.box_geom[1,1] - tol)[0]
            I_Bdir = torch.where(self.XX[:,2] < self.box_geom[2,0] + tol)[0]
            I_Fdir = torch.where(self.XX[:,2] > self.box_geom[2,1] - tol)[0]
            
            # Assuming no periodic BC for now:
            self.I_Xtot = torch.hstack((I_Ldir,I_Rdir,I_Ddir,I_Udir,I_Bdir,I_Fdir))
            self.I_Ctot = torch.sort(torch_setdiff1d(torch.arange(self.ntot), self.I_Xtot))[0]
            
    
    # ONLY NEEDED FOR SPARSE SOLVE
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
    
    # ONLY NEEDED FOR SPARSE SOLVE
    def build_petsc(self,solvertype,verbose):
        info_dict = dict()
        tmp = self.A_CC.tocsr()
        pA = PETSc.Mat().createAIJ(tmp.shape, csr=(tmp.indptr,tmp.indices,tmp.data),comm=PETSc.COMM_WORLD)
        print("Created the PETSc matrix")
        
        ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
        ksp.setOperators(pA)
        ksp.setType('preonly')
        
        ksp.getPC().setType('lu')
        ksp.getPC().setFactorSolverType(solvertype)
        
        px = PETSc.Vec().createWithArray(np.ones(tmp.shape[0]),comm=PETSc.COMM_WORLD)
        pb = PETSc.Vec().createWithArray(np.ones(tmp.shape[0]),comm=PETSc.COMM_WORLD)

        print("Initiated the ksp operator, still need to do a solve to break it in")        
        tic = time()
        ksp.solve(pb, px)
        toc_build = time() - tic
        if (verbose):
            print("\t--time for %s build through petsc = %5.2f seconds"\
                  % (solvertype,toc_build))
        print("Broke it in!")
               
        info_dict['toc_build_blackbox']   = toc_build
        info_dict['solver_type']          = "petsc_"+solvertype
        
        self.petsc_LU = ksp
        return info_dict
    
    # ONLY NEEDED FOR SPARSE SOLVE
    def build_blackboxsolver(self,solvertype,verbose):
        info_dict = dict()
        print("Made it to build_blackboxsolver")
        if (not self.periodic_bc):
            if self.d==2:
                A_CC = self.A[self.I_Ctot][:,self.I_Ctot].tocsc()
            else:
                # Since the 3D A uses indices for the total boundary, we need to
                # hit only indices in I_unique here:
                #I_unique = self.hps.I_unique.detach().cpu().numpy()
                #A_CC = self.A[I_unique[self.I_Ctot]][:,I_unique[self.I_Ctot]].tocsc()
                I_copy1 = self.hps.I_copy1.detach().cpu().numpy()
                I_copy2 = self.hps.I_copy2.detach().cpu().numpy()
                A_CC = self.A[I_copy1][:,I_copy1].tocsc()
                A_CC_add = self.A[I_copy2][:,I_copy2].tocsc()
                A_CC = A_CC + A_CC_add
            self.A_CC = A_CC
            #dense_A_CC = self.A_CC.toarray()
            #print(dense_A_CC)
            #print(dense_A_CC.shape)
            #print(np.linalg.det(dense_A_CC))
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
        print("Trimmed the unnecessary parts to make A_CC, now assembly with PETSc")
        if (not petsc_available):
            info_dict = self.build_superLU(verbose)
        else:
            info_dict = self.build_petsc(solvertype,verbose)
        return info_dict

    # MOSTLY NEEDED FOR SPARSE SOLVE
    def build(self,sparse_assembly,
              solver_type,verbose=True):
        
        self.sparse_assembly = sparse_assembly
        self.solver_type     = solver_type
        ########## sparse assembly ##########
        if (sparse_assembly == 'reduced_cpu'):
            device = torch.device('cpu')
        elif (sparse_assembly == 'reduced_gpu'):
            device = torch.device('cuda')

        tic = time()
        self.A,assembly_time_dict = self.hps.sparse_mat(device,verbose)
        toc_assembly_tot = time() - tic

        csr_stor  = self.A.data.nbytes
        csr_stor += self.A.indices.nbytes + self.A.indptr.nbytes
        csr_stor /= 1e9
        if (verbose):
            print("SPARSE ASSEMBLY")
            print("\t--time for (sparse assembly) (%5.2f) s"\
                  % (toc_assembly_tot))
            print("\t--memory for (A sparse) (%5.2f) GB"\
                  % (csr_stor))
        
        if self.d==2:
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
                
    # ONLY NEEDED FOR SPARSE SOLVE
    def get_rhs(self,uu_dir_func,ff_body_func=None,sum_body_load=True):
        I_Ctot  = self.I_Ctot;  I_Xtot  = self.I_Xtot; 
        nrhs = 1
            
        # Dirichlet data
        uu_dir = uu_dir_func(self.XX[I_Xtot,:])

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
    
    # This takes our computed solution sol and plugs it into A to get the difference, A sol - f
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
    
    # ONLY NEEDED FOR SPARSE SOLVE
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
        if self.d==2:
            if (self.solver_type == 'slabLU'):
                raise ValueError("not included in this version")
            else:
                sol,rel_err,toc_solve = self.solve_helper_blackbox(uu_dir_func,ff_body_func)

            # self.A is black box matrix:
            sol_tot = torch.zeros(self.A.shape[0],1)
            # Here we set the solution on box boundaries not exterior to whole to sol
            if (not self.periodic_bc):
                sol_tot[self.I_Ctot] = sol
            else:            
                sol_tot[self.I_Ctot[self.I_Ctot_unique]] = sol
                sol_tot[self.I_Ctot[self.I_Ctot_copy2]]  = sol[self.I_Ctot_copy1]
            # Here we set the true exterior to the given data:
            sol_tot[self.I_Xtot] = uu_dir_func(self.hps.xx_active[self.I_Xtot])
            del sol
        else: # 3D
            if (self.solver_type == 'slabLU'):
                raise ValueError("not included in this version")
            else:
                sol,rel_err,toc_solve = self.solve_helper_blackbox(uu_dir_func,ff_body_func)
                #print(sol)

            # self.A is black box matrix:
            sol_tot = torch.zeros(len(self.hps.I_unique),1)
            
            sol_tot[self.I_Ctot] = sol #uu_dir_func(self.hps.xx_active[self.I_Ctot])
            # Here we set the true exterior to the given data:
            sol_tot[self.I_Xtot] = uu_dir_func(self.hps.xx_active[self.I_Xtot])

            #print(uu_dir_func(self.hps.xx_active[self.I_Ctot]))
            #print(sol_tot[self.I_Ctot])

            print("Relative error on unique boundary is:")
            print(torch.linalg.norm(sol - uu_dir_func(self.hps.xx_active[self.I_Ctot])) / torch.linalg.norm(uu_dir_func(self.hps.xx_active[self.I_Ctot])))
        
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
            if self.d==2:
                true_err = torch.linalg.norm(sol_tot-uu_true) / torch.linalg.norm(uu_true)
            if self.d==3:
                # special protocol for 3D case, temporarily avoiding edges of boxes:
                sol_tot = torch.reshape(sol_tot, (self.hps.nboxes,self.hps.p**3))
                uu_true = torch.reshape(uu_true, (self.hps.nboxes,self.hps.p**3))

                Jx   = torch.tensor(self.hps.H.JJ.Jx)#.to(device)
                Jc   = torch.tensor(self.hps.H.JJ.Jc)#.to(device)
                Jtot = torch.hstack((Jc,Jx))
                true_err = torch.linalg.norm(sol_tot[:,Jtot]-uu_true[:,Jtot]) / torch.linalg.norm(uu_true[:,Jtot])
                #true_err = torch.linalg.norm(sol_tot[:,Jx]-uu_true[:,Jx]) / torch.linalg.norm(uu_true[:,Jx])
            del uu_true
            true_err = true_err.item()

        return sol_tot,rel_err,true_err,resloc_hps,toc_solve