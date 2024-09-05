import argparse                        # For parsing command-line arguments
import torch                           # PyTorch library for tensor computations and GPU support
import numpy as np                     # For numerical operations
from time import time                  # For timing operations
torch.set_default_dtype(torch.double)  # Set default tensor type to double precision

from domain_driver import *  # Importing domain driver utilities for PDE solving
from built_in_funcs import *  # Importing built-in functions for specific PDEs or conditions
import pickle  # For serializing and saving results
import os  # For interacting with the operating system, e.g., environment variables

# Initialize argument parser to define and read command-line arguments
parser = argparse.ArgumentParser("Call direct solver for 2D/3D domain.")
# Add expected command-line arguments for the script
parser.add_argument('--p',type=int,required=False)  # Polynomial order for certain methods
parser.add_argument('--n', type=int, required=True)  # Number of discretization points
parser.add_argument('--d', type=int, required=False, default=2)  # Spatial dimension of problem

# Arguments defining the PDE problem
parser.add_argument('--pde', type=str, required=True)  # Type of PDE
parser.add_argument('--domain', type=str, required=True)  # Domain shape
parser.add_argument('--box_xlim', type=float, required=False, default=1.0)  # Domain x limits
parser.add_argument('--box_ylim', type=float, required=False, default=1.0)  # Domain y limits
parser.add_argument('--box_zlim', type=float, required=False, default=1.0)  # Domain z limits

# Boundary condition and problem specifics
parser.add_argument('--bc', type=str, required=True)  # Boundary condition
parser.add_argument('--ppw',type=int, required=False)  # Points per wavelength for oscillatory problems
parser.add_argument('--nwaves',type=float, required=False)  # Number of wavelengths
parser.add_argument('--kh', type=float, required=False)       # checks if we have a given non-constant wavenumber

# Solver and computational specifics
parser.add_argument('--solver',type=str,required=False)  # Solver to use
parser.add_argument('--sparse_assembly',type=str,required=False, default='reduced_gpu')  # Assembly method
parser.add_argument('--pickle',type=str,required=False)  # File path for pickling results
parser.add_argument('--store_sol',action='store_true')  # Flag to store solution
parser.add_argument('--disable_cuda',action='store_true')  # Flag to disable CUDA
parser.add_argument('--periodic_bc', action='store_true')  # Flag for periodic boundary conditions

# Components tests:
parser.add_argument('--test_components', type=bool, required=False, default=False) # Test discretization components such as interpolation

args = parser.parse_args()  # Parse arguments from command line

# Check if we test components:
test_components = args.test_components

# Extract and setup basic parameters from parsed arguments
n = args.n; d = args.d
box_geom = torch.tensor([[0,args.box_xlim],[0,args.box_ylim]])  # Domain geometry tensor
if d==3:
    box_geom = torch.tensor([[0,args.box_xlim],[0,args.box_ylim],[0,args.box_zlim]])

# Print configuration based on whether ppw is set
if (args.ppw is not None):
    print("\n RUNNING PROBLEM WITH...")  # Detailed problem configuration
else:
    print("RUNNING PROBLEM WITH...")  # Simplified problem configuration if ppw is not provided
    
# Disable CUDA if requested
if (args.disable_cuda):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
# Check CUDA availability and adjust settings accordingly
print("CUDA available %s" % torch.cuda.is_available())
if (torch.cuda.is_available()):
    print("--num cuda devices %d" % torch.cuda.device_count())
if ((not torch.cuda.is_available()) and (args.sparse_assembly == 'reduced_gpu')):
    args.sparse_assembly = 'reduced_cpu'
    print("Changed sparse assembly to reduced_cpu")

# Configure the PDE operator and domain based on specified arguments
if ((args.pde == 'poisson') and (args.domain == 'square')):
    if (args.ppw is not None):
        raise ValueError
    
    # laplace operator
    op = pdo.PDO_2d(pdo.ones,pdo.ones)
    if d==3:
        op = pdo.PDO_3d(pdo.ones,pdo.ones,pdo.ones)
    kh = 0
    curved_domain = False

elif ((args.pde == 'mixed') and (args.domain == 'square')):
    if (args.ppw is not None):
        raise ValueError

    # operator:
    if d==2:
        raise ValueError
    def c(xx):
        r = torch.sqrt(xx[:,0]*xx[:,0] + xx[:,1]*xx[:,1] + xx[:,2]*xx[:,2])
        r4 = r * r * r * r
        return (xx[:,0]*xx[:,1] + xx[:,0]*xx[:,2] + xx[:,1]*xx[:,2]) / r4

    op = pdo.PDO_3d(pdo.ones,pdo.ones,pdo.ones,
                    c12=pdo.const(c=1/3),c13=pdo.const(c=1/3),c23=pdo.const(c=1/3),
                    c=c)
    kh = 0
    curved_domain = False

    #print(op.c)
    #print(op.c1)
    #print(op.c2)
    #print(op.c3)
    #print(op.c11)
    #print(op.c22)
    #print(op.c33)
    #print(op.c12)
    #print(op.c13)
    #print(op.c23)

elif ( (args.pde).startswith('bfield')):
    ppw_set = args.ppw is not None
    nwaves_set = args.nwaves is not None
    kh_set = args.kh is not None
    
    if ((not ppw_set and not nwaves_set and not kh_set)):
        raise ValueError('oscillatory bfield chosen but ppw and nwaves NOT set')
    elif (ppw_set and nwaves_set) or (kh_set and nwaves_set) or (ppw_set and kh_set):
        raise ValueError('At least two of the three between ppw, nwaves, and kh are set. Only use 1!')
    elif (kh_set):
        kh = args.kh
    elif (ppw_set):
        nwaves = int(n/args.ppw)
        kh = (nwaves+0.03)*2*np.pi+1.8 # This wrong for 3d?
    else:
        nwaves = args.nwaves
        kh = (nwaves)*2*np.pi
    
      
    if (args.pde == 'bfield_constant'):
        bfield = bfield_constant
    elif (args.pde == 'bfield_variable'):
        bfield = bfield_variable
    elif (args.pde == 'bfield_bumpy'):
        bfield = bfield_bumpy
    elif (args.pde == 'bfield_gaussian_bumps'):
        bfield = bfield_gaussian_bumps
    elif (args.pde == 'bfield_cavity'):
        bfield = bfield_cavity_scattering
    elif (args.pde == 'bfield_crystal'):
        bfield = bfield_crystal
    elif (args.pde == 'bfield_crystal_waveguide'):
        bfield = bfield_crystal_waveguide
    elif (args.pde == 'bfield_crystal_rhombus'):
        bfield = bfield_crystal_rhombus
    else:
        raise ValueError
        
    curved_domain = False
    if (args.domain == 'square'):
        
        def c(xx):
            return bfield(xx,kh)
        # var coeff Helmholtz operator
        op = pdo.PDO_2d(pdo.ones,pdo.ones,c=c)
        if d==3:
            op = pdo.PDO_3d(pdo.ones,pdo.ones,pdo.ones,c=c)
        
    elif (args.domain == 'curved'):
        
        op, param_map, \
        inv_param_map = pdo.get_param_map_and_pdo('sinusoidal', bfield, kh, d=d)
        curved_domain=True
        
    elif (args.domain == 'annulus'):
        
        op, param_map, \
        inv_param_map = pdo.get_param_map_and_pdo('annulus', bfield, kh)
        curved_domain=True
        
    elif (args.domain == 'curvy_annulus'):
        
        op, param_map, \
        inv_param_map = pdo.get_param_map_and_pdo('curvy_annulus', bfield, kh)
        curved_domain=True
    else:
        raise ValueError
    
else:
    raise ValueError
    
##### Set the domain and discretization parameters
# Additional conditional logic for different discretization strategies
if (args.p is None):
    raise ValueError('HPS selected but p not provided')
p = args.p
npan = n / (p-2)
a = 1/(2*npan)

dom = Domain_Driver(box_geom,op,\
                    kh,a,p=p,d=d,periodic_bc = args.periodic_bc)
N = (p-2) * (p*dom.hps.n[0]*dom.hps.n[1] + dom.hps.n[0] + dom.hps.n[1])
"""
print("PDO:")
from pprint import pprint
pprint(vars(op))
import inspect
print(inspect.getsource(op.c))
print(inspect.getsource(bfield))
"""

################################# BUILD OPERATOR #########################
# Build operator based on specified parameters and solver information
"""
print(args.sparse_assembly)

build_info = dom.build(sparse_assembly=args.sparse_assembly,\
                        solver_type = args.solver, verbose=True)
build_info['N']    = N
build_info['n']    = n

build_info['pde']  = args.pde
build_info['bc']   = args.bc
build_info['domain'] = args.domain
build_info['solver'] = args.solver
build_info['sparse_assembly'] = args.sparse_assembly
build_info['box_geom'] = box_geom
build_info['kh']   = kh 
build_info['periodic_bc'] = args.periodic_bc
build_info['a'] = a
build_info['p'] = p
"""
################################# SOLVE PDE ###################################
# Solve the PDE with specified configurations and print results
print("SOLVE RESULTS")
solve_info = dict()

if (args.bc == 'free_space'):
    if (args.pde == 'bfield_constant'):
        ff_body = None; known_sol = True
        
        if (not curved_domain):
            uu_dir = lambda xx: uu_dir_func_greens(d, xx,kh)
        else:
            uu_dir = lambda xx: uu_dir_func_greens(d, param_map(xx),kh)
    elif (args.pde == 'bfield_variable'):
        ff_body = None; known_sol = True

        if (not curved_domain):
            uu_dir = lambda xx: uu_true_variable_helmholtz(d, xx,kh)
        else:
            print("Error! Curved domain for bfield_variable")
    else:
        print("Error! Free space for not bfield constant or variable")

        
elif (args.bc == 'pulse'):
    ff_body = None; known_sol = False
    
    if (not curved_domain):
        uu_dir = lambda xx: uu_dir_pulse(xx,kh)
    else:
        uu_dir = lambda xx: uu_dir_pulse(param_map(xx),kh)
    
elif (args.bc == 'ones'):
    ff_body = None; known_sol = False
    
    ones_func = lambda xx: torch.ones(xx.shape[0],1)
    if (not curved_domain):
        uu_dir = lambda xx: ones_func(xx)
    else:
        uu_dir = lambda xx: ones_func(param_map(xx))  
        
elif (args.bc == 'log_dist'):
    
    if ((args.pde == 'poisson') or (args.pde == 'mixed')):
        assert kh == 0
        assert (not curved_domain)

        uu_dir  = lambda xx: uu_dir_func_greens(d, xx,kh)
        ff_body = None
        known_sol = True
    else:
        raise ValueError
else:
    raise ValueError("invalid bc")

"""

if (args.solver == 'slabLU'):
    
    raise ValueError("this code is not included in this version")
    
elif (args.solver == 'superLU'):

    #uu_sol,res, true_res,resloc_hps,toc_solve = dom.solve(uu_dir,ff_body,known_sol=known_sol)
    uu_sol,res, true_res,resloc_hps,toc_solve = dom.solve(uu_dir,ff_body,known_sol=known_sol)

    print("\t--SuperLU solved Ax=b residual %5.2e with known solution residual %5.2e and resloc_HPS %5.2e in time %5.2f s"\
          %(res,true_res,resloc_hps,toc_solve))
    solve_info['res_solve_superLU']            = res
    solve_info['trueres_solve_superLU']        = true_res
    solve_info['resloc_hps_solve_superLU']     = resloc_hps
    solve_info['toc_solve_superLU']            = toc_solve

else:

    #uu_sol,res, true_res,resloc_hps,toc_solve = dom.solve(uu_dir,ff_body,known_sol=known_sol)
    uu_sol,res, true_res,resloc_hps,toc_solve = dom.solve(uu_dir,ff_body,known_sol=known_sol)

    print("\t--Builtin solver %s solved Ax=b residual %5.2e with known solution residual %5.2e and resloc_HPS %5.2e in time %5.2f s"\
          %(args.solver,res,true_res,resloc_hps,toc_solve))
    solve_info['res_solve_petsc']            = res
    solve_info['trueres_solve_petsc']        = true_res
    solve_info['resloc_hps_solve_petsc']     = resloc_hps
    solve_info['toc_solve_petsc']            = toc_solve
    

# Optional: Store solution and/or pickle results for later use
if (args.store_sol):
    print("\t--Storing solution")
    XX = dom.hps.xx_tot
    solve_info['xx']        = XX
    solve_info['sol']       = uu_sol
    
if (args.pickle is not None):
    file_loc = args.pickle
    print("Pickling results to file %s"% (file_loc))
    f = open(file_loc,"wb+")
    build_info.update(solve_info)
    pickle.dump(build_info,f)
    #pickle.dump(solve_info,f)
    f.close()
"""
################################# EVALUATING SYSTEM COMPONENTS ###################################
# Evaluate certain parts of the 3D problem 

if (d==3):
    interpolation_info = dict()
    # First we generate the arrays of Chebyshev and Gaussian nodes:
    cheb_ext  = torch.from_numpy(dom.hps.H.zz.T[dom.hps.H.JJ.Jxreorder])
    gauss_ext = torch.from_numpy(dom.hps.H.zzG)

    uu_cheb  = uu_dir(cheb_ext)
    uu_gauss = uu_dir(gauss_ext)
    uu_interC = torch.from_numpy(dom.hps.H.Interp_mat) @ uu_gauss
    uu_interG = torch.from_numpy(dom.hps.H.Interp_mat_reverse) @ uu_cheb

    print("Relative error of Gaussian-to-Chebyshev interpolation is:")
    print(torch.norm(uu_cheb - uu_interC) / torch.norm(uu_cheb))
    print("With condition number and shape:")
    print(np.linalg.cond(dom.hps.H.Interp_mat), dom.hps.H.Interp_mat.shape)
    print("Relative error of Chebyshev-to-Gaussian interpolation is:")
    print(torch.norm(uu_gauss - uu_interG) / torch.norm(uu_gauss))
    print("With condition number and shape:")
    print(np.linalg.cond(dom.hps.H.Interp_mat_reverse), dom.hps.H.Interp_mat_reverse.shape)

    
    # Check if edges / corners are equal as expected:
    u, c = np.unique(dom.hps.H.JJ.Jxreorder, return_counts=True)
    dup = u[c > 1]
    maxVar = 0
    for elem in dup:
        variance = torch.std(uu_interC[dom.hps.H.JJ.Jxreorder == elem])
        maxVar = np.max([maxVar, variance.item()])
    print("Largest variance between redundant values is " + str(maxVar))

    interpolation_info["GtC_error"]     = torch.norm(uu_cheb - uu_interC) / torch.norm(uu_cheb)
    interpolation_info["GtC_cond"]      = np.linalg.cond(dom.hps.H.Interp_mat)
    interpolation_info["CtG_error"]     = torch.norm(uu_gauss - uu_interG) / torch.norm(uu_gauss)
    interpolation_info["CtG_cond"]      = np.linalg.cond(dom.hps.H.Interp_mat_reverse)
    interpolation_info["redundant_var"] = maxVar

    file_loc = "test_interpolation_operator/test_results_p_" + str(p) + "_kh_" + str(kh) + ".pkl"
    #print("Pickling results to file %s"% (file_loc))
    #f = open(file_loc, "wb+")
    #pickle.dump(interpolation_info, f)
    #f.close()


# Test the accuracy of I_copy1 and I_copy2:
if (d==3 and 1==0):
    zz_copy1 = dom.hps.xx_ext[dom.hps.I_copy1,:]
    zz_copy2 = dom.hps.xx_ext[dom.hps.I_copy2,:]
    print("Numerical error of copy1 vs copy2 is:")
    print(torch.linalg.norm(zz_copy1 - zz_copy2) / torch.linalg.norm(zz_copy1))

    # Let's also check that I_copy1 isn't on the domain boundary:
    tol   = 0.01 * dom.hps.hmin
    I_dir = torch.where((zz_copy1[:,0] < dom.box_geom[0,0] + tol)
                      | (zz_copy1[:,0] > dom.box_geom[0,1] - tol)
                      | (zz_copy1[:,1] < dom.box_geom[1,0] + tol)
                      | (zz_copy1[:,1] > dom.box_geom[1,1] - tol)
                      | (zz_copy1[:,2] < dom.box_geom[2,0] + tol)
                      | (zz_copy1[:,2] > dom.box_geom[2,1] - tol))[0]

    print("Number of I_copy entries on domain boundary (should be 0): " + str(len(I_dir)))

# Test DtN_loc accuracy:
if d==3:
    # Here we'll test our DtN operators on a known function. First we define the known function and its
    # first order derivatives (this test is for Laplace only):
    def u_true(xx):
        #return torch.exp(xx[:,0]) * torch.sin(xx[:,1])
        return uu_dir_func_greens(3,xx,kh)
    
    def du1_true(xx):
        #return torch.exp(xx[:,0]) * torch.sin(xx[:,1])
        return du_dir_func_greens(0,3,xx,kh)
    
    def du2_true(xx):
        #return torch.exp(xx[:,0]) * torch.cos(xx[:,1])
        return du_dir_func_greens(1,3,xx,kh)
    
    def du3_true(xx):
        #return torch.zeros((xx.shape[0]))
        return du_dir_func_greens(2,3,xx,kh)

    size_face = dom.hps.q**2
    size_ext = 6 * size_face

    # Now we get our DtN maps. These don't depend on u_true
    device = torch.device('cpu')
    DtN_loc = dom.hps.get_DtNs(device,'build')

    # Here we get our dirichlet data, reshape it, and then multiply with DtNs to get our Neumann data
    uu_dir_gauss = u_true(dom.hps.xx_ext)

    #uu_neumann_from_A = torch.from_numpy(dom.A @ uu_dir_gauss)
    #uu_neumann_from_A = torch.reshape(uu_neumann_from_A, (DtN_loc.shape[0],-1))

    uu_dir_gauss = torch.reshape(uu_dir_gauss, (DtN_loc.shape[0],-1))
    uu_dir_gauss = torch.unsqueeze(uu_dir_gauss, -1)

    uu_neumann_approx = torch.matmul(DtN_loc, uu_dir_gauss)
    uu_neumann_approx = torch.squeeze(uu_neumann_approx)

    # Next we fold our spatial inputs and compute our actual neumann data:
    xx_folded = torch.reshape(dom.hps.xx_ext, (DtN_loc.shape[0],DtN_loc.shape[1], -1))

    uu_neumann = torch.zeros((xx_folded.shape[0], xx_folded.shape[1]))
    for i in range(xx_folded.shape[0]):
        uu_neumann[i,:size_face]              = -du1_true(xx_folded[i,:size_face,:])
        uu_neumann[i,size_face:2*size_face]   =  du1_true(xx_folded[i,size_face:2*size_face,:])
        uu_neumann[i,2*size_face:3*size_face] = -du2_true(xx_folded[i,2*size_face:3*size_face,:])
        uu_neumann[i,3*size_face:4*size_face] =  du2_true(xx_folded[i,3*size_face:4*size_face,:])
        uu_neumann[i,4*size_face:5*size_face] = -du3_true(xx_folded[i,4*size_face:5*size_face,:])
        uu_neumann[i,5*size_face:6*size_face] =  du3_true(xx_folded[i,5*size_face:6*size_face,:])

    print(torch.abs(uu_neumann_approx[0] - uu_neumann[0]) / torch.abs(uu_neumann[0]))

    print(torch.abs(uu_neumann_approx[1] - uu_neumann[1]) / torch.abs(uu_neumann[1]))

    print("\nRelative error of Neumann computation using tensor DtNs is")
    print(torch.linalg.norm(uu_neumann_approx - uu_neumann) / torch.linalg.norm(uu_neumann))
    #print("Relative error of Neumann computation using sparse matrix A is")
    #print(torch.linalg.norm(uu_neumann_from_A - uu_neumann) / torch.linalg.norm(uu_neumann))

    #import sys
    #np.set_printoptions(threshold=sys.maxsize)
    #print(dom.hps.H.Nxc[...,dom.hps.H.JJ.Jc])
    #print(dom.hps.H.JJ.Jc)

if (d==3 and 1==0):
    I_copy1  = dom.hps.I_copy1
    I_copy2  = dom.hps.I_copy2
    A_CC     = dom.A[I_copy1][:,I_copy1]
    A_CC_add = dom.A[I_copy2][:,I_copy2]

    error = 0
    for i in range(len(I_copy1)):
        for j in range(len(I_copy1)):
            error = error + np.abs(A_CC[i,j] - dom.A[I_copy1[i], I_copy1[j]])
            error = error + np.abs(A_CC_add[i,j] - dom.A[I_copy2[i], I_copy2[j]])

    print("Error in copies of A going into A_CC is " + str(error))
