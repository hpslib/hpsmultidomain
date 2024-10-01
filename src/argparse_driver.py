import argparse                        # For parsing command-line arguments
import torch                           # PyTorch library for tensor computations and GPU support
import numpy as np                     # For numerical operations
from time import time                  # For timing operations
torch.set_default_dtype(torch.double)  # Set default tensor type to double precision

from driver_solve import run_solver

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
param_map = None

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
    def c(xx, center=torch.tensor([-1.1,+1.,+1.2])):
        dd0  = xx[:,0] - center[0]
        dd1  = xx[:,1] - center[1]
        dd2  = xx[:,2] - center[2]
        ddsq = dd0*dd0 + dd1*dd1 + dd2*dd2 # r^2
        r4   = ddsq * ddsq
        return (dd0*dd1 + dd0*dd2 + dd1*dd2) / r4

    op = pdo.PDO_3d(pdo.ones,pdo.ones,pdo.ones,
                    c12=pdo.const(c=1/3),c13=pdo.const(c=1/3),c23=pdo.const(c=1/3),
                    c=c)
    kh = 0
    curved_domain = False

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

        """print(op)
        print(param_map)
        print(op.c)
        print(op.c1)
        print(op.c2)
        print(op.c3)
        print(op.c11)
        print(op.c22)
        print(op.c33)
        print(op.c12)
        print(op.c13)
        print(op.c23)"""
        
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
    
elif args.pde == "convection_diffusion":
    if (args.ppw is not None):
        raise ValueError
    
    # convection_diffusion operator
    if d==2:
        print("convection_diffusion is 3D only")
        raise ValueError
    if d==3:
        op = pdo.PDO_3d(pdo.ones,pdo.ones,pdo.ones,c3=pdo.const(c=-63.))
    kh = 0
    curved_domain = False


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

################################# SOLVE PDE ###################################
# Solve the PDE with specified configurations and print results

uu_sol,res, true_res,resloc_hps,toc_solve,forward_bdry_error,reverse_bdry_error, solve_info = run_solver(dom, args, curved_domain, kh, param_map)

# Optional: Store solution and/or pickle results for later use
if (args.store_sol):
    print("\t--Storing solution")
    XX = dom.hps.xx_tot
    solve_info['xx']        = XX
    solve_info['sol']       = uu_sol

################################# EVALUATING SYSTEM COMPONENTS ###################################
# Evaluate certain parts of the 3D problem 
"""
if (d==3):
    interpolation_info = dict()
    # First we generate the arrays of Chebyshev and Gaussian nodes:
    cheb_ext  = torch.from_numpy(dom.hps.H.zz.T[dom.hps.H.JJ.Jxreorder])
    gauss_ext = torch.from_numpy(dom.hps.H.zzG)

    uu_cheb  = uu_dir(cheb_ext)
    uu_gauss = uu_dir(gauss_ext)
    uu_interC = torch.from_numpy(dom.hps.H.Interp_mat) @ uu_gauss
    uu_interG = torch.from_numpy(dom.hps.H.Interp_mat_reverse) @ uu_cheb

    GtC_error = torch.norm(uu_cheb - uu_interC) / torch.norm(uu_cheb)
    GtC_error = GtC_error.item()
    GtC_cond  = np.linalg.cond(dom.hps.H.Interp_mat)
    CtG_error = torch.norm(uu_gauss - uu_interG) / torch.norm(uu_gauss)
    CtG_error = CtG_error.item()
    CtG_cond  = np.linalg.cond(dom.hps.H.Interp_mat_reverse)

    print("Relative error of Gaussian-to-Chebyshev interpolation is:")
    print(GtC_error)
    print("With condition number and shape:")
    print(GtC_cond, dom.hps.H.Interp_mat.shape)
    print("Relative error of Chebyshev-to-Gaussian interpolation is:")
    print(CtG_error)
    print("With condition number and shape:")
    print(CtG_cond, dom.hps.H.Interp_mat_reverse.shape)

    
    # Check if edges / corners are equal as expected:
    u, c = np.unique(dom.hps.H.JJ.Jxreorder, return_counts=True)
    dup = u[c > 1]
    maxVar = 0
    for elem in dup:
        variance = torch.std(uu_interC[dom.hps.H.JJ.Jxreorder == elem])
        maxVar = np.max([maxVar, variance.item()])
    print("Largest variance between redundant values is " + str(maxVar))

    interpolation_info["GtC_error"]     = GtC_error
    interpolation_info["GtC_cond"]      = GtC_cond
    interpolation_info["CtG_error"]     = CtG_error
    interpolation_info["CtG_cond"]      = CtG_cond
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
        if args.domain=='square':
            return uu_dir_func_greens(3,xx,kh)
        else:
            return uu_dir_func_greens(3,param_map(xx),kh)
    
    def du1_true(xx):
        #return torch.exp(xx[:,0]) * torch.sin(xx[:,1])
        if args.domain=='square':
            return du_dir_func_greens(0,3,xx,kh)
        else:
            return du_dir_func_greens(0,3,param_map(xx),kh)
    
    def du2_true(xx):
        #return torch.exp(xx[:,0]) * torch.cos(xx[:,1])
        if args.domain=='square':
            return du_dir_func_greens(1,3,xx,kh)
        else:
            return du_dir_func_greens(1,3,param_map(xx),kh)
    
    def du3_true(xx):
        #return torch.zeros((xx.shape[0]))
        if args.domain=='square':
            return du_dir_func_greens(2,3,xx,kh)
        else:
            return du_dir_func_greens(2,3,param_map(xx),kh)

    size_face = dom.hps.q**2
    size_ext = 6 * size_face

    # Now we get our DtN maps. These don't depend on u_true
    device = torch.device('cpu')
    DtN_loc = dom.hps.get_DtNs(device,'build')

    # Here we get our dirichlet data, reshape it, and then multiply with DtNs to get our Neumann data
    uu_dir_gauss = u_true(dom.hps.xx_ext)

    uu_neumann_from_A = torch.from_numpy(dom.A @ uu_dir_gauss)
    uu_neumann_from_A = torch.reshape(uu_neumann_from_A, (DtN_loc.shape[0],-1))

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

    #print(torch.abs(uu_neumann_approx[0] - uu_neumann[0]) / torch.abs(uu_neumann[0]))
    #print(torch.abs(uu_neumann_approx[1] - uu_neumann[1]) / torch.abs(uu_neumann[1]))

    neumann_tensor_error = torch.linalg.norm(uu_neumann_approx - uu_neumann) / torch.linalg.norm(uu_neumann)
    neumann_tensor_error = neumann_tensor_error.item()
    neumann_sparse_error = torch.linalg.norm(uu_neumann_from_A - uu_neumann) / torch.linalg.norm(uu_neumann)
    neumann_sparse_error = neumann_sparse_error.item()
    dtn_cond = torch.linalg.cond(DtN_loc[0])
    dtn_cond = dtn_cond.item()

    print("DtN Condition number:")
    print(dtn_cond)
    print("\nRelative error of Neumann computation using tensor DtNs is")
    print(neumann_tensor_error)
    print("Relative error of Neumann computation using sparse matrix A is")
    print(neumann_sparse_error)

    dtn_info = dict()
    dtn_info["neumann_tensor_error"] = neumann_tensor_error
    dtn_info["neumann_sparse_error"] = neumann_sparse_error
    dtn_info["dtn_cond"] = dtn_cond
    
    center=np.array([-1.1,+1.,+1.2])
    
    xx = dom.hps.grid_xx.flatten(start_dim=0,end_dim=-2)
    if curved_domain:
        xx = param_map(xx)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    sequence_containing_x_vals = xx[:,0] - center[0]
    sequence_containing_y_vals = xx[:,1] - center[1]
    sequence_containing_z_vals = xx[:,2] - center[2]

    norms = np.sqrt(sequence_containing_x_vals*sequence_containing_x_vals
                   + sequence_containing_y_vals*sequence_containing_y_vals
                   + sequence_containing_z_vals*sequence_containing_z_vals)

    result = uu_sol.flatten()

    true_vals = uu_dir(xx)
    true_vals = true_vals.squeeze(-1)

    rel_errors = torch.abs(result - true_vals) / (torch.abs(true_vals) + 1)

    #print(rel_errors)

    sc = ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals, c=rel_errors, marker='o')
    plt.colorbar(sc)
    plt.show()
    
    

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

if (args.pickle is not None):
    file_loc = args.pickle
    print("Pickling results to file %s"% (file_loc))
    f = open(file_loc,"wb+")
    build_info.update(solve_info)
    build_info.update(interpolation_info)
    build_info.update(dtn_info)
    #print(build_info)
    pickle.dump(build_info,f)
    #pickle.dump(solve_info,f)
    f.close()

"""