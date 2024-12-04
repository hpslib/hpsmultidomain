import argparse                        # For parsing command-line arguments
import torch                           # PyTorch library for tensor computations and GPU support
import numpy as np                     # For numerical operations
from time import time                  # For timing operations
torch.set_default_dtype(torch.double)  # Set default tensor type to double precision

from driver_build import *
from driver_solve import run_solver

from domain_driver import *  # Importing domain driver utilities for PDE solving
from built_in_funcs import *  # Importing built-in functions for specific PDEs or conditions
import pickle  # For serializing and saving results
import os  # For interacting with the operating system, e.g., environment variables

# Initialize argument parser to define and read command-line arguments
parser = argparse.ArgumentParser("Call direct solver for 2D/3D domain.")
# Add expected command-line arguments for the script
parser.add_argument('--p',type=int,required=False)  # Polynomial order for certain methods
parser.add_argument('--n', type=int, required=False)  # Number of discretization points
parser.add_argument('--d', type=int, required=False, default=2)  # Spatial dimension of problem


parser.add_argument('--n0', type=int, required=False)  # Number of discretization points in x0
parser.add_argument('--n1', type=int, required=False)  # Number of discretization points in x1
parser.add_argument('--n2', type=int, required=False)  # Number of discretization points in x2

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
parser.add_argument('--delta_t', type=float, required=False)  # checks if we have a given time step (only needed for convection-diffusion)

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
redundant_n = (args.n is not None) and ((args.n0 is not None) or (args.n1 is not None) or (args.n2 is not None))
if redundant_n:
    ValueError("Cannot have n and n0,n1,n2 set")
elif args.n is not None:
    n = np.array([args.n, args.n, args.n])
elif ((args.n0 is not None) and (args.n1 is not None) and (args.n2 is not None)):
    n = np.array([args.n0, args.n1, args.n2])
else:
    ValueError("Need to set either n or n0,n1,n2")

d = args.d
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

################################# BUILD OPERATOR #########################

# Configure the PDE operator and domain based on specified arguments
# If we have a square domain (and thus do not need parameter maps), then
# param_map, inv_param_map = None and curved_domain = False.
# If there is no oscillation then kh = 0.
op, param_map, inv_param_map, curved_domain, kh, delta_t = configure_pde_domain(args)
    
##### Set the domain and discretization parameters
if (args.p is None):
    raise ValueError('HPS selected but p not provided')
p = args.p
npan = n / (p-2)
a = 1/(2*npan) # a is now an array

# Inilialize the domain driver object - we do this separately from
# build_operator_with_info so that, in the future, we could experiment
# with different operators in an already-built domain
dom = Domain_Driver(box_geom,op,kh,a,p=p,d=d,periodic_bc = args.periodic_bc)

# Build operator based on specified parameters and solver information
print(args.sparse_assembly)
build_info = build_operator_with_info(dom, args, box_geom, kh)

################################# SOLVE PDE ###################################
# Solve the PDE with specified configurations and print results
uu_dir,uu_sol,res,true_res,resloc_hps,toc_solve,forward_bdry_error,reverse_bdry_error,solve_info = run_solver(dom, args, curved_domain, kh, param_map, delta_t)

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
    #build_info.update(interpolation_info)
    #build_info.update(dtn_info)
    #print(build_info)
    pickle.dump(build_info,f)
    #pickle.dump(solve_info,f)
    f.close()

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
"""

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

    size_face = DtN_loc[0].shape[0] // 6

    dtn_abs = torch.abs(DtN_loc[0])
    largest = torch.max(dtn_abs).item()
    smallest = torch.min(dtn_abs).item()

    print("DtN shape is " + str(DtN_loc[0].shape))
    print("DtN Condition number is " + str(dtn_cond))
    print("Largest / smallest is " + str(largest) + " / " + str(smallest))

    for j in range(1): #DtN_loc.shape[0]):
        print("Entry " + str(j) + ". Recall order is L R D U B F")
        for i in range(6):
            dtn_partial_cond = torch.linalg.cond(DtN_loc[j, i*size_face:(i+1)*size_face])
            print("Condition number for this face is " + str(dtn_partial_cond.item()))

    
    print("\nRelative error of Neumann computation using tensor DtNs is")
    print(neumann_tensor_error)
    print("Relative error of Neumann computation using sparse matrix A is")
    print(neumann_sparse_error)
    

    #dtn_info = dict()
    #dtn_info["neumann_tensor_error"] = neumann_tensor_error
    #dtn_info["neumann_sparse_error"] = neumann_sparse_error
    #dtn_info["dtn_cond"] = dtn_cond
    

    center=np.array([-1.1,+1.,+1.2])
    
    #xx = dom.hps.grid_xx.flatten(start_dim=0,end_dim=-2)
    # result = uu_sol
    xx = dom.hps.grid_ext

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

    Jx = torch.tensor(dom.hps.H.JJ.Jxreorder)

    result = uu_sol[:,Jx].flatten()

    max_result = torch.linalg.norm(result, ord=np.inf)

    ax.view_init(azim=-30)
    plt.rc('text',usetex=True)
    plt.rc('font',**{'family':'serif','size':14})
    plt.rc('text.latex',preamble=r'\usepackage{amsfonts,bm}')
    sc = ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals, c=result, marker='o', cmap="seismic", vmin=-max_result, vmax=max_result)
    plt.title("Result of Helmholtz Equation on Curved Domain, K = " + str(kh))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(sc, shrink=0.5)
    plt.rcParams['figure.figsize'] = [14, 6]
    plt.savefig("3D-domain-faces.pdf")
    plt.show()
    
    
"""
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
    #build_info.update(interpolation_info)
    #build_info.update(dtn_info)
    #print(build_info)
    pickle.dump(build_info,f)
    #pickle.dump(solve_info,f)
    f.close()
"""
