import argparse                        # For parsing command-line arguments
import jax # for reasons
import torch                           # PyTorch library for tensor computations and GPU support
import numpy as np                     # For numerical operations
from time import time                  # For timing operations
torch.set_default_dtype(torch.double)  # Set default tensor type to double precision

from hpsmultidomain.driver_build import *
from hpsmultidomain.driver_solve import run_solver

from hpsmultidomain.domain_driver import *  # Importing domain driver utilities for PDE solving
from hpsmultidomain.built_in_funcs import *  # Importing built-in functions for specific PDEs or conditions
from hpsmultidomain.visualize_problem import visualize_problem
import pickle  # For serializing and saving results
import os  # For interacting with the operating system, e.g., environment variables

from hpsmultidomain.geom import *

# Initialize argument parser to define and read command-line arguments
parser = argparse.ArgumentParser("Call direct solver for 2D/3D domain.")
# Add expected command-line arguments for the script
parser.add_argument('--p',type=int,required=False)  # Polynomial order for certain methods
parser.add_argument('--n', type=int, required=False)  # Number of discretization points
parser.add_argument('--d', type=int, required=False, default=2)  # Spatial dimension of problem

parser.add_argument('--n0', type=int, required=False)  # Number of discretization points in x0
parser.add_argument('--n1', type=int, required=False)  # Number of discretization points in x1
parser.add_argument('--n2', type=int, required=False)  # Number of discretization points in x2

parser.add_argument('--p0', type=int, required=False)  # Polynomail order in x0
parser.add_argument('--p1', type=int, required=False)  # Polynomail order in x1
parser.add_argument('--p2', type=int, required=False)  # Polynomail order in x2

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
parser.add_argument('--num_timesteps', type=int, required=False)  # checks if we have a given number of timesteps (only needed for convection-diffusion)

# Solver and computational specifics
parser.add_argument('--solver',type=str,required=False)  # Solver to use
parser.add_argument('--sparse_assembly',type=str,required=False, default='reduced_gpu')  # Assembly method
parser.add_argument('--pickle',type=str,required=False)  # File path for pickling results
parser.add_argument('--store_sol',action='store_true')  # Flag to store solution
parser.add_argument('--disable_cuda',action='store_true')  # Flag to disable CUDA
parser.add_argument('--periodic_bc', action='store_true')  # Flag for periodic boundary conditions

# Components tests:
parser.add_argument('--test_components', type=bool, required=False, default=False) # Test discretization components such as interpolation

# Visualize solution?:
parser.add_argument('--visualize', type=str, required=False, default=False)

args = parser.parse_args()  # Parse arguments from command line

# Check if we test components:
test_components = args.test_components
param_map = None

# Extract and setup basic parameters from parsed arguments:
if args.d == 2:
    if args.n is not None:
        args.n = np.array([args.n, args.n])
    elif ((args.n0 is not None) and (args.n1 is not None)):
        args.n = np.array([args.n0, args.n1])
    else:
        ValueError("Need to set either n or (for 2D only) n0,n1")

    if args.p is not None:
        args.p = np.array([args.p, args.p])
    elif ((args.p0 is not None) and (args.p1 is not None)):
        args.p = np.array([args.p0, args.p1])
    else:
        ValueError("Need to set either p or (for 2D only) p0,p1")
elif args.d == 3:
    if args.n is not None:
        args.n = np.array([args.n, args.n, args.n])
    elif ((args.n0 is not None) and (args.n1 is not None) and (args.n2 is not None)):
        args.n = np.array([args.n0, args.n1, args.n2])
    else:
        ValueError("Need to set either n or (for 3D only) n0,n1,n2")

    if args.p is not None:
        args.p = np.array([args.p, args.p, args.p])
    elif ((args.p0 is not None) and (args.p1 is not None) and (args.p2 is not None)):
        args.p = np.array([args.p0, args.p1, args.p2])
    else:
        ValueError("Need to set either p or (for 3D only) p0,p1,p2")
else:
    ValueError("dimension d must be 2 or 3")

d = args.d
box = torch.tensor([[0,0],[args.box_xlim,args.box_ylim]])  # Domain geometry tensor
if d==3:
    box = torch.tensor([[0,0,0],[args.box_xlim,args.box_ylim,args.box_zlim]])

box_geom = BoxGeometry(box)

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
op, param_map, inv_param_map, curved_domain, kh, delta_t, num_timesteps = configure_pde_domain(args)
    
##### Set the domain and discretization parameters
if (args.p is None):
    raise ValueError('HPS selected but p not provided')
p = args.p
npan = args.n / (p-2)
if args.d == 2:
    a = np.array([args.box_xlim, args.box_ylim]) / (2*npan) # a is now an array
else: #args.d == 3
    a = np.array([args.box_xlim, args.box_ylim, args.box_zlim]) / (2*npan) # a is now an array

print("p = ", p)
print("a = ", a)

# Inilialize the domain driver object - we do this separately from
# build_operator_with_info so that, in the future, we could experiment
# with different operators in an already-built domain
dom = Domain_Driver(box_geom,op,kh,a,p=p,d=d,periodic_bc = args.periodic_bc)

# Build operator based on specified parameters and solver information
print(args.sparse_assembly)
build_info = build_operator_with_info(dom, args, box_geom, kh)

################################# SOLVE PDE ###################################
# Solve the PDE with specified configurations and print results
uu_dir,uu_sol,res,true_res,resloc_hps,toc_solve,forward_bdry_error,reverse_bdry_error,solve_info = run_solver(dom, args, curved_domain, kh, param_map, delta_t, num_timesteps)
print(uu_sol.shape)


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

# Optional: visualization of computed solution
if args.visualize:
    visualize_problem(dom, curved_domain, param_map, uu_sol, p, args.visualize, kh, d=args.d, n=args.n[0], f=convection_steady_state_patch)
