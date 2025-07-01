import argparse                        # For parsing command-line arguments
import torch                           # PyTorch library for tensor computations and GPU support
import numpy as np                     # For numerical operations
from time import time                  # For timing operations
torch.set_default_dtype(torch.double)  # Set default tensor type to double precision

from driver_build import *
from driver_solve import run_solver

from domain_driver import *  # Importing domain driver utilities for PDE solving
from built_in_funcs import *  # Importing built-in functions for specific PDEs or conditions
from visualize_problem import visualize_problem
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
parser.add_argument('--visualize', type=bool, required=False, default=False)

args = parser.parse_args()  # Parse arguments from command line

# Check if we test components:
test_components = args.test_components
param_map = None

# Extract and setup basic parameters from parsed arguments
redundant_n = (args.n is not None) and ((args.n0 is not None) or (args.n1 is not None) or (args.n2 is not None))
if redundant_n:
    ValueError("Cannot have n and n0,n1,n2 set")
elif args.n is not None:
    args.n = np.array([args.n, args.n, args.n])
elif ((args.n0 is not None) and (args.n1 is not None) and (args.n2 is not None)):
    args.n = np.array([args.n0, args.n1, args.n2])
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
op, param_map, inv_param_map, curved_domain, kh, delta_t, num_timesteps = configure_pde_domain(args)
    
##### Set the domain and discretization parameters
if (args.p is None):
    raise ValueError('HPS selected but p not provided')
p = args.p
npan = args.n / (p-2)
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
    visualize_problem(dom, curved_domain, param_map, uu_sol, p, kh)
