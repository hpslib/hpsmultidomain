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
parser = argparse.ArgumentParser("Call direct solver for 2D domain.")
# Add expected command-line arguments for the script
parser.add_argument('--p',type=int,required=False)  # Polynomial order for certain methods
parser.add_argument('--n', type=int, required=True)  # Number of discretization points

# Arguments defining the PDE problem
parser.add_argument('--pde', type=str, required=True)  # Type of PDE
parser.add_argument('--domain', type=str, required=True)  # Domain shape
parser.add_argument('--box_xlim', type=float, required=False, default=1.0)  # Domain x limits
parser.add_argument('--box_ylim', type=float, required=False, default=1.0)  # Domain y limits

# Boundary condition and problem specifics
parser.add_argument('--bc', type=str, required=True)  # Boundary condition
parser.add_argument('--ppw',type=int, required=False)  # Points per wavelength for oscillatory problems
parser.add_argument('--nwaves',type=float, required=False)  # Number of wavelengths

# Solver and computational specifics
parser.add_argument('--solver',type=str,required=False)  # Solver to use
parser.add_argument('--sparse_assembly',type=str,required=False, default='reduced_gpu')  # Assembly method
parser.add_argument('--pickle',type=str,required=False)  # File path for pickling results
parser.add_argument('--store_sol',action='store_true')  # Flag to store solution
parser.add_argument('--disable_cuda',action='store_true')  # Flag to disable CUDA
parser.add_argument('--periodic_bc', action='store_true')  # Flag for periodic boundary conditions

args = parser.parse_args()  # Parse arguments from command line

# Extract and setup basic parameters from parsed arguments
n = args.n;
box_geom = torch.tensor([[0,args.box_xlim],[0,args.box_ylim]])  # Domain geometry tensor

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
    kh = 0
    curved_domain = False

elif ( (args.pde).startswith('bfield')):
    ppw_set = args.ppw is not None
    nwaves_set = args.nwaves is not None
    
    if ((not ppw_set and not nwaves_set)):
        raise ValueError('oscillatory bfield chosen but ppw and nwaves NOT set')
    elif (ppw_set and nwaves_set):
        raise ValueError('ppw and nwaves both set')
    elif (ppw_set):
        nwaves = int(n/args.ppw)
        kh = (nwaves+0.03)*2*np.pi+1.8;
    else:
        nwaves = args.nwaves
        kh = (nwaves)*2*np.pi
    
      
    if (args.pde == 'bfield_constant'):
        bfield = bfield_constant
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
        
    elif (args.domain == 'curved'):
        
        op, param_map, \
        inv_param_map = pdo.get_param_map_and_pdo('sinusoidal', bfield, kh)
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
npan = n / (p-2); a = 1/(2*npan)
dom = Domain_Driver(box_geom,op,\
                    kh,a,p=p,periodic_bc = args.periodic_bc)
N = (p-2) * (p*dom.hps.n[0]*dom.hps.n[1] + dom.hps.n[0] + dom.hps.n[1])

################################# BUILD OPERATOR #########################
# Build operator based on specified parameters and solver information
    
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
print("SOLVE RESULTS")
solve_info = dict()

if (args.bc == 'free_space'):
    assert args.pde == 'bfield_constant'
    ff_body = None; known_sol = True
    
    if (not curved_domain):
        uu_dir = lambda xx: uu_dir_func_greens(xx,kh)
    else:
        uu_dir = lambda xx: uu_dir_func_greens(param_map(xx),kh)
        
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
    
    if (args.pde == 'poisson'):
        assert kh == 0
        assert (not curved_domain)

        uu_dir  = lambda xx: uu_dir_func_greens(xx,kh)
        ff_body = None
        known_sol = True
    else:
        raise ValueError
else:
    raise ValueError("invalid bc")

if (args.solver == 'slabLU'):
    
    raise ValueError("this code is not included in this version")
    
elif (args.solver == 'superLU'):

    uu_sol,res, true_res,resloc_hps,toc_solve = dom.solve(uu_dir,ff_body,known_sol=known_sol)
    uu_sol,res, true_res,resloc_hps,toc_solve = dom.solve(uu_dir,ff_body,known_sol=known_sol)

    print("\t--SuperLU solved Ax=b residual %5.2e with known solution residual %5.2e and resloc_HPS %5.2e in time %5.2f s"\
          %(res,true_res,resloc_hps,toc_solve))
    solve_info['res_solve_superLU']            = res
    solve_info['trueres_solve_superLU']        = true_res
    solve_info['resloc_hps_solve_superLU']     = resloc_hps
    solve_info['toc_solve_superLU']            = toc_solve

else:

    uu_sol,res, true_res,resloc_hps,toc_solve = dom.solve(uu_dir,ff_body,known_sol=known_sol)
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
    pickle.dump(build_info,f)
    pickle.dump(solve_info,f)
    f.close()