import torch                           # PyTorch library for tensor computations and GPU support
import numpy as np                     # For numerical operations

from domain_driver import *  # Importing domain driver utilities for PDE solving
from built_in_funcs import *  # Importing built-in functions for specific PDEs or conditions

def run_solver(dom, args, curved_domain, kh=0, param_map=None):
    print("SOLVE RESULTS")
    solve_info = dict()
    num_timesteps = 1
    d = args.d

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
    elif (args.bc == 'convection_diffusion'):
        if d == 2:
            print("Convection_diffusion is 3D only")
            raise ValueError
        if (args.pde == 'convection_diffusion'):
            uu_dir        = lambda xx: uu_dir_func_convection(xx)
            known_sol     = True
            num_timesteps = 10
            delta_t       = 1.0
            ff_body       = lambda xx: -delta_t * uu_dir_func_convection(xx)
        else:
            raise ValueError
    else:
        raise ValueError("invalid bc")


    if (args.solver == 'slabLU'):
        raise ValueError("this code is not included in this version")

    for i in range(num_timesteps):
        ff_body_func = ff_body
        ff_body_vec  = None
        if (i > 0):
            ff_body_vec  = -delta_t * uu_sol
            ff_body_func = None
        uu_sol,res, true_res,resloc_hps,toc_solve,forward_bdry_error,reverse_bdry_error = dom.solve(uu_dir,ff_body_func=ff_body_func,ff_body_vec=ff_body_vec,known_sol=known_sol)

    if (args.solver == 'superLU'):
        print("\t--SuperLU solved Ax=b residual %5.2e with known solution residual %5.2e and resloc_HPS %5.2e in time %5.2f s"\
            %(res,true_res,resloc_hps,toc_solve))
        solve_info['res_solve_superLU']            = res
        solve_info['trueres_solve_superLU']        = true_res
        solve_info['resloc_hps_solve_superLU']     = resloc_hps
        solve_info['toc_solve_superLU']            = toc_solve

        solve_info['forward_bdry_error'] = forward_bdry_error
        solve_info['reverse_bdry_error'] = reverse_bdry_error
    else:
        print("\t--Builtin solver %s solved Ax=b residual %5.2e with known solution residual %5.2e and resloc_HPS %5.2e in time %5.2f s"\
            %(args.solver,res,true_res,resloc_hps,toc_solve))
        solve_info['res_solve_petsc']            = res
        solve_info['trueres_solve_petsc']        = true_res
        solve_info['resloc_hps_solve_petsc']     = resloc_hps
        solve_info['toc_solve_petsc']            = toc_solve

        solve_info['forward_bdry_error'] = forward_bdry_error
        solve_info['reverse_bdry_error'] = reverse_bdry_error

    return uu_dir, uu_sol,res, true_res,resloc_hps,toc_solve,forward_bdry_error,reverse_bdry_error, solve_info