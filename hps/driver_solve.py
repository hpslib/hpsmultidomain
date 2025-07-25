import torch                           # PyTorch library for tensor computations and GPU support
import numpy as np                     # For numerical operations

from domain_driver import *  # Importing domain driver utilities for PDE solving
from built_in_funcs import *  # Importing built-in functions for specific PDEs or conditions

def run_solver(dom, args, curved_domain, kh=0, param_map=None, delta_t=0, num_timesteps=1):
    """
    Once we have defined a domain and constrcuted a sparse system for solving the boundaries of
    the subdomains, we use this function to solve.
    """
    print("SOLVE RESULTS")
    solve_info = dict()
    d = args.d

    # We determine the Dirichlet BC and body load based on our pde and domain:
    if (args.bc == 'free_space'):
        if (args.pde == 'bfield_constant'):
            ff_body = None; known_sol = True
            
            if (not curved_domain):
                uu_dir = lambda xx: uu_dir_func_greens(d, xx,kh)
            else:
                uu_dir = lambda xx: uu_dir_func_greens(d, param_map(xx),kh) #, center=torch.tensor([-3.1,+3.,+3.2]))
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

    elif (args.bc == 'zeros'):
        ff_body = None; known_sol = False
        
        zeros_func = lambda xx: torch.zeros(xx.shape[0],1)
        if (not curved_domain):
            uu_dir = lambda xx: zeros_func(xx)
        else:
            uu_dir = lambda xx: zeros_func(param_map(xx))  

        if (args.pde == 'bfield_gravity'):
            ff_body = lambda xx: -torch.ones(xx.shape[0],1,device=xx.device)
            
    elif (args.bc == 'log_dist'):
        
        if ((args.pde == 'poisson') or (args.pde == 'mixed')):
            assert kh == 0
            
            if not curved_domain:
                uu_dir = lambda xx: uu_dir_func_greens(d, xx,kh)
                if args.periodic_bc:
                    uu_dir = lambda xx: uu_dir_func_periodic(xx)
                ff_body = None
                known_sol = True
            elif not args.periodic_bc:
                uu_dir = lambda xx: uu_dir_func_greens(d, param_map(xx),kh)
                ff_body = None
                known_sol = True
            else:
                uu_dir = lambda xx: uu_dir_func_periodic(param_map(xx),kh)
                ff_body = None
                known_sol = True
        else:
            raise ValueError
    elif (args.bc == 'convection_diffusion'):
        if d == 2:
            raise ValueError("Convection_diffusion is 3D only")
        if (args.pde == 'convection_diffusion'):
            # Dirichlet BC is from the time step we are solving for now:
            uu_dir        = lambda xx: torch.zeros(xx.shape[0], 1)
            known_sol     = False
            ff_body       = lambda xx: convection_u_init(xx)
        else:
            raise ValueError
    elif (args.bc == 'parabolic_heat'):
        if d == 2:
            raise ValueError("parabolic_heat is 3D only")
        if (args.pde == 'parabolic_heat'):
            # Dirichlet BC is from the time step we are solving for now:
            uu_dir        = lambda xx: uu_dir_func_parabolic_heat(xx, delta_t)
            known_sol     = True
            ff_body       = lambda xx: -uu_dir_func_parabolic_heat(xx, 0)
        else:
            raise ValueError
    else:
        raise ValueError("invalid bc")


    if (args.solver == 'slabLU'):
        raise ValueError("this code is not included in this version")


    ff_body_func = ff_body
    ff_body_vec  = None
    total_toc_system_solve = 0
    total_toc_leaf_solve   = 0

    # For elliptic PDEs this loop runs only once. For parabolic PDEs it executes for the desired number of time steps,
    # saving the final result.
    for i in range(num_timesteps):
        print("\nFOR the %d timestep:\n" % i)

        # Modifying body load and DBC for new time step if needed
        if i > 0:
            ff_body_vec  = uu_sol
            ff_body_func = None
            # Update the Dirichlet BC for the new timestep (parabolic heat only):
            if (args.bc == 'parabolic_heat'):
                uu_dir = lambda xx: uu_dir_func_parabolic_heat(xx, delta_t*(i+1))

        # Solving the system and gathering both the system solve and leaf solve times:
        uu_sol,res, true_res,resloc_hps,toc_system_solve,toc_leaf_solve,forward_bdry_error,reverse_bdry_error = dom.solve(uu_dir,ff_body_func=ff_body_func,ff_body_vec=ff_body_vec,known_sol=known_sol)
        total_toc_system_solve += toc_system_solve
        total_toc_leaf_solve += toc_leaf_solve
        
        # Observing norm change in u for parabolic problems to see if unrealistic accumulation happens:
        if i > 0:
            sol_norm = torch.linalg.norm(uu_sol, ord=1)
            print("With current vector 1-norm of " + str(sol_norm.item()))

    sol_norm = torch.linalg.norm(uu_sol)

    print("\n\n")

    # Saving solve info:
    if (args.solver == 'superLU'):
        print("\t--SuperLU solved Ax=b residual %5.2e with known solution residual %5.2e and resloc_HPS %5.2e in time %5.2f s"\
            %(res,true_res,resloc_hps,total_toc_system_solve+total_toc_leaf_solve))
        solve_info['res_solve_superLU']            = res
        solve_info['trueres_solve_superLU']        = true_res
        solve_info['resloc_hps_solve_superLU']     = resloc_hps
        solve_info['toc_solve_superLU']            = total_toc_system_solve
        solve_info['toc_solve_leaf']               = total_toc_leaf_solve

        solve_info['forward_bdry_error'] = forward_bdry_error
        solve_info['reverse_bdry_error'] = reverse_bdry_error
    else:
        print("\t--Builtin solver %s solved Ax=b residual %5.2e with known solution residual %5.2e and resloc_HPS %5.2e in time %5.2f s"\
            %(args.solver,res,true_res,resloc_hps,total_toc_system_solve+total_toc_leaf_solve))
        solve_info['res_solve_petsc']        = res
        solve_info['trueres_solve_petsc']    = true_res
        solve_info['resloc_hps_solve_petsc'] = resloc_hps
        solve_info['toc_solve_petsc']        = total_toc_system_solve
        solve_info['toc_solve_leaf']         = total_toc_leaf_solve

        solve_info['forward_bdry_error'] = forward_bdry_error
        solve_info['reverse_bdry_error'] = reverse_bdry_error

    print("The solution residual is only relevant for problems that do not have a known true solution.")

    return uu_dir, uu_sol,res, sol_norm,resloc_hps,toc_system_solve+toc_leaf_solve,forward_bdry_error,reverse_bdry_error, solve_info
