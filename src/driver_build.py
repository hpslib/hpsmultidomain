import torch                           # PyTorch library for tensor computations and GPU support
import numpy as np                     # For numerical operations

from domain_driver import *  # Importing domain driver utilities for PDE solving
from built_in_funcs import *  # Importing built-in functions for specific PDEs or conditions

#
# Configure the PDE operator and domain based on specified arguments
#
def configure_pde_domain(args):
    param_map     = None
    inv_param_map = None
    curved_domain = False

    kh = 0
    delta_t = 0

    if ((args.pde == 'poisson') and (args.domain == 'square')):
        if (args.ppw is not None):
            raise ValueError
        
        # laplace operator
        op = pdo.PDO_2d(pdo.ones,pdo.ones)
        if args.d==3:
            op = pdo.PDO_3d(pdo.ones,pdo.ones,pdo.ones)
        kh = 0
        curved_domain = False

    elif ((args.pde == 'mixed') and (args.domain == 'square')):
        if (args.ppw is not None):
            raise ValueError

        # operator:
        if args.d==2:
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
            nwaves = int(args.n/args.ppw)
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
            if args.d==3:
                op = pdo.PDO_3d(pdo.ones,pdo.ones,pdo.ones,c=c)
            
        elif (args.domain == 'curved'):
            
            op, param_map, \
            inv_param_map = pdo.get_param_map_and_pdo('sinusoidal', bfield, kh, d=args.d)
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
        
    elif args.pde == "convection_diffusion":
        if (args.ppw is not None):
            raise ValueError
        
        # convection_diffusion operator
        if args.d==2:
            print("convection_diffusion is 3D only")
            raise ValueError
        if args.d==3:
            delta_t = args.delta_t
            if delta_t is None:
                raise ValueError("delta_t must be specified for parabolic problem")
            op = pdo.PDO_3d(pdo.const(c=-delta_t),pdo.const(c=-delta_t),pdo.const(c=-delta_t),
                            c3=pdo.const(c=-2*delta_t),
                            c=pdo.const(c=-1))
        kh = 0
        curved_domain = False

    elif args.pde == "parabolic_heat":
        if (args.ppw is not None):
            raise ValueError

        # parabolic_heat operator
        if args.d==2:
            raise ValueError("parabolic_heat is 3D only")
        if args.d==3:
            delta_t = args.delta_t
            if delta_t is None:
                raise ValueError("delta_t must be specified for parabolic problem")
            op = pdo.PDO_3d(pdo.const(c=-delta_t),pdo.const(c=-delta_t),pdo.const(c=-delta_t),c=pdo.const(c=-1))
        kh = 0
        curved_domain = False


    else:
        raise ValueError

    return op, param_map, inv_param_map, curved_domain, kh, delta_t

#
# Given a domain decomposition, constructs the operator needed for the HPS computation:
#
def build_operator_with_info(dom, args, box_geom, kh=0):
    N = (args.p-2) * (args.p*dom.hps.n[0]*dom.hps.n[1] + dom.hps.n[0] + dom.hps.n[1])

    build_info = dom.build(sparse_assembly=args.sparse_assembly,\
                            solver_type = args.solver, verbose=True)
    build_info['N']    = N
    build_info['n']    = args.n
    build_info['pde']  = args.pde
    build_info['bc']   = args.bc
    build_info['domain'] = args.domain
    build_info['solver'] = args.solver
    build_info['sparse_assembly'] = args.sparse_assembly
    build_info['box_geom'] = box_geom
    build_info['kh']   = kh 
    build_info['periodic_bc'] = args.periodic_bc
    build_info['a'] = dom.hps.a
    build_info['p'] = args.p
    build_info['delta_t'] = args.delta_t

    return build_info