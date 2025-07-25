import pickle
import os
import torch
os.environ['LANG']='en_US.UTF-8'

torch.set_default_dtype(torch.double)  # Ensure all torch tensors are double precision for accuracy

def run_test_via_argparse(domain, pde, bc, disc_n, p, box_xlim=1.0, box_ylim=1.0, periodic_bc=False, ppw=None, kh=None, delta_t=None, num_timesteps=None, components=False, store_sol=False, solver='superLU', assembly_type="reduced_cpu", pickle_loc='tmp_test_file'):

    s = 'python hps/argparse_driver.py --n %d --pde %s --bc %s --pickle %s' % (disc_n,pde,bc,pickle_loc)

    s += ' --p %d' % (p)    
    s += ' --domain %s' % (domain)

    if (pde == "bfield_constant") or (pde == "bfield_variable") or (pde == "bfield_gravity"):
        if kh is not None:
            s += ' --kh %d' % (kh)
        else:
            s += ' --ppw %d' % (ppw)

    s += ' --solver %s' % (solver)
    s += ' --sparse_assembly %s' % (assembly_type)

    s += ' --box_xlim %f' % box_xlim
    s += ' --box_ylim %f' % box_ylim

    #s += ' --disable_cuda'
    if (periodic_bc):
        s += ' --periodic_bc'

    # Specify 3D:
    s += ' --d 3'

    if delta_t is not None:
        s += ' --delta_t %f' % (delta_t)
    
    if num_timesteps is not None:
        s += ' --num_timesteps %d' % (num_timesteps)

    if store_sol:
        s += ' --store_sol'

    # Specify whether to test components like interpolation:
    if components:
        s += ' --test_components True'

    r = os.system(s)
    """
    if (r == 0):
        f = open(pickle_loc,"rb")

        _ = pickle.load(f)
        d = pickle.load(f)

        assert d['trueres_solve_superLU'] < 2e-5
        os.system('rm %s' % pickle_loc)
    else:
        raise ValueError("test failed")
    """

def test_helm_poisson(components=False):
    run_test_via_argparse('square', 'poisson', 'log_dist', 100, 12, components=components)

def test_helm_poisson_annulus():
    run_test_via_argparse('annulus')

def test_helm_poisson_curvyannulus():
    run_test_via_argparse('curvy_annulus',box_xlim=6.0, periodic_bc=True)

def test_interpolation():
    test_helm_poisson(components=True)
