from argparse import Namespace

from hpsmultidomain.argparse_driver import run_from_args


def make_args(**overrides):
    defaults = dict(
        p=6,
        n=24,
        d=3,
        n0=None,
        n1=None,
        n2=None,
        p0=None,
        p1=None,
        p2=None,
        pde='poisson',
        domain='square',
        box_xlim=1.0,
        box_ylim=1.0,
        box_zlim=1.0,
        bc='log_dist',
        ppw=None,
        nwaves=None,
        kh=None,
        delta_t=None,
        num_timesteps=None,
        solver='superLU',
        sparse_assembly='reduced_cpu',
        pickle=None,
        store_sol=False,
        disable_cuda=True,
        periodic_bc=False,
        test_components=False,
        visualize=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def run_test_case(**overrides):
    return run_from_args(make_args(**overrides))

def test_helm_poisson(components=False):
    results = run_test_case(domain='square', pde='poisson', bc='log_dist', test_components=components)
    assert results["info"]['trueres_solve_superLU'] < 5e-5

def test_interpolation():
    test_helm_poisson(components=True)
