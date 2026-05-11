from argparse import Namespace

import pytest

from hpsmultidomain.argparse_driver import run_from_args


def make_args(**overrides):
    defaults = dict(
        p=12,
        n=100,
        d=2,
        n0=None,
        n1=None,
        n2=None,
        p0=None,
        p1=None,
        p2=None,
        pde='bfield_constant',
        domain='square',
        box_xlim=1.0,
        box_ylim=1.0,
        box_zlim=1.0,
        bc='free_space',
        ppw=40,
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


def run_test_case(domain, box_xlim=1.0, box_ylim=1.0, periodic_bc=False):
    results = run_from_args(
        make_args(domain=domain, box_xlim=box_xlim, box_ylim=box_ylim, periodic_bc=periodic_bc)
    )
    assert results["info"]['trueres_solve_superLU'] < 2e-5

def test_helm_poisson():
    run_test_case('square')

@pytest.mark.xfail(reason="Annulus free-space regression is not currently accurate on main.", strict=False)
def test_helm_poisson_annulus():
    run_test_case('annulus')

@pytest.mark.xfail(reason="Curvy annulus free-space regression is not currently accurate on main.", strict=False)
def test_helm_poisson_curvyannulus():
    run_test_case('curvy_annulus', box_xlim=6.0, periodic_bc=True)
