"""
Body loads given as a callable or as a vector on XXfull. The reduced body load
(reduce_body) takes the dtype of what is computed -- the operator's and the
load's, so real for a real problem -- instead of defaulting to complex. A
callable load then works for real problems in 2D and 3D, with Dirichlet and
Neumann faces, and matches the same load given as a vector; a complex load with
real Dirichlet data gives a complex solution.
"""
import io
from contextlib import redirect_stdout

import numpy as np
import pytest
import torch

import hpsmultidomain.hps_parallel_leaf_ops as leaf_ops
from hpsmultidomain.argparse_driver import run_from_args
from hpsmultidomain.domain_driver import Domain_Driver
from hpsmultidomain.geom import BoxGeometry
from hpsmultidomain.pdo import PDO_2d, PDO_3d, const
from test_3d import make_args as make_args_3d

torch.set_default_dtype(torch.double)

CPU = torch.device("cpu")
BOX_2D = [[0.0, -0.5], [2.0, 0.5]]      # 4 x 2 leaves at a = 0.25; y faces at -/+0.5


def _quiet(fn, *args, **kw):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def _driver_2d(interpolate=False, build=True, **kw):
    """-Lap u + u on BOX_2D; interpolate=True gives Gauss faces."""
    pdo = PDO_2d(c11=const(1.0), c22=const(1.0), c=const(1.0), c12=const(0.0) if interpolate else None)
    dd = _quiet(Domain_Driver, BoxGeometry(torch.tensor(BOX_2D)), pdo, 0, 0.25, p=8, d=2, **kw)
    if build:
        _quiet(dd.build, "reduced_cpu", "superLU", verbose=False)
        _quiet(dd.build_factorize, "superLU", False)
    return dd


def u_2d(xx):           # periodic over the box width 2; -Lap u + u = pi^2 u
    return (torch.sin(torch.pi * xx[:, 0]) * torch.exp(xx[:, 1])).unsqueeze(-1)


def f_2d(xx):
    return torch.pi**2 * u_2d(xx)


def _close(a, b, rtol=1e-13):
    return a.dtype == b.dtype and torch.linalg.norm(a - b) <= rtol * torch.linalg.norm(b)


def _rel_err(dd, sol, exact):
    """Relative error on the leaf collocation nodes (JJ.Jtot: leaf corners excluded)."""
    nb, P = int(dd.hps.nboxes), int(np.prod(dd.hps.p))
    J = torch.as_tensor(dd.hps.H.JJ.Jtot)
    num, ref = sol.reshape(nb, P, -1)[:, J], exact(dd.XXfull).reshape(nb, P, -1)[:, J]
    return (torch.linalg.norm(num - ref) / torch.linalg.norm(ref)).item()


# ---- the reduced body load ----------------------------------------------------------

@pytest.mark.parametrize("interpolate", [False, True])
def test_reduced_body_load_takes_the_dtype_of_the_load(interpolate):
    dd = _driver_2d(interpolate, build=False)
    f_cplx = lambda xx: (1 + 2j) * f_2d(xx)
    reduce = lambda func=None, vec=None: _quiet(dd.hps.reduce_body, CPU, func, vec)
    assert reduce(func=f_2d).dtype == torch.float64
    assert reduce(func=f_cplx).dtype == torch.complex128
    assert reduce(vec=f_2d(dd.XXfull)).dtype == torch.float64
    assert reduce(vec=f_cplx(dd.XXfull)).dtype == torch.complex128


@pytest.mark.parametrize("interpolate", [False, True])
def test_callable_and_vector_loads_reduce_alike(interpolate):
    dd = _driver_2d(interpolate, build=False)
    by_func = _quiet(dd.hps.reduce_body, CPU, f_2d, None)
    by_vec = _quiet(dd.hps.reduce_body, CPU, None, f_2d(dd.XXfull))
    assert _close(by_func, by_vec)


def test_store_chunk_allocates_from_the_first_chunk_and_widens():
    real, cplx = torch.ones(2, 3, 1), 1j * torch.ones(2, 3, 1, dtype=torch.complex128)
    buf = leaf_ops.store_chunk(None, 4, 0, real, CPU)
    assert buf.shape == (4, 3, 1) and buf.dtype == torch.float64
    buf = leaf_ops.store_chunk(buf, 4, 2, cplx, CPU)       # a wider chunk widens the buffer
    assert buf.dtype == torch.complex128
    assert torch.equal(buf[:2], real.to(torch.complex128)) and torch.equal(buf[2:], cplx)
    buf = leaf_ops.store_chunk(leaf_ops.store_chunk(None, 4, 0, cplx, CPU), 4, 2, real, CPU)
    assert buf.dtype == torch.complex128 and torch.equal(buf[2:], real.to(torch.complex128))


# ---- solves -------------------------------------------------------------------------

@pytest.mark.parametrize("interpolate, bc_types", [
    (False, {"x": "dirichlet", "y": "dirichlet"}),
    (True, {"x": "dirichlet", "y": "dirichlet"}),
    (False, {"x": "periodic", "y": "neumann"}),
], ids=["cheb-dirichlet", "gauss-dirichlet", "cheb-periodic-neumann"])
def test_real_solve_with_a_callable_load(interpolate, bc_types):
    """A callable load on a real problem: a real solution, the same as with the
    load as a vector, and accurate."""
    dd = _driver_2d(interpolate, bc_types=bc_types)
    # outward du/dn on the Neumann faces (y faces here): n_y du/dy = n_y u
    g = dd.normals_Ntot[:, 1:2] * u_2d(dd.XX_active[dd.I_Ntot])
    by_func = _quiet(dd.solve_dir_full, u_2d, f_2d, uu_neu=g)
    by_vec = _quiet(dd.solve_dir_full, u_2d, f_2d(dd.XXfull), uu_neu=g)
    assert by_func.dtype == torch.float64 and _close(by_func, by_vec)
    assert _rel_err(dd, by_func, u_2d) < 1e-5


def test_real_solve_with_a_callable_load_in_3d():
    pdo = PDO_3d(c11=const(1.0), c22=const(1.0), c33=const(1.0), c=const(1.0))
    dd = _quiet(Domain_Driver, BoxGeometry(torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])), pdo, 0, 0.25, p=6, d=3)
    _quiet(dd.build, "reduced_cpu", "superLU", verbose=False)
    _quiet(dd.build_factorize, "superLU", False)
    u = lambda xx: (torch.sin(xx[:, 0]) * torch.exp(xx[:, 1]) * torch.cos(xx[:, 2])).unsqueeze(-1)   # -Lap u = u
    f = lambda xx: 2 * u(xx)
    by_func = _quiet(dd.solve_dir_full, u, f)
    by_vec = _quiet(dd.solve_dir_full, u, f(dd.XXfull))
    assert by_func.dtype == torch.float64 and _close(by_func, by_vec)
    assert _rel_err(dd, by_func, u) < 1e-6


def test_complex_load_with_real_dirichlet_data():
    """u = u_2d + i v with v = 0 on the boundary of BOX_2D: the Dirichlet data is
    real, the load complex, and so is the solution."""
    dd = _driver_2d(bc_types={"x": "dirichlet", "y": "dirichlet"})
    v = lambda xx: (torch.sin(torch.pi * xx[:, 0] / 2) * torch.cos(torch.pi * xx[:, 1])).unsqueeze(-1)
    f = lambda xx: f_2d(xx) + 1j * ((torch.pi**2 / 4 + torch.pi**2) * v(xx) + v(xx))   # -Lap v + v
    sol = _quiet(dd.solve_dir_full, u_2d, f)
    assert sol.dtype == torch.complex128
    assert _rel_err(dd, sol, lambda xx: u_2d(xx) + 1j * v(xx)) < 1e-5


def test_driver_3d_mode_with_a_callable_load():
    """The argparse driver's zeros / bfield_gravity mode passes a real callable load
    (-1); it raised in get_rhs while the reduced load defaulted to complex."""
    results = _quiet(run_from_args, make_args_3d(bc="zeros", pde="bfield_gravity", n=8, p=6, kh=5.0))
    sol = results["uu_sol"]
    assert sol.dtype == torch.float64 and torch.isfinite(sol).all() and sol.abs().max() > 0
