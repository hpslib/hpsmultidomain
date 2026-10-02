"""
End-to-end 2D solves with Neumann faces: get_rhs, the factorized A_CC, the Ji
scatter and the leaf reconstruction (expand_boundary_data,
fill_missing_boundary_values, the leaf solves). Checks that the reconstructed
solution carries the solved wall values and satisfies the Neumann condition,
converges spectrally in p, and works for general operators and several
right-hand sides; and that mapped geometries refuse Neumann faces.
"""
import io
from contextlib import redirect_stdout

import numpy as np
import pytest
import torch

from bc_helpers import (SOLVABLE, body_load, curved_geometry, factorized, grad_exact, leaf_gradient,
                        manufactured, rel_err_off_corners, u_exact, wall_nodes)
from hpsmultidomain.domain_driver import Domain_Driver
from hpsmultidomain.pdo import PDO_2d, const

CHEB_CONFIGS = [{"x": "periodic", "y": "neumann"}, {"x": "dirichlet", "y": "neumann"},
                {"x": "neumann", "y": "dirichlet"}, {"x": "neumann", "y": "neumann"},
                {"x_lo": "dirichlet", "x_hi": "neumann", "y_lo": "neumann", "y_hi": "dirichlet"}]


def _quiet(fn, *args, **kw):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def _solve_error(dd, u, f, g):
    """true_err of solve(known_sol=True): the leaf collocation nodes, corners excluded."""
    return _quiet(dd.solve, u, ff_body_vec=f, uu_neu_vec=g, known_sol=True)[2]


# ---- manufactured solutions ---------------------------------------------------------

@pytest.mark.parametrize("load", ["vector", "callable"])
@pytest.mark.parametrize("interpolate, bc_types", SOLVABLE)
def test_neumann_solve_recovers_manufactured_solution(interpolate, bc_types, load):
    """The body load as a grid vector or a callable. known_sol=True also compares
    the skeleton values with u(XX[Ji])."""
    dd = factorized(interpolate, bc_types=bc_types)
    u, f, g = manufactured(dd)
    body = dict(ff_body_vec=f) if load == "vector" else dict(ff_body_func=body_load())
    out = _quiet(dd.solve, u, uu_neu_vec=g, known_sol=True, **body)
    true_err, reverse_bdry_error = out[2], out[7]
    assert out[0].dtype == torch.float64
    assert true_err < 1e-5 and reverse_bdry_error < 1e-5


def test_neumann_data_is_the_plain_normal_derivative():
    """Variable c11 = c22 = D: the Neumann data is the plain du/dn (passed here as
    a callable through solve_dir_full); the conormal D du/dn gives a wrong solution."""
    D = lambda xx: 2.0 + xx[:, 0] + xx[:, 1]
    dd = factorized(D=D, bc_types={"y": "neumann"})
    u, f, _ = manufactured(dd, D=D)
    dudn = lambda xx: torch.sign(xx[:, 1:2]) * u(xx)          # y faces at -/+0.5: -/+ du/dy = -/+ u
    plain = _quiet(dd.solve_dir_full, u, f, uu_neu=dudn)
    conormal = _quiet(dd.solve_dir_full, u, f, uu_neu=lambda xx: D(xx)[:, None] * dudn(xx))
    assert rel_err_off_corners(dd, plain, u) < 1e-5
    assert rel_err_off_corners(dd, conormal, u) > 0.1


@pytest.mark.parametrize("interpolate", [False, True])
def test_laplace_with_neumann_faces_solves_given_a_dirichlet_face(interpolate):
    """The singularity check needs both conditions: one Dirichlet face makes c = 0 fine."""
    dd = factorized(interpolate, c=None, bc_types={"x": "dirichlet", "y": "neumann"})
    assert _solve_error(dd, *manufactured(dd, c=0.0)) < 1e-5


# ---- reconstruction ------------------------------------------------------------------

@pytest.mark.parametrize("bc_types", CHEB_CONFIGS, ids=lambda b: "-".join("%s=%s" % kv for kv in b.items()))
def test_reconstruction_carries_the_wall_values_and_the_neumann_condition(bc_types):
    """Chebyshev faces, arbitrary (random) data: the reconstructed solution holds the
    solved wall values at the leaf wall nodes, and its own du/dn there is the imposed
    data -- both to roundoff, whatever the data, since they hold algebraically."""
    dd = factorized(bc_types=bc_types)
    gen = torch.Generator().manual_seed(0)
    g = torch.randn(len(dd.I_Ntot), 1, generator=gen)
    f = torch.randn(dd.XXfull.shape[0], 1, generator=gen)
    uD = torch.randn(len(dd.I_Xtot), 1, generator=gen)
    out = _quiet(dd.solve_dir_full, uD, f, uu_neu=g)
    skel = _quiet(dd.solve_helper_blackbox, lambda xx: uD, uu_dir_vec=uD, ff_body_vec=f, uu_neu_vec=g)[0]
    box, node = wall_nodes(dd)
    assert torch.equal(out[:, 0].reshape(int(dd.hps.nboxes), -1)[box, node], skel[len(dd.I_Ctot):, 0])
    dudn = (leaf_gradient(dd, out)[box, node] * dd.normals_Ntot).sum(1)
    assert torch.linalg.norm(dudn - g[:, 0]) <= 1e-12 * torch.linalg.norm(g)


@pytest.mark.parametrize("bc_types", [{"x": "dirichlet", "y": "neumann"}, {"x": "neumann", "y": "dirichlet"}],
                         ids=["neumann-y", "neumann-x"])
def test_reconstruction_on_gauss_faces_has_the_wall_derivative(bc_types):
    """Gauss faces: the data lives on Gauss nodes, so check the reconstructed
    solution's du/dn at the Chebyshev wall nodes against the exact one (spectral)."""
    dd = factorized(True, p=10, bc_types=bc_types)
    out = _quiet(dd.solve_dir_full, *manufactured(dd)[:2], uu_neu=manufactured(dd)[2])
    grad = leaf_gradient(dd, out)
    gx, B = dd.hps.grid_xx, dd.box_geom
    lo, hi = gx.min(1).values[:, None, :], gx.max(1).values[:, None, :]
    edge = ((gx - lo).abs() < 1e-12) | ((gx - hi).abs() < 1e-12)
    corner = edge[..., 0] & edge[..., 1]
    errs, refs = [], []
    for face, (ax, side, sgn) in {"x_lo": (0, 0, -1.), "x_hi": (0, 1, 1.), "y_lo": (1, 0, -1.), "y_hi": (1, 1, 1.)}.items():
        if dd.bc_types[face] != "neumann":
            continue
        on = ((gx[..., ax] - B[ax, side]).abs() < 1e-10) & ~corner
        exact = sgn * grad_exact(gx[on])[:, ax]
        errs.append(sgn * grad[on][:, ax] - exact)
        refs.append(exact)
    err, ref = torch.cat(errs), torch.cat(refs)
    assert torch.linalg.norm(err) < 1e-6 * torch.linalg.norm(ref)


# ---- convergence and operators ----------------------------------------------------------

@pytest.mark.parametrize("interpolate, bc_types", [(False, {"x": "periodic", "y": "neumann"}),
                                                   (True, {"x": "dirichlet", "y": "neumann"})],
                         ids=["cheb-periodic-neumann", "gauss-dirichlet-neumann"])
def test_neumann_solutions_converge_spectrally_in_p(interpolate, bc_types):
    """A wrong sign or a dropped term would show up as stalled convergence."""
    errs = []
    for p in (6, 8, 10, 12):
        dd = factorized(interpolate, p=p, bc_types=bc_types)
        errs.append(_solve_error(dd, *manufactured(dd)))
    assert all(e1 < e0 / 100 for e0, e1 in zip(errs, errs[1:])), errs
    assert errs[-1] < 1e-10, errs


def _uxx(xx): return -torch.pi**2 * u_exact(xx)[:, 0]
def _ux(xx): return grad_exact(xx)[:, 0]
def _uy(xx): return u_exact(xx)[:, 0]          # = u_yy
def _uxy(xx): return grad_exact(xx)[:, 0]      # = d/dy u_x


_C11 = lambda xx: 1.0 + 0.25 * xx[:, 0]
OPERATORS = {
    # (interpolate, pdo, -c11 u_xx - c22 u_yy - c12 u_xy + c1 u_x + c2 u_y + c u, bc_types)
    "cheb-variable-anisotropic-convection": (
        False, PDO_2d(c11=_C11, c22=const(2.0), c1=const(0.5), c2=const(-0.3), c=const(1.0)),
        lambda xx: -_C11(xx) * _uxx(xx) - 2.0 * _uy(xx) + 0.5 * _ux(xx) - 0.3 * _uy(xx) + _uy(xx),
        {"x": "dirichlet", "y": "neumann"}),
    "cheb-anisotropic-convection-periodic": (
        False, PDO_2d(c11=const(1.0), c22=const(2.0), c1=const(0.5), c2=const(-0.3), c=const(1.0)),
        lambda xx: -_uxx(xx) - 2.0 * _uy(xx) + 0.5 * _ux(xx) - 0.3 * _uy(xx) + _uy(xx),
        {"x": "periodic", "y": "neumann"}),
    "gauss-mixed-derivative-neumann-y": (
        True, PDO_2d(c11=const(1.0), c22=const(1.0), c12=const(0.4), c=const(1.0)),
        lambda xx: -_uxx(xx) - _uy(xx) - 0.4 * _uxy(xx) + _uy(xx),
        {"x": "dirichlet", "y": "neumann"}),
    "gauss-mixed-derivative-neumann-x": (
        True, PDO_2d(c11=const(1.0), c22=const(1.0), c12=const(0.4), c=const(1.0)),
        lambda xx: -_uxx(xx) - _uy(xx) - 0.4 * _uxy(xx) + _uy(xx),
        {"x": "neumann", "y": "dirichlet"}),
}


@pytest.mark.parametrize("name", OPERATORS)
def test_general_operators_with_neumann_faces(name):
    """Variable and anisotropic coefficients, convection, and a real c12 on Gauss
    faces: the Neumann data stays the plain du/dn."""
    interpolate, pdo, rhs, bc_types = OPERATORS[name]
    dd = factorized(p=10, pdo=pdo, bc_types=bc_types)
    assert dd.hps.interpolate == interpolate
    u, _, g = manufactured(dd)
    assert _solve_error(dd, u, rhs(dd.XXfull)[:, None], g) < 1e-7


def test_two_right_hand_sides_solve_column_by_column():
    """No body load; two solutions of -Lap u + u = 0, solved together."""
    dd = factorized(p=10, bc_types={"x": "dirichlet", "y": "neumann"})
    s2 = np.sqrt(2.0)
    V = [lambda xx: (torch.cosh(s2 * xx[:, 0]) * torch.cos(xx[:, 1]))[:, None],     # -Lap V = -V
         lambda xx: torch.exp(xx[:, 0])[:, None]]
    dV = [lambda xx: torch.stack((s2 * torch.sinh(s2 * xx[:, 0]) * torch.cos(xx[:, 1]),
                                  -torch.cosh(s2 * xx[:, 0]) * torch.sin(xx[:, 1])), 1),
          lambda xx: torch.stack((torch.exp(xx[:, 0]), 0 * xx[:, 1]), 1)]
    xx = dd.XX_active[dd.I_Ntot]
    G = torch.cat([(d(xx) * dd.normals_Ntot).sum(1, keepdim=True) for d in dV], 1)
    sol = _quiet(dd.solve_dir_full, lambda xx: torch.cat([v(xx) for v in V], 1), uu_neu=G)
    for k, v in enumerate(V):
        assert rel_err_off_corners(dd, sol[:, k:k + 1], v) < 1e-9


# ---- geometry ----------------------------------------------------------------------

def test_neumann_faces_refused_on_mapped_geometries():
    """A DtN row on a mapped geometry gives the reference-face derivative, not the
    physical du/dn; Dirichlet faces on the same map are unaffected."""
    geom, pdo = curved_geometry()
    with pytest.raises(NotImplementedError, match="mapped geometry"):
        _quiet(Domain_Driver, geom, pdo, 0, 1 / 8, p=8, d=2, bc_types={"y": "neumann"})
    _quiet(Domain_Driver, geom, pdo, 0, 1 / 8, p=8, d=2, bc_types={"y": "dirichlet"})
