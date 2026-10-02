"""
Shared helpers for the boundary-condition and Neumann tests in this directory:
driver builders on a non-square, offset box, a manufactured solution with its
body load and Neumann data, a hand-built right-hand side, and error measures.
"""
import io
import itertools
from contextlib import redirect_stdout

import pytest
import torch

from hpsmultidomain.domain_driver import Domain_Driver
from hpsmultidomain.geom import BoxGeometry, ParametrizedGeometry2D
from hpsmultidomain.pdo import PDO_2d, const

torch.set_default_dtype(torch.double)

CPU = torch.device("cpu")
BOX = [[0.0, -0.5], [2.0, 0.5]]      # 4 x 2 leaves at a = 0.25; y faces at -/+0.5
ALL_BCS = list(itertools.product(["dirichlet", "neumann", "periodic"], ["dirichlet", "neumann"]))
NEUMANN_BCS = [(bx, by) for bx, by in ALL_BCS if "neumann" in (bx, by)]

# Every Neumann configuration on Chebyshev faces; on Gauss faces only those with a
# Dirichlet face (one is enough): see the Gauss refusal test in test_bc_types.py.
SOLVABLE = ([pytest.param(False, {"x": bx, "y": by}, id="cheb-%s-%s" % (bx, by)) for bx, by in NEUMANN_BCS]
            + [pytest.param(True, {"x": bx, "y": by}, id="gauss-%s-%s" % (bx, by))
               for bx, by in NEUMANN_BCS if "dirichlet" in (bx, by)]
            + [pytest.param(i, {"x_lo": "dirichlet", "x_hi": "neumann", "y": "neumann"},
                            id="%s-only-x_lo-dirichlet" % name) for i, name in ((False, "cheb"), (True, "gauss"))])


def driver(interpolate=False, c=1.0, D=None, p=8, pdo=None, **kw):
    """-D Lap u + c u on BOX (D = 1 unless given; c=None: no zeroth-order term), or the
    given pdo; interpolate=True gives Gauss faces (through a zero c12)."""
    if pdo is None:
        D = const(1.0) if D is None else D
        pdo = PDO_2d(c11=D, c22=D, c=None if c is None else const(c),
                     c12=const(0.0) if interpolate else None)
    with redirect_stdout(io.StringIO()):
        return Domain_Driver(BoxGeometry(torch.tensor(BOX)), pdo, 0, 0.25, p=p, d=2, **kw)


def built(interpolate=False, c=1.0, D=None, p=8, pdo=None, **kw):
    dd = driver(interpolate, c, D, p, pdo, **kw)
    with redirect_stdout(io.StringIO()):
        dd.build("reduced_cpu", "superLU", verbose=False)
    return dd


def factorized(*args, **kw):
    dd = built(*args, **kw)
    with redirect_stdout(io.StringIO()):
        dd.build_factorize("superLU", False)
    return dd


def on_faces(dd):
    xx, g, tol = dd.XX_active, dd.box_geom, 0.01 * dd.hps.hmin
    return {"x_lo": xx[:, 0] < g[0, 0] + tol, "x_hi": xx[:, 0] > g[0, 1] - tol,
            "y_lo": xx[:, 1] < g[1, 0] + tol, "y_hi": xx[:, 1] > g[1, 1] - tol}


# u = sin(pi x) e^y, periodic over the box width 2, and its derivatives
def u_exact(xx):
    return (torch.sin(torch.pi * xx[:, 0]) * torch.exp(xx[:, 1])).unsqueeze(-1)


def grad_exact(xx):
    e = torch.exp(xx[:, 1])
    return torch.stack((torch.pi * torch.cos(torch.pi * xx[:, 0]) * e, torch.sin(torch.pi * xx[:, 0]) * e), 1)


def manufactured(dd, D=None, c=1.0):
    """u_exact as an (N, 1) callable, the body load f = -D Lap u + c u on the grid
    (XXfull), and the outward du/dn at the Neumann points (I_Ntot order).
    Lap u = (1 - pi^2) u; D = 1 unless given."""
    Dv = torch.ones(dd.XXfull.shape[0]) if D is None else D(dd.XXfull)
    f = ((torch.pi**2 - 1) * Dv + c).unsqueeze(-1) * u_exact(dd.XXfull)
    xx = dd.XX_active[dd.I_Ntot]
    return u_exact, f, (grad_exact(xx) * dd.normals_Ntot).sum(1, keepdim=True)


def hand_rhs(dd, u, f, g):
    """The right-hand side built by hand from the assembled blocks, in the order Ji:
    interior rows h[c1] + h[c2], Neumann rows g_N + h[n], minus the Dirichlet lift."""
    b = dd.hps.reduce_body(CPU, None, f)[dd.Ji, 0].numpy()
    b[len(dd.I_Ctot):] += g[:, 0].numpy()
    return b - dd.A_CX @ u(dd.XX_active[dd.I_Xtot])[:, 0].numpy()


def rel_err_off_corners(dd, sol, u):
    """Relative error of a solve_dir_full result on the grid, leaving out the leaf
    corners: with Chebyshev faces they are not unknowns, and the reconstruction only
    extrapolates them (~1e-3 at p = 8, pre-existing)."""
    true = u(dd.XXfull)
    g = dd.hps.grid_xx
    lo, hi = g.min(1).values[:, None, :], g.max(1).values[:, None, :]
    edge = ((g - lo).abs() < 1e-12) | ((g - hi).abs() < 1e-12)
    keep = ~(edge[..., 0] & edge[..., 1]).reshape(-1)
    return (torch.linalg.norm((sol - true)[keep]) / torch.linalg.norm(true[keep])).item()


def wall_nodes(dd):
    """Leaf and grid node of each Neumann point (hps.I_single order = I_Ntot order).
    Chebyshev faces only: there the leaf face slots are grid nodes."""
    size_ext = len(dd.hps.H.JJ.Jx)
    Jx = torch.as_tensor(dd.hps.H.JJ.Jx)
    return dd.hps.I_single // size_ext, Jx[dd.hps.I_single % size_ext]


def leaf_gradient(dd, sol):
    """du/dx and du/dy at every grid node of a reconstructed solution, leaf by leaf
    with the leaf differentiation matrices: (nboxes, p^2, 2)."""
    U = sol[:, 0].reshape(int(dd.hps.nboxes), -1)
    Ds = dd.hps.H.Ds
    return torch.stack((U @ Ds[3].T, U @ Ds[4].T), -1)


def curved_geometry():
    """The map of test_hps_multidomain_curved: x = xi1, y = xi2 / psi(xi1) on the unit
    reference square, with its transformed Laplace PDO."""
    mag = 0.3
    psi = lambda x: 1 - mag * torch.sin(4 * x)
    dpsi = lambda x: -mag * 4 * torch.cos(4 * x)
    ddpsi = lambda x: mag * 16 * torch.sin(4 * x)
    geom = ParametrizedGeometry2D(
        torch.tensor([[0, 0], [1.0, 1.0]]),
        lambda xx: xx[..., 0], lambda xx: xx[..., 1] / psi(xx[..., 0]),
        lambda xx: xx[..., 0], lambda xx: xx[..., 1] * psi(xx[..., 0]),
        y1_d1=lambda xx: torch.ones_like(xx[..., 0]), y2_d1=lambda xx: xx[..., 1] * dpsi(xx[..., 0]),
        y2_d2=lambda xx: psi(xx[..., 0]), y2_d1d1=lambda xx: xx[..., 1] * ddpsi(xx[..., 0]))
    return geom, geom.transform_helmholtz_pdo(lambda yy, kh: 0 * yy[..., 0], 0)
