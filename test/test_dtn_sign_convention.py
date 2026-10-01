"""
Settles the DtN sign / normal / scaling convention that Neumann boundary
conditions will rely on (Domain_Driver._assemble_neumann_blocks), using
manufactured solutions:

  1. A leaf DtN row returns the OUTWARD normal derivative du/dn of that leaf
     (-du/dx on the left face, +du/dx right, -du/dy down, +du/dy up), in
     physical units -- not the coordinate derivative.
  2. With a body load f (A u = f), the leaf flux is  du/dn = DtN g - h,  where
     h is what get_DtNs(mode='reduce_body') returns.
  3. It is the plain derivative du/dn, not the conormal c11 du/dn, even for
     variable c11 = c22.
  4. On the assembled objects (block-diagonal A, box indexing, I_unique), the
     planned Neumann row  A[n] g - h[n]  equals the domain's outward du/dn at
     the Neumann points of a Domain_Driver built with bc_types, and interior
     rows satisfy flux continuity  (A[c1] + A[c2]) g - (h[c1] + h[c2]) = 0.

Every check runs on both leaf face discretizations: Chebyshev faces
(interpolate=False) and Gauss faces (interpolate=True, triggered by giving a
c12 coefficient). The domain is a non-square, offset box so that a wrong
scaling or orientation cannot hide.
"""
import io
from contextlib import redirect_stdout

import numpy as np
import pytest
import torch

from hpsmultidomain.domain_driver import Domain_Driver
from hpsmultidomain.geom import BoxGeometry
from hpsmultidomain.pdo import PDO_2d, const

torch.set_default_dtype(torch.double)

CPU = torch.device("cpu")
BOX = [[0.0, -0.5], [2.0, 0.5]]      # 4 x 2 leaves at a = 0.25
A_LEAF = 0.25
P = 12
TOL = 1e-8                           # spectral accuracy at p = 12 is ~1e-10


# ---- manufactured solutions: u, grad u, Laplacian u (torch, xx is (N, 2)) ---

def _cubic(xx):            # harmonic, polynomial (collocation-exact)
    x, y = xx[:, 0], xx[:, 1]
    return x**3 - 3 * x * y**2, torch.stack((3 * x**2 - 3 * y**2, -6 * x * y), 1), 0 * x


def _exp_sin(xx):          # harmonic, non-polynomial
    x, y = xx[:, 0], xx[:, 1]
    e = torch.exp(x)
    return e * torch.sin(y), torch.stack((e * torch.sin(y), e * torch.cos(y)), 1), 0 * x


def _general(xx):          # not harmonic: needs a body load
    x, y = xx[:, 0], xx[:, 1]
    s, c = torch.sin(2 * x), torch.cos(3 * y)
    u = s * c + x**3 * y
    grad = torch.stack((2 * torch.cos(2 * x) * c + 3 * x**2 * y, -3 * s * torch.sin(3 * y) + x**3), 1)
    lap = -13 * s * c + 6 * x * y
    return u, grad, lap


COEFFS = {                 # c11 = c22 = D(x, y)
    "one": lambda xx: torch.ones(xx.shape[0]),
    "two": lambda xx: 2.0 * torch.ones(xx.shape[0]),
    "variable": lambda xx: 2.0 + xx[:, 0] + xx[:, 1],
}


def _driver(D, interpolate, bc_types=None):
    pdo = PDO_2d(c11=D, c22=D, c12=const(0.0) if interpolate else None)
    with redirect_stdout(io.StringIO()):
        dd = Domain_Driver(BoxGeometry(torch.tensor(BOX)), pdo, 0, A_LEAF, p=P, d=2,
                           bc_types=bc_types)
    assert dd.hps.interpolate == interpolate
    return dd


def _leaf_faces(hps):
    """Leaf face coordinates in DtN row order, (nboxes, size_ext, 2), and the
    outward unit normal of each row's leaf face, from the leaf's own bounds."""
    nb = int(hps.nboxes)
    xx = hps.xx_ext.reshape(nb, -1, 2)
    lo = hps.grid_xx.min(dim=1).values[:, None, :]
    hi = hps.grid_xx.max(dim=1).values[:, None, :]
    tol = 1e-10
    on = [(xx[..., 0] - lo[..., 0]).abs() < tol, (xx[..., 0] - hi[..., 0]).abs() < tol,
          (xx[..., 1] - lo[..., 1]).abs() < tol, (xx[..., 1] - hi[..., 1]).abs() < tol]
    assert torch.all(sum(m.int() for m in on) == 1), "every face point lies on exactly one leaf face"
    n = torch.zeros_like(xx)
    n[..., 0] = on[1].double() - on[0].double()
    n[..., 1] = on[3].double() - on[2].double()
    return xx, n


def _rel(a, b):
    return (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()


def _dtn_and_body(dd, D, sol):
    """Leaf DtN applied to the exact face data, the reduce_body term h for
    f = A u = -D Lap u, and the exact plain normal derivative, all per leaf
    face point."""
    hps = dd.hps
    xx, n = _leaf_faces(hps)
    flat = xx.reshape(-1, 2)
    u, grad, _ = sol(flat)
    nb = xx.shape[0]
    with redirect_stdout(io.StringIO()):
        DtN = hps.get_DtNs(CPU, mode="build")
        h = hps.get_DtNs(CPU, mode="reduce_body",
                         ff_body_func=lambda q: (-D(q) * sol(q)[2]).unsqueeze(-1))
    flux = (DtN @ u.reshape(nb, -1, 1))[..., 0]
    h = h[..., 0].real
    dudn = (grad.reshape(nb, -1, 2) * n).sum(-1)
    dcoord = (grad.reshape(nb, -1, 2) * n.abs()).sum(-1)       # d/dx on x-faces, d/dy on y-faces
    return flux, h, dudn, dcoord, flat


@pytest.mark.parametrize("interpolate", [False, True])
@pytest.mark.parametrize("sol", [_cubic, _exp_sin], ids=["cubic", "exp_sin"])
def test_dtn_returns_outward_normal_derivative(interpolate, sol):
    """Source-free leaves: DtN g is the outward du/dn, not d/dx or d/dy."""
    D = COEFFS["one"]
    dd = _driver(D, interpolate)
    flux, h, dudn, dcoord, _ = _dtn_and_body(dd, D, sol)
    assert torch.linalg.norm(h) < 1e-12 * torch.linalg.norm(flux), "harmonic u: no body load"
    assert _rel(flux, dudn) < TOL
    assert _rel(flux, dcoord) > 0.5, "the coordinate-derivative convention must be ruled out"


@pytest.mark.parametrize("interpolate", [False, True])
@pytest.mark.parametrize("coeff", ["one", "two", "variable"])
def test_body_load_enters_as_minus_h_and_flux_is_plain_derivative(interpolate, coeff):
    """du/dn = DtN g - h with h = reduce_body; plain du/dn, not c11 du/dn."""
    D = COEFFS[coeff]
    dd = _driver(D, interpolate)
    flux, h, dudn, _, flat = _dtn_and_body(dd, D, _general)
    assert _rel(flux - h, dudn) < TOL
    assert _rel(flux + h, dudn) > 1e-2, "the opposite body-load sign must be ruled out"
    if coeff != "one":
        conormal = D(flat).reshape(dudn.shape) * dudn
        assert _rel(flux - h, conormal) > 1e-2, "the conormal convention must be ruled out"


@pytest.mark.parametrize("interpolate", [False, True])
def test_planned_neumann_rows_on_assembled_objects(interpolate):
    """The row formulas of _assemble_neumann_blocks, on the real A / indices."""
    D = COEFFS["variable"]
    dd = _driver(D, interpolate, bc_types={"y": "neumann"})
    hps = dd.hps
    assert dd.has_neumann and len(dd.I_Ntot) > 0

    with redirect_stdout(io.StringIO()):
        A, _ = hps.sparse_mat(CPU)                               # block-diagonal leaf DtNs
        h = hps.get_DtNs(CPU, mode="reduce_body",
                         ff_body_func=lambda q: (-D(q) * _general(q)[2]).unsqueeze(-1))
    h = h.reshape(-1).real.numpy()
    g = _general(hps.xx_ext)[0].numpy()
    Ag = A @ g

    # Neumann rows: the single leaf's flux = the DOMAIN's outward du/dn
    n_box = dd.I_Ntot_in_unique.numpy()
    pts = dd.XX_active[dd.I_Ntot]
    assert torch.allclose(hps.xx_ext[n_box], pts), "box and active indexing agree"
    y_lo, y_hi = dd.box_geom[1, 0], dd.box_geom[1, 1]
    n_y = torch.where(pts[:, 1] < 0.5 * (y_lo + y_hi), -1.0, 1.0)
    assert torch.all(((pts[:, 1] - y_lo).abs() < 1e-10) | ((pts[:, 1] - y_hi).abs() < 1e-10))
    dudn_domain = (_general(pts)[1][:, 1] * n_y).numpy()
    row_n = Ag[n_box] - h[n_box]
    assert np.linalg.norm(row_n - dudn_domain) < TOL * np.linalg.norm(dudn_domain)

    # interior rows: flux continuity, as the existing assembly A[c1] + A[c2]
    c1, c2 = hps.I_copy1.numpy(), hps.I_copy2.numpy()
    resid = (Ag[c1] + Ag[c2]) - (h[c1] + h[c2])
    assert np.linalg.norm(resid) < TOL * np.linalg.norm(Ag[c1])
