"""
Boundary-condition type infrastructure (2D) and the discrete Neumann system:
resolve_bc_types; the Domain_Driver boundary partition into Dirichlet (I_Xtot),
Neumann (I_Ntot) and unknown skeleton (I_Ctot) points; the leaf-face slot
bookkeeping (interior pairs hps.I_copy1 / I_copy2, single copies hps.I_single on
Neumann faces, eliminated Dirichlet points); the Neumann blocks that build()
assembles; the Neumann right-hand side (get_rhs); and the refusal to factorize
singular configurations. End-to-end solves are in test_neumann_2d.py.
"""
import io
from contextlib import redirect_stdout

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from bc_helpers import (ALL_BCS, CPU, NEUMANN_BCS, SOLVABLE, built, driver, hand_rhs,
                        manufactured, on_faces, rel_err_off_corners)
from hpsmultidomain.domain_driver import Domain_Driver, apply_sparse_lowmem, resolve_bc_types
from hpsmultidomain.geom import BoxGeometry
from hpsmultidomain.pdo import PDO_3d, const


# ---- resolve_bc_types ---------------------------------------------------------

def test_resolve_defaults_axes_faces_and_legacy_periodic():
    assert resolve_bc_types(2) == {f: "dirichlet" for f in ("x_lo", "x_hi", "y_lo", "y_hi")}
    assert resolve_bc_types(2, bc_types={"y": "Neumann"}) == \
        {"x_lo": "dirichlet", "x_hi": "dirichlet", "y_lo": "neumann", "y_hi": "neumann"}
    # a face entry overrides its axis entry, whatever the dict order
    assert resolve_bc_types(2, bc_types={"y_hi": "dirichlet", "y": "neumann"})["y_hi"] == "dirichlet"
    assert resolve_bc_types(2, periodic_bc=True) == resolve_bc_types(2, bc_types={"x": "periodic"})
    assert resolve_bc_types(2, periodic_bc=True, bc_types={"y": "neumann"})["x_lo"] == "periodic"
    assert resolve_bc_types(3) is None


@pytest.mark.parametrize("periodic_bc, bc_types, err", [
    (False, {"y": "periodic"}, ValueError),            # y faces cannot be periodic
    (False, {"x_lo": "periodic"}, ValueError),         # periodic must pair up
    (False, {"z": "neumann"}, ValueError),             # unknown face
    (False, {"x": "robin"}, ValueError),               # unknown type
    (True, {"x": "neumann"}, ValueError),              # conflicts with periodic_bc
])
def test_resolve_rejects_invalid_specs(periodic_bc, bc_types, err):
    with pytest.raises(err):
        resolve_bc_types(2, periodic_bc=periodic_bc, bc_types=bc_types)


def test_bc_types_rejected_in_3d():
    pdo = PDO_3d(c11=const(1.0), c22=const(1.0), c33=const(1.0))
    box = BoxGeometry(torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    with pytest.raises(NotImplementedError):
        Domain_Driver(box, pdo, 0, 0.5, p=4, d=3, bc_types={"y": "neumann"})


# ---- partition ------------------------------------------------------------------

@pytest.mark.parametrize("interpolate", [False, True])
@pytest.mark.parametrize("bx, by", ALL_BCS)
def test_partition_follows_face_types(interpolate, bx, by):
    dd = driver(interpolate, bc_types={"x": bx, "y": by})
    X, N, C = (set(t.tolist()) for t in (dd.I_Xtot, dd.I_Ntot, dd.I_Ctot))
    assert not (X & N) and not (X & C) and not (N & C)
    assert X | N | C == set(range(dd.ntot))     # periodic x_hi points are not in XX_active

    faces = on_faces(dd)
    want = {"dirichlet": torch.zeros(dd.ntot, dtype=torch.bool),
            "neumann": torch.zeros(dd.ntot, dtype=torch.bool)}
    for face, kind in dd.bc_types.items():
        if kind in want:
            want[kind] |= faces[face]
    assert X == set(torch.where(want["dirichlet"])[0].tolist())
    assert N == set(torch.where(want["neumann"])[0].tolist())
    assert dd.has_neumann == (by == "neumann" or bx == "neumann")
    assert torch.equal(dd.I_Ntot_in_unique, dd.hps.I_unique[dd.I_Ntot])
    if bx == "periodic":
        assert dd.periodic_bc and not faces["x_hi"].any()

    # normals_Ntot: unit, axis-aligned, and a small step along it leaves the box
    nrm = dd.normals_Ntot
    assert nrm.shape == (len(dd.I_Ntot), 2) and torch.all(nrm.abs().sum(1) == 1)
    out = dd.XX_active[dd.I_Ntot] + 0.01 * nrm
    g = dd.box_geom
    inside = (out[:, 0] > g[0, 0]) & (out[:, 0] < g[0, 1]) & (out[:, 1] > g[1, 0]) & (out[:, 1] < g[1, 1])
    assert not inside.any()


@pytest.mark.parametrize("interpolate", [False, True])
@pytest.mark.parametrize("periodic", [False, True])
def test_dirichlet_and_periodic_match_the_previous_hardcoded_rule(interpolate, periodic):
    """The old hps_disc rule: every face Dirichlet, or (periodic) the y faces
    Dirichlet and the x_hi face excluded from the unknowns."""
    legacy = driver(interpolate, periodic_bc=periodic)
    new = driver(interpolate, bc_types={"x": "periodic" if periodic else "dirichlet"})
    faces = on_faces(legacy)
    if periodic:
        old_X = faces["y_lo"] | faces["y_hi"]
        old_C = ~(old_X | faces["x_hi"])
    else:
        old_X = faces["x_lo"] | faces["x_hi"] | faces["y_lo"] | faces["y_hi"]
        old_C = ~old_X
    for dd in (legacy, new):
        assert torch.equal(dd.I_Xtot, torch.where(old_X)[0])
        assert torch.equal(dd.I_Ctot, torch.where(old_C)[0])
        assert len(dd.I_Ntot) == 0 and not dd.has_neumann


# ---- leaf-face slot bookkeeping ------------------------------------------------------

@pytest.mark.parametrize("interpolate", [False, True])
@pytest.mark.parametrize("bx, by", ALL_BCS)
def test_leaf_face_slots_split_into_pairs_single_copies_and_dirichlet(interpolate, bx, by):
    """Every leaf face slot is exactly one of: copy 1 or copy 2 of an interior
    pair, a single copy on a Neumann face, or an eliminated Dirichlet point;
    and the XX_active-level partition indexes the same slots in the same order."""
    dd = driver(interpolate, bc_types={"x": bx, "y": by})
    hps = dd.hps
    c1, c2, s, x = hps.I_copy1, hps.I_copy2, hps.I_single, dd.I_Xtot_in_unique
    assert torch.equal(torch.sort(torch.cat((c1, c2, s, x))).values, torch.arange(hps.xx_ext.shape[0]))
    assert torch.equal(torch.sort(torch.cat((c1, s, x))).values, hps.I_unique)
    assert torch.equal(hps.I_unique[dd.I_Ctot], c1)
    assert torch.equal(hps.I_unique[dd.I_Ntot], s)
    assert torch.equal(dd.Ji, torch.cat((dd.I_Ctot, dd.I_Ntot)))
    assert (len(s) > 0) == dd.has_neumann
    # the two copies of a pair are one physical point (periodic: across the seam)
    gap = hps.xx_ext[c1] - hps.xx_ext[c2]
    if bx == "periodic":
        w = dd.box_geom[0, 1] - dd.box_geom[0, 0]
        gap[:, 0] = torch.remainder(gap[:, 0] + w / 2, w) - w / 2
    assert gap.abs().max() < 1e-12


@pytest.mark.parametrize("interpolate", [False, True])
@pytest.mark.parametrize("bx, by", NEUMANN_BCS)
def test_single_copy_rows_and_columns_are_the_dirichlet_ones_moved(interpolate, bx, by):
    """A Neumann point has one row, A[n], and one column, A[:, n]: exactly its
    row and column when the same face is Dirichlet. So the Neumann blocks are
    the Dirichlet blocks with those points moved from X to C."""
    neu = built(interpolate, bc_types={"x": bx, "y": by})
    dir_ = driver(interpolate, bc_types={f: "dirichlet" if k == "neumann" else k
                                          for f, k in neu.bc_types.items()})
    # the leaf DtNs do not depend on the boundary types, so reuse A: the
    # comparison is then pure bookkeeping and must be exact
    dir_.A = neu.A
    dir_.build_blackboxsolver("superLU", False)
    X = dir_.I_Xtot.numpy()
    JN, JD = np.searchsorted(X, neu.I_Ntot.numpy()), np.searchsorted(X, neu.I_Xtot.numpy())
    assert np.array_equal(X[JN], neu.I_Ntot.numpy()) and np.array_equal(X[JD], neu.I_Xtot.numpy())
    want = {"A_CC": sp.bmat([[dir_.A_CC, dir_.A_CX[:, JN]], [dir_.A_XC[JN], dir_.A_XX[JN][:, JN]]]),
            "A_CX": sp.vstack((dir_.A_CX[:, JD], dir_.A_XX[JN][:, JD])),
            "A_XC": sp.hstack((dir_.A_XC[JD], dir_.A_XX[JD][:, JN])),
            "A_XX": dir_.A_XX[JD][:, JD]}
    for name, M in want.items():      # dense: with periodic x some blocks have no columns
        assert np.array_equal(getattr(neu, name).toarray(), M.toarray()), name


@pytest.mark.parametrize("interpolate", [False, True])
@pytest.mark.parametrize("bx, by", ALL_BCS)
def test_constants_have_zero_flux_through_the_assembled_blocks(interpolate, bx, by):
    """Laplace (c = 0): a constant has zero flux out of every leaf, and the
    unknown columns (pairs summed, single copies once) plus the Dirichlet
    columns cover every leaf face slot exactly once, so A_CC 1 + A_CX 1 = 0.
    A missing or doubled single-copy column breaks this next to a Neumann face."""
    dd = built(interpolate, c=None, bc_types={"x": bx, "y": by})
    r = dd.A_CC @ np.ones(dd.A_CC.shape[1]) + dd.A_CX @ np.ones(dd.A_CX.shape[1])
    assert np.abs(r).max() < 1e-12 * abs(dd.A_CC).sum(axis=1).max()


@pytest.mark.parametrize("interpolate, bc_types", SOLVABLE)
def test_assembled_neumann_system_recovers_skeleton_values(interpolate, bc_types):
    """With the right-hand side built by hand from the blocks, the factorized
    A_CC returns u at XX[Ji]: the assembly on its own, without get_rhs."""
    dd = built(interpolate, bc_types=bc_types)
    u, f, g = manufactured(dd)
    with redirect_stdout(io.StringIO()):
        dd.build_factorize("superLU", False)
    sol = dd._solve_factorized_system(hand_rhs(dd, u, f, g)).ravel()
    true = u(dd.XX_active[dd.Ji])[:, 0].numpy()
    assert np.linalg.norm(sol - true) / np.linalg.norm(true) < 1e-5


def test_neumann_blocks_without_dirichlet_points_have_empty_x_blocks():
    """Periodic x + Neumann y: no Dirichlet point at all, so the X blocks are empty."""
    dd = built(bc_types={"x": "periodic", "y": "neumann"})
    nu = len(dd.I_Ctot) + len(dd.I_Ntot)
    assert len(dd.I_Ntot) > 0 and len(dd.I_Xtot) == 0
    assert dd.Aii.shape == (nu, nu) and dd.Aix.shape == (nu, 0)
    assert dd.Axi.shape == (0, nu) and dd.Axx.shape == (0, 0)
    with pytest.raises(NotImplementedError, match="statically condensed"):
        dd._require_uncondensed_supported()


# ---- Neumann right-hand side -------------------------------------------------------------

@pytest.mark.parametrize("interpolate", [False, True])
@pytest.mark.parametrize("bx, by", NEUMANN_BCS)
def test_get_rhs_matches_the_hand_built_right_hand_side(interpolate, bx, by):
    """get_rhs puts g_N + h[n] - A[n][:, ext] u_D on the Neumann rows, after the
    interior rows, in the order of the assembled blocks."""
    dd = built(interpolate, bc_types={"x": bx, "y": by})
    u, f, g = manufactured(dd)
    b = dd.get_rhs(u, ff_body_vec=f, uu_neu_vec=g)[:, 0].numpy()
    want = hand_rhs(dd, u, f, g)
    assert np.linalg.norm(b - want) <= 1e-13 * np.linalg.norm(want)


@pytest.mark.parametrize("interpolate", [False, True])
@pytest.mark.parametrize("bx", ["dirichlet", "periodic"])
def test_get_rhs_unchanged_without_neumann_faces(interpolate, bx):
    """Without Neumann faces get_rhs is bitwise the previous formula."""
    dd = built(interpolate, bc_types={"x": bx, "y": "dirichlet"})
    u, f, _ = manufactured(dd)
    uD = u(dd.XX_active[dd.I_Xtot])
    old = -apply_sparse_lowmem(dd.A, dd.hps.I_copy1, dd.I_Xtot_in_unique, uD)
    old = old - apply_sparse_lowmem(dd.A, dd.hps.I_copy2, dd.I_Xtot_in_unique, uD)
    old += dd.hps.reduce_body(CPU, None, f)[dd.I_Ctot]
    assert torch.equal(dd.get_rhs(u, ff_body_vec=f), old)


def test_neumann_data_forms():
    """Default zero, vector and function forms, columns, and rejected shapes."""
    dd = built(bc_types={"x": "dirichlet", "y": "neumann"})
    u, f, g = manufactured(dd)
    nN = len(dd.I_Ntot)
    rhs = lambda uu=u, **kw: dd.get_rhs(uu, ff_body_vec=f, **kw)
    assert torch.equal(rhs(), rhs(uu_neu_vec=torch.zeros(nN, 1)))           # no data: du/dn = 0
    assert torch.equal(rhs(uu_neu_vec=g[:, 0]), rhs(uu_neu_vec=g))          # 1-D = one column
    func = lambda xx: torch.sign(xx[:, 1:2]) * u(xx)                         # y faces at -/+0.5
    assert torch.equal(rhs(uu_neu_func=func), rhs(uu_neu_vec=func(dd.XX_active[dd.I_Ntot])))
    assert torch.equal(rhs(uu_neu_func=func, uu_neu_vec=g), rhs(uu_neu_vec=g))   # the vector wins
    # two right-hand sides: column by column, as two separate calls
    u2 = lambda xx: torch.cat((u(xx), 2 * u(xx)), 1)
    both = dd.get_rhs(u2, uu_neu_vec=torch.cat((g, 3 * g), 1))
    assert torch.equal(both, torch.cat((dd.get_rhs(u, uu_neu_vec=g),
                                        dd.get_rhs(lambda xx: 2 * u(xx), uu_neu_vec=3 * g)), 1))
    for bad in (torch.zeros(nN + 1, 1), torch.zeros(nN, 3)):
        with pytest.raises(ValueError, match="Neumann data"):
            rhs(uu_neu_vec=bad)
    no_neu = built(bc_types={"x": "dirichlet", "y": "dirichlet"})
    with pytest.raises(ValueError, match="Neumann data"):
        no_neu.get_rhs(u, uu_neu_vec=g)


def test_verify_discretization_refuses_neumann_faces():
    dd = built(bc_types={"y": "neumann"})
    with pytest.raises(NotImplementedError, match="Neumann"):
        dd.verify_discretization(0)


# ---- singular configurations are refused --------------------------------------------------

@pytest.mark.parametrize("interpolate", [False, True])
@pytest.mark.parametrize("bx", ["neumann", "periodic"])
@pytest.mark.parametrize("c", [None, 0.0])
def test_no_dirichlet_face_and_no_zeroth_order_term_refused_as_singular(interpolate, bx, c):
    """Neumann and periodic faces only, c absent or zero: every constant solves the
    homogeneous problem, so A_CC is singular. build() assembles it; factorizing is refused."""
    dd = built(interpolate, c=c, bc_types={"x": bx, "y": "neumann"})
    r = dd.A_CC @ np.ones(dd.A_CC.shape[1])
    assert np.abs(r).max() < 1e-12 * abs(dd.A_CC).sum(axis=1).max(), "constants are a null vector"
    with pytest.raises(ValueError, match="singular"):
        dd.build_factorize("superLU", False)


@pytest.mark.parametrize("bx", ["neumann", "periodic"])
def test_gauss_faces_without_a_dirichlet_face_refused_if_face_map_has_kernel(bx):
    """If the Gauss-to-Chebyshev face map has a kernel, every leaf DtN shares it
    and, with no Dirichlet face to pin it, A_CC is singular even with c != 0:
    build() assembles the blocks, but factorizing them is refused."""
    dd = built(True, bc_types={"x": bx, "y": "neumann"})
    G = np.asarray(dd.hps.H.Interp_mat_unique)
    if np.linalg.matrix_rank(G) == G.shape[1]:
        pytest.skip("the Gauss-to-Chebyshev face map is injective for this q; nothing to refuse")
    s = np.linalg.svd(dd.A_CC.toarray(), compute_uv=False)
    assert s[-1] < 1e-12 * s[0], "the refused A_CC really is singular"
    with pytest.raises(NotImplementedError, match="kernel"):
        dd.build_factorize("superLU", False)


def test_dirichlet_build_and_solve_unaffected():
    dd = driver(bc_types={"x": "periodic", "y": "dirichlet"})
    with redirect_stdout(io.StringIO()):
        dd.build("reduced_cpu", "superLU", verbose=False)
        dd.build_factorize("superLU", False)
        # u = sin(pi x) e^y is periodic over the box width 2;  -Lap u + u = pi^2 u
        u = lambda xx: (torch.sin(torch.pi * xx[:, 0]) * torch.exp(xx[:, 1])).unsqueeze(-1)
        f = lambda xx: torch.pi**2 * u(xx)
        # body load as a grid vector (callable loads: test/test_body_loads.py)
        sol = dd.solve_dir_full(u, f(dd.XXfull))
    assert rel_err_off_corners(dd, sol, u) < 1e-6
