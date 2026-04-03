import torch
import pytest

from hpsmultidomain.built_in_funcs import uu_dir_func_greens
from hpsmultidomain.domain_driver import Domain_Driver
from hpsmultidomain.geom import BoxGeometry
from hpsmultidomain.pdo import PDO_2d, PDO_3d, const


def build_solver(d, a, p, kh, statically_condense):
    if d == 2:
        box = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        pdo = PDO_2d(c11=const(1.0), c22=const(1.0), c=const(-kh**2))
    else:
        box = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pdo = PDO_3d(c11=const(1.0), c22=const(1.0), c33=const(1.0), c=const(-kh**2))

    solver = Domain_Driver(
        BoxGeometry(box),
        pdo,
        kh,
        a,
        p=p,
        d=d,
        statically_condense=statically_condense,
    )
    solver.build("reduced_cpu", "superLU", verbose=False)
    solver.build_factorize("superLU", False)
    return solver


def solve_full_domain_error(d, a, p, kh, statically_condense):
    solver = build_solver(d, a, p, kh, statically_condense)
    center = solver.geom.bounds[1] + 10

    uu_full_true = uu_dir_func_greens(d, solver.XXfull, kh, center=center)
    uu_active = uu_dir_func_greens(d, solver.XX, kh, center=center)
    uu_sol = solver.solve_dir_full(uu_active[solver.Jx])

    relerr = torch.linalg.norm(uu_sol - uu_full_true) / torch.linalg.norm(uu_full_true)
    return relerr.item(), uu_sol, uu_full_true


@pytest.mark.parametrize(
    "d,a,p,kh",
    [
        (2, 1 / 4, 8, 0),
        (2, 1 / 4, 8, 6),
        (3, 1 / 4, 6, 0),
        (3, 1 / 4, 6, 4),
    ],
)
def test_static_condense_toggle_full_domain_error(d, a, p, kh):
    err_condensed, sol_condensed, uu_true = solve_full_domain_error(d, a, p, kh, True)
    err_uncondensed, sol_uncondensed, _ = solve_full_domain_error(d, a, p, kh, False)

    sol_diff = torch.linalg.norm(sol_condensed - sol_uncondensed) / torch.linalg.norm(uu_true)

    assert err_uncondensed <= err_condensed + 1e-12
    assert sol_diff.item() <= max(1e-10, 5 * err_condensed)
