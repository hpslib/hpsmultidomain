"""Solve and plot a constant-coefficient 2D Helmholtz problem.

This tutorial uses the hpsmultidomain API directly. It solves

    -Delta u - kh^2 u = 0

on [0, 2] x [0, 1]. A localized Dirichlet signal enters at the left boundary,
and the solution is treated as unknown. The HPS result is interpolated onto a
uniform Cartesian grid only for visualization.

Run from the repository root with:

    python -m tutorials.constant_coefficient_2d
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from hpsmultidomain import pdo
from hpsmultidomain.domain_driver import Domain_Driver
from hpsmultidomain.geom import BoxGeometry
from tutorials.uniform_grid import interpolate_to_uniform_grid


torch.set_default_dtype(torch.double)


def constant_bfield(xx, kh):
    """Return the constant zeroth-order coefficient c(x) = -kh^2."""
    return -(kh**2) * torch.ones(
        xx.shape[0], dtype=xx.dtype, device=xx.device
    )


def boundary_data(xx):
    """Apply a localized Dirichlet excitation on the left boundary."""
    x = xx[:, 0]
    y = xx[:, 1]
    left_boundary = torch.isclose(
        x, torch.tensor(0.0, dtype=x.dtype, device=x.device)
    )
    profile = torch.exp(-100.0 * (y - 0.5) ** 2)
    return (left_boundary * profile).unsqueeze(-1).to(torch.cdouble)


def build_solver(kh=12.0, p=(12, 12), npan=(12, 6), solver_type="mumps"):
    """Construct, assemble, and factorize the tutorial problem."""
    box = torch.tensor([[0.0, 0.0], [2.0, 1.0]])
    geometry = BoxGeometry(box)
    p = np.asarray(p, dtype=int)
    npan = np.asarray(npan, dtype=int)
    leaf_half_width = (box[1].numpy() - box[0].numpy()) / (2 * npan)

    operator = pdo.PDO_2d(
        c11=pdo.ones,
        c22=pdo.ones,
        c=lambda xx: constant_bfield(xx, kh),
    )

    solver = Domain_Driver(
        geometry,
        operator,
        kh,
        leaf_half_width,
        p=p,
        d=2,
    )
    solver.build("reduced_cpu", solver_type, verbose=False)
    solver.build_factorize(solver_type, verbose=False)
    return solver


def plot_solution(X, Y, solution_grid):
    """Plot the real part of a solution already sampled on a uniform grid."""
    field = np.real(solution_grid)
    limit = np.max(np.abs(field))

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    image = ax.pcolormesh(
        X,
        Y,
        field,
        shading="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        rasterized=True,
    )
    ax.set(
        title="Helmholtz solution, real part",
        xlabel="x",
        ylabel="y",
        aspect="equal",
    )
    fig.colorbar(image, ax=ax, label="Re(u)")
    return fig, ax


def main():
    solver = build_solver()
    solution = solver.solve_dir_full(boundary_data)
    X, Y, solution_grid = interpolate_to_uniform_grid(
        solver, solution, points_per_leaf=33
    )

    print(f"HPS solution values: {solution.shape[0]:,}")
    print(f"Uniform plotting grid: {X.shape[1]} x {X.shape[0]}")
    plot_solution(X, Y, solution_grid)
    plt.show()


if __name__ == "__main__":
    main()
