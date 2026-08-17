"""Plotting helper for the 2D hpsmultidomain tutorials."""

import numpy as np
import torch
from scipy.interpolate import BarycentricInterpolator


def interpolate_to_uniform_grid(solver, solution, points_per_leaf=33):
    """Interpolate a 2D HPS solution onto one uniform Cartesian grid.

    Parameters
    ----------
    solver
        The ``Domain_Driver`` that produced the solution.
    solution : torch.Tensor or numpy.ndarray
        Values returned by ``solver.solve_dir_full``.
    points_per_leaf : int, optional
        Uniform samples along each side of a leaf, including endpoints.

    Returns
    -------
    X, Y, values : numpy.ndarray
        Arrays ready for ``matplotlib.pyplot.pcolormesh(X, Y, values)``.
    """
    if solver.d != 2:
        raise ValueError("This tutorial helper supports only 2D solutions.")
    if not isinstance(points_per_leaf, (int, np.integer)) or points_per_leaf < 2:
        raise ValueError("points_per_leaf must be an integer greater than or equal to 2.")

    values = (
        solution.detach().cpu().numpy()
        if isinstance(solution, torch.Tensor)
        else np.asarray(solution)
    )
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("solution must have shape (npoints,) or (npoints, nrhs).")

    nleaf_x, nleaf_y = (int(value) for value in solver.hps.n)
    px, py = (int(value) for value in solver.hps.p)
    expected_points = nleaf_x * nleaf_y * px * py
    if values.shape[0] != expected_points:
        raise ValueError(
            f"solution has {values.shape[0]} points; expected {expected_points}."
        )

    nrhs = values.shape[1]
    leaf_values = values.reshape(nleaf_x, nleaf_y, px, py, nrhs).copy()
    leaf_coords = (
        solver.hps.grid_xx.detach()
        .cpu()
        .numpy()
        .reshape(nleaf_x, nleaf_y, px, py, 2)
    )

    stride = points_per_leaf - 1
    nx = nleaf_x * stride + 1
    ny = nleaf_y * stride + 1
    bounds = solver.geom.bounds
    bounds = (
        bounds.detach().cpu().numpy()
        if isinstance(bounds, torch.Tensor)
        else np.asarray(bounds)
    )
    x_uniform = np.linspace(bounds[0, 0], bounds[1, 0], nx)
    y_uniform = np.linspace(bounds[0, 1], bounds[1, 1], ny)

    uniform_values = np.zeros((nx, ny, nrhs), dtype=values.dtype)
    sample_count = np.zeros((nx, ny, 1), dtype=np.int16)

    for i in range(nleaf_x):
        x_slice = slice(i * stride, i * stride + points_per_leaf)
        for j in range(nleaf_y):
            y_slice = slice(j * stride, j * stride + points_per_leaf)
            x_nodes = leaf_coords[i, j, :, 0, 0]
            y_nodes = leaf_coords[i, j, 0, :, 1]
            current = leaf_values[i, j]

            # The global HPS trace omits leaf corners. Reconstruct each one
            # spectrally from the two adjacent edges before plotting.
            for ix, iy in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
                from_x_edge = BarycentricInterpolator(
                    x_nodes[1:-1], current[1:-1, iy], axis=0
                )(x_nodes[ix])
                from_y_edge = BarycentricInterpolator(
                    y_nodes[1:-1], current[ix, 1:-1], axis=0
                )(y_nodes[iy])
                current[ix, iy] = 0.5 * (from_x_edge + from_y_edge)

            x_query = x_uniform[x_slice]
            y_query = y_uniform[y_slice]
            along_x = BarycentricInterpolator(x_nodes, current, axis=0)(x_query)
            leaf_uniform = BarycentricInterpolator(
                y_nodes, along_x, axis=1
            )(y_query)

            uniform_values[x_slice, y_slice] += leaf_uniform
            sample_count[x_slice, y_slice] += 1

    uniform_values /= sample_count
    X, Y = np.meshgrid(x_uniform, y_uniform, indexing="xy")
    uniform_values = np.swapaxes(uniform_values, 0, 1)
    if nrhs == 1:
        uniform_values = uniform_values[..., 0]
    return X, Y, uniform_values
