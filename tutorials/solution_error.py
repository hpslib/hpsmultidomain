"""Error helper for known-solution tutorials."""

import numpy as np
import torch


def relative_solution_error(solver, solution, exact_solution):
    """Return the relative error on the independent HPS collocation values."""
    nboxes = int(solver.hps.nboxes)
    points_per_leaf = int(np.prod(solver.hps.p))
    numerical = solution.reshape(nboxes, points_per_leaf, -1)
    reference = exact_solution(solver.XXfull).reshape(
        nboxes, points_per_leaf, -1
    )
    collocation = torch.as_tensor(solver.hps.H.JJ.Jtot)
    error = torch.linalg.norm(
        numerical[:, collocation] - reference[:, collocation]
    )
    return (error / torch.linalg.norm(reference[:, collocation])).item()
