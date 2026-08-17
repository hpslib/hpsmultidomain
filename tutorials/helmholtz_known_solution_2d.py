"""Short 2D Helmholtz example with a known solution."""

import numpy as np
import torch

from hpsmultidomain import pdo
from hpsmultidomain.built_in_funcs import uu_dir_func_greens
from hpsmultidomain.domain_driver import Domain_Driver
from hpsmultidomain.geom import BoxGeometry
from tutorials.solution_error import relative_solution_error


torch.set_default_dtype(torch.double)
kh = 8.0
source = torch.tensor([11.0, 11.0])


def exact_solution(xx):
    return uu_dir_func_greens(2, xx, kh, center=source)


def main():
    box = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    npan = np.array([4, 4])
    a = (box[1].numpy() - box[0].numpy()) / (2 * npan)
    operator = pdo.PDO_2d(pdo.ones, pdo.ones, c=pdo.const(-kh**2))

    solver = Domain_Driver(BoxGeometry(box), operator, kh, a, p=10, d=2)
    solver.build("reduced_cpu", "mumps", verbose=False)
    solver.build_factorize("mumps", verbose=False)

    solution = solver.solve_dir_full(exact_solution)
    relative_error = relative_solution_error(solver, solution, exact_solution)
    print(f"2D relative error: {relative_error:.3e}")


if __name__ == "__main__":
    main()
