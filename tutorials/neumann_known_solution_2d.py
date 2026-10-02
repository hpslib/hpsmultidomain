"""Short 2D example with Neumann boundary conditions and a known solution.

A periodic channel: periodic in x, Neumann on the walls y = -1/2 and y = 1/2,
for the screened operator  -Lap u + u = f  (with only Neumann and periodic faces
the operator needs a zeroth-order term; without one, constants solve the
homogeneous problem and Domain_Driver refuses to factorize it).

Neumann data is the OUTWARD normal derivative du/dn, the plain derivative,
given at the Neumann points solver.XX[solver.I_Ntot]; solver.normals_Ntot holds
their outward unit normals. Without Neumann data, du/dn = 0.
"""

import torch

from hpsmultidomain import pdo
from hpsmultidomain.domain_driver import Domain_Driver
from hpsmultidomain.geom import BoxGeometry
from tutorials.solution_error import relative_solution_error


torch.set_default_dtype(torch.double)


def exact_solution(xx):
    # periodic over the channel length 2; Lap u = (1 - pi^2) u
    return (torch.sin(torch.pi * xx[:, 0]) * torch.exp(xx[:, 1])).unsqueeze(-1)


def exact_gradient(xx):
    e = torch.exp(xx[:, 1])
    return torch.stack((torch.pi * torch.cos(torch.pi * xx[:, 0]) * e,
                        torch.sin(torch.pi * xx[:, 0]) * e), 1)


def solve(p):
    box = torch.tensor([[0.0, -0.5], [2.0, 0.5]])
    operator = pdo.PDO_2d(pdo.ones, pdo.ones, c=pdo.const(1.0))
    solver = Domain_Driver(BoxGeometry(box), operator, 0, 0.25, p=p, d=2,
                           bc_types={"x": "periodic", "y": "neumann"})
    solver.build("reduced_cpu", "mumps", verbose=False)
    solver.build_factorize("mumps", verbose=False)

    # Neumann data: outward du/dn at the Neumann points
    walls = solver.XX[solver.I_Ntot]
    dudn = (exact_gradient(walls) * solver.normals_Ntot).sum(1, keepdim=True)
    # body load f = -Lap u + u = pi^2 u, given on the full grid (see below)
    body = torch.pi**2 * exact_solution(solver.XXfull)

    # No face is Dirichlet, so the Dirichlet data is never evaluated; any
    # callable works. The body load is passed as a grid vector: a callable
    # body load currently fails for real-valued problems.
    solution = solver.solve_dir_full(exact_solution, body, uu_neu=dudn)
    return relative_solution_error(solver, solution, exact_solution)


def main():
    for p in (6, 8, 10, 12):
        print(f"p = {p:2d}: relative error {solve(p):.3e}")


if __name__ == "__main__":
    main()
