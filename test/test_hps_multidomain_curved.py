import torch
import numpy as np

from geometry import ParametrizedGeometry2D, ParametrizedGeometry3D
from hps_multidomain import HPSMultidomain

def get_known_greens(points, kh, center):
    # Dummy: returns exp(-|x|^2)
    raise ValueError("todo")

def test_hps_multidomain_curved_2d():
    a = 1 / 8
    p = 16
    kh = 0

    mag = 0.3
    psi = lambda x: 1 - mag * torch.sin(4 * x)
    dpsi = lambda x: -mag * 4 * torch.cos(4 * x)
    ddpsi = lambda x: mag * 16 * torch.sin(4 * x)

    z1 = lambda xx: xx[..., 0]
    z2 = lambda xx: xx[..., 1] / psi(xx[..., 0])

    y1 = lambda xx: xx[..., 0]
    y2 = lambda xx: xx[..., 1] * psi(xx[..., 0])

    y1_d1 = lambda xx: torch.ones_like(xx[..., 0])
    y2_d1 = lambda xx: xx[..., 1] * dpsi(xx[..., 0])
    y2_d2 = lambda xx: psi(xx[..., 0])
    y2_d1d1 = lambda xx: xx[..., 1] * ddpsi(xx[..., 0])

    box_geom = torch.tensor([[0, 0], [1.0, 1.0]])

    param_geom = ParametrizedGeometry2D(
        box_geom, z1, z2, y1, y2,
        y1_d1=y1_d1, y2_d1=y2_d1,
        y2_d2=y2_d2, y2_d1d1=y2_d1d1
    )

    def bfield_constant(xx, kh):
        return -(kh ** 2) * torch.ones_like(xx[..., 0])

    pdo_mod = param_geom.transform_helmholtz_pdo(bfield_constant, kh)

    solver = HPSMultidomain(pdo_mod, param_geom, a, p)
    relerr = solver.verify_discretization(kh)
    assert relerr < 1e-6, f"Relative error too high in 2D: {relerr:.2e}"

    points_bnd = solver.geom.parameter_map(solver.XX)
    points_full = solver.geom.parameter_map(solver._XXfull)

    uu_full = get_known_greens(points_full, kh, center=solver.geom.bounds[1]+10)
    uu_bnd = get_known_greens(points_bnd, kh, center=solver.geom.bounds[1]+10)

    uu_sol = solver.solve_dir_full(uu_bnd[solver.Jx])
    relerr = np.linalg.norm(uu_sol - uu_full) / np.linalg.norm(uu_full)
    assert relerr < 3e-10, f"Relative error too high in 2D: {relerr:.2e}"

    assert kh == 0

    def get_mms(points):
        return np.sin(np.pi * points[:, 0]) * np.cos(np.pi * points[:, 1])[:, np.newaxis]

    def get_body_load(points):
        ff = 2 * (np.pi**2) * get_mms(points)
        return ff[:, np.newaxis]

    uu_full = get_mms(points_full.numpy())
    uu_bnd = get_mms(points_bnd.numpy())

    uu_sol = solver.solve_dir_full(uu_bnd[solver.Jx], get_body_load(points_full.numpy()))
    relerr = np.linalg.norm(uu_sol - uu_full) / np.linalg.norm(uu_full)
    assert relerr < 3e-7, f"Relative error too high in 2D: {relerr:.2e}"
