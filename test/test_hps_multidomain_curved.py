import torch
import numpy as np

from hps.geom import ParametrizedGeometry2D, ParametrizedGeometry3D
from hps.domain_driver import Domain_Driver

from hps.built_in_funcs import uu_dir_func_greens

def test_hps_multidomain_curved_2d(sparse_assembly='reduced_gpu',solver_type='MUMPS'):

    # Check CUDA availability and adjust settings accordingly
    print("CUDA available %s" % torch.cuda.is_available())
    if (torch.cuda.is_available()):
        print("--num cuda devices %d" % torch.cuda.device_count())
    if ((not torch.cuda.is_available()) and (sparse_assembly == 'reduced_gpu')):
        sparse_assembly = 'reduced_cpu'
        print("Changed sparse assembly to reduced_cpu")
    
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

    solver = Domain_Driver(param_geom, pdo_mod, 0, a, p=p, d=2)
    solver.build(sparse_assembly, solver_type,verbose=False)
    relerr = solver.verify_discretization(kh)
    assert relerr < 1e-6, f"Relative error too high in 2D: {relerr:.2e}"
    print("Relative error for box interfaces: ", relerr)

    
    points_bnd = solver.geom.parameter_map(solver.XX)
    points_full = solver.geom.parameter_map(solver.XXfull)
    
    uu_full = uu_dir_func_greens(2,points_full,kh,center=solver.geom.bounds[1]+10)
    uu_bnd  = uu_dir_func_greens(2,points_bnd,kh,center=solver.geom.bounds[1]+10)

    uu_sol = solver.solve_dir_full(uu_bnd[solver.Jx])
    relerr = np.linalg.norm(uu_sol - uu_full) / np.linalg.norm(uu_full)
    assert relerr < 3e-10, f"Relative error too high in 2D: {relerr:.2e}"
    print("Relative error for box interfaces and interiors: ", relerr)

    assert kh == 0
    
    def get_mms(points):
        return (torch.sin(torch.pi * points[:, 0]) * torch.cos(torch.pi * points[:, 1])).unsqueeze(-1)

    def get_body_load(points):
        ff = 2 * (torch.pi**2) * get_mms(points)
        return ff

    uu_full = get_mms(points_full)
    uu_bnd = get_mms(points_bnd)

    uu_sol = solver.solve_dir_full(uu_bnd[solver.Jx], get_body_load(points_full))
    relerr = np.linalg.norm(uu_sol - uu_full) / np.linalg.norm(uu_full)
    assert relerr < 3e-7, f"Relative error too high in 2D: {relerr:.2e}"
    print("Relative error for box interfaces and interiors with body load: ", relerr)
    

test_hps_multidomain_curved_2d()