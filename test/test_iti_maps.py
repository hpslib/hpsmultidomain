import pytest
import numpy as np
import torch

from hpsmultidomain.built_in_funcs import uu_dir_func_greens
from hpsmultidomain.domain_driver import Domain_Driver
from hpsmultidomain.geom import BoxGeometry
from hpsmultidomain.pdo import PDO_2d, PDO_3d, const


torch.set_default_dtype(torch.double)


def _make_manufactured_problem(dim, kh):
    if dim == 2:
        wave = (4.0, 1.0)
        pdo = PDO_2d(c11=const(1.0), c22=const(1.0), c=const(-kh**2))
        box = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        def u_true(xx):
            alpha, beta = wave
            return torch.exp(1j * (alpha * xx[:, 0:1] + beta * xx[:, 1:2]))

        def f_body(xx):
            alpha, beta = wave
            return (alpha**2 + beta**2 - kh**2) * u_true(xx)

        expected_npan = {
            0.25: (2, 2),
            0.125: (4, 4),
        }
    else:
        wave = (2.0, 1.0, 3.0)
        pdo = PDO_3d(c11=const(1.0), c22=const(1.0), c33=const(1.0), c=const(-kh**2))
        box = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

        def u_true(xx):
            alpha, beta, gamma = wave
            return torch.exp(1j * (alpha * xx[:, 0:1] + beta * xx[:, 1:2] + gamma * xx[:, 2:3]))

        def f_body(xx):
            alpha, beta, gamma = wave
            return (alpha**2 + beta**2 + gamma**2 - kh**2) * u_true(xx)

        expected_npan = {
            0.25: (2, 2, 2),
        }

    return BoxGeometry(box), pdo, u_true, f_body, expected_npan


def _run_iti_comparison(dim, a, p, kh):
    geom, pdo, u_true, f_body, expected_npan = _make_manufactured_problem(dim, kh)

    solver_dtn = Domain_Driver(geom, pdo, kh, a, p=p, d=dim, use_iti_maps=False)
    solver_dtn.build('reduced_cpu', 'superLU', verbose=False)
    solver_dtn.build_factorize('superLU', False)
    sol_dtn = solver_dtn.solve_dir_full(u_true, f_body)
    true_dtn = u_true(solver_dtn.XXfull)
    err_dtn = (torch.linalg.norm(sol_dtn - true_dtn) / torch.linalg.norm(true_dtn)).item()

    solver_iti = Domain_Driver(geom, pdo, kh, a, p=p, d=dim, use_iti_maps=True, impedance_eta=kh)
    solver_iti.build('reduced_cpu', 'superLU', verbose=False)
    assert solver_iti.Aii.shape[0] == 2 * solver_dtn.Aii.shape[0]
    assert solver_iti.Aii.shape[0] == solver_iti.iti_ndirected
    solver_iti.build_factorize('superLU', False)
    sol_iti_dir = solver_iti.solve_dir_full(u_true, f_body)
    true_iti = u_true(solver_iti.XXfull)

    err_iti_dir = (torch.linalg.norm(sol_iti_dir - true_iti) / torch.linalg.norm(true_iti)).item()
    dir_diff = (torch.linalg.norm(sol_iti_dir - sol_dtn) / torch.linalg.norm(sol_dtn)).item()
    npan = tuple(int(v) for v in solver_iti.hps.n.tolist())

    return {
        'err_dtn': err_dtn,
        'err_iti_dir': err_iti_dir,
        'dir_diff': dir_diff,
        'npan': npan,
        'expected_npan': expected_npan[a],
    }


@pytest.mark.parametrize(
    "a,p,kh",
    [
        (0.25, 8, 6.0),
        (0.125, 8, 6.0),
    ],
)
def test_iti_dirichlet_matches_dtn_2d(a, p, kh):
    results = _run_iti_comparison(dim=2, a=a, p=p, kh=kh)
    assert results['npan'] == results['expected_npan']
    assert results['err_iti_dir'] <= results['err_dtn'] * (1 + 1e-10) + 1e-12
    assert results['dir_diff'] < 1e-10


def test_iti_maps_3d():
    results = _run_iti_comparison(dim=3, a=0.25, p=5, kh=5.0)
    assert results['npan'] == results['expected_npan']
    assert results['err_iti_dir'] <= results['err_dtn'] * (1 + 1e-10) + 1e-12
    assert results['dir_diff'] < 1e-10


@pytest.mark.parametrize(
    "dim,a,p,kh,center",
    [
        (2, 0.25, 8, 6.0, torch.tensor([-0.75, 1.35])),
        (3, 0.25, 5, 5.0, torch.tensor([-0.75, 1.35, 1.2])),
    ],
)
def test_iti_maps_known_constant_helmholtz_solution(dim, a, p, kh, center):
    if dim == 2:
        geom = BoxGeometry(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
        pdo = PDO_2d(c11=const(1.0), c22=const(1.0), c=const(-kh**2))
    else:
        geom = BoxGeometry(torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
        pdo = PDO_3d(c11=const(1.0), c22=const(1.0), c33=const(1.0), c=const(-kh**2))

    def u_true(xx):
        return uu_dir_func_greens(dim, xx, kh, center=center)

    solver_dtn = Domain_Driver(geom, pdo, kh, a, p=p, d=dim, use_iti_maps=False)
    solver_dtn.build('reduced_cpu', 'superLU', verbose=False)
    solver_dtn.build_factorize('superLU', False)
    sol_dtn = solver_dtn.solve_dir_full(u_true)
    true_dtn = u_true(solver_dtn.XXfull)
    err_dtn = torch.linalg.norm(sol_dtn - true_dtn) / torch.linalg.norm(true_dtn)

    solver_iti = Domain_Driver(geom, pdo, kh, a, p=p, d=dim, use_iti_maps=True, impedance_eta=kh)
    solver_iti.build('reduced_cpu', 'superLU', verbose=False)
    assert solver_iti.Aii.shape[0] == 2 * solver_dtn.Aii.shape[0]
    assert solver_iti.Aii.shape[0] == solver_iti.iti_ndirected
    solver_iti.build_factorize('superLU', False)
    sol_iti = solver_iti.solve_dir_full(u_true)
    true_iti = u_true(solver_iti.XXfull)
    err_iti = torch.linalg.norm(sol_iti - true_iti) / torch.linalg.norm(true_iti)

    sol_diff = torch.linalg.norm(sol_iti - sol_dtn) / torch.linalg.norm(sol_dtn)

    assert err_iti <= err_dtn * (1 + 1e-10) + 1e-12
    assert sol_diff < 1e-10


def test_doubled_iti_system_maps_impedance_to_impedance_2d():
    kh = 5.0
    eta = kh
    alpha = 3.0
    beta = 4.0
    geom = BoxGeometry(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    pdo = PDO_2d(c11=const(1.0), c22=const(1.0), c=const(-kh**2))

    def u_true(xx):
        phase = alpha * xx[:, 0:1] + beta * xx[:, 1:2]
        return torch.exp(1j * phase)

    solver = Domain_Driver(geom, pdo, kh, 0.25, p=12, d=2, use_iti_maps=True, impedance_eta=eta)
    solver.build('reduced_cpu', 'superLU', verbose=False)

    size_ext = solver.iti_size_ext
    local_surface = np.arange(int(solver.hps.nboxes) * size_ext, dtype=int).reshape(int(solver.hps.nboxes), size_ext)
    x_dir = np.zeros((solver.iti_ndirected, 1), dtype=np.complex128)
    u_ext = np.zeros((len(solver.I_Xtot), 1), dtype=np.complex128)
    local_res_num = 0.0
    local_res_den = 0.0

    for box, leaf_data in enumerate(solver.iti_leaf_data):
        loc_int = leaf_data['local_int']
        loc_ext = leaf_data['local_ext']
        glob_int = leaf_data['global_int']
        glob_ext = leaf_data['global_ext']
        dup_inds = local_surface[box]

        xx_leaf = solver.hps.xx_ext[dup_inds]
        u_leaf = u_true(xx_leaf).detach().cpu().numpy().astype(np.complex128)
        u_int = u_leaf[loc_int]
        u_bdry = u_leaf[loc_ext]

        flux_int = leaf_data['dir_T_cc'] @ u_int + leaf_data['dir_T_cx'] @ u_bdry
        incoming = 1j * eta * u_int - flux_int
        outgoing = 1j * eta * u_int + flux_int

        outgoing_from_map = leaf_data['dir_R'] @ incoming + leaf_data['dir_B'] @ u_bdry
        local_res_num += np.linalg.norm(outgoing_from_map - outgoing) ** 2
        local_res_den += np.linalg.norm(outgoing) ** 2

        x_dir[glob_int] = incoming
        u_ext[glob_ext] = u_bdry

    local_relerr = np.sqrt(local_res_num / local_res_den)
    system_residual = solver.A_CC @ x_dir + solver.A_CX @ u_ext
    system_relerr = np.linalg.norm(system_residual) / max(np.linalg.norm(x_dir), 1e-15)

    assert local_relerr < 1e-12
    assert system_relerr < 1e-10


def test_doubled_iti_system_matches_directed_equations():
    kh = 5.0
    geom = BoxGeometry(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    pdo = PDO_2d(c11=const(1.0), c22=const(1.0), c=const(-kh**2))
    solver = Domain_Driver(geom, pdo, kh, 0.25, p=8, d=2, use_iti_maps=True, impedance_eta=kh)
    solver.build('reduced_cpu', 'superLU', verbose=False)

    rng = np.random.default_rng(17)
    u_ext = rng.standard_normal((len(solver.I_Xtot), 2)) + 1j * rng.standard_normal((len(solver.I_Xtot), 2))

    rhs = -solver.A_CX @ u_ext
    x_dir = np.linalg.solve(solver.A_CC.toarray(), rhs)

    directed_A = np.eye(solver.iti_ndirected, dtype=np.complex128) - solver.iti_swap @ solver.iti_R_dir
    directed_CX = -(solver.iti_swap @ solver.iti_B_dir)
    directed_res = directed_A @ x_dir + directed_CX @ u_ext
    directed_relerr = np.linalg.norm(directed_res) / max(np.linalg.norm(x_dir), 1e-15)

    assert x_dir.shape[0] == solver.Aii.shape[0]
    assert x_dir.shape[0] == solver.iti_ndirected
    assert directed_relerr < 1e-10
