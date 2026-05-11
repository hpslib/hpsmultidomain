import argparse
import os
import pickle

import numpy as np
import torch

from hpsmultidomain.driver_build import build_operator_with_info, configure_pde_domain
from hpsmultidomain.driver_solve import run_solver
from hpsmultidomain.domain_driver import Domain_Driver
from hpsmultidomain.geom import BoxGeometry
from hpsmultidomain.visualize_problem import visualize_problem

torch.set_default_dtype(torch.double)


def build_parser():
    parser = argparse.ArgumentParser("Call direct solver for 2D/3D domain.")

    parser.add_argument('--p', type=int, required=False)
    parser.add_argument('--n', type=int, required=False)
    parser.add_argument('--d', type=int, required=False, default=2)

    parser.add_argument('--n0', type=int, required=False)
    parser.add_argument('--n1', type=int, required=False)
    parser.add_argument('--n2', type=int, required=False)

    parser.add_argument('--p0', type=int, required=False)
    parser.add_argument('--p1', type=int, required=False)
    parser.add_argument('--p2', type=int, required=False)

    parser.add_argument('--pde', type=str, required=True)
    parser.add_argument('--domain', type=str, required=True)
    parser.add_argument('--box_xlim', type=float, required=False, default=1.0)
    parser.add_argument('--box_ylim', type=float, required=False, default=1.0)
    parser.add_argument('--box_zlim', type=float, required=False, default=1.0)

    parser.add_argument('--bc', type=str, required=True)
    parser.add_argument('--ppw', type=int, required=False)
    parser.add_argument('--nwaves', type=float, required=False)
    parser.add_argument('--kh', type=float, required=False)
    parser.add_argument('--delta_t', type=float, required=False)
    parser.add_argument('--num_timesteps', type=int, required=False)

    parser.add_argument('--solver', type=str, required=False)
    parser.add_argument('--sparse_assembly', type=str, required=False, default='reduced_gpu')
    parser.add_argument('--pickle', type=str, required=False)
    parser.add_argument('--store_sol', action='store_true')
    parser.add_argument('--disable_cuda', action='store_true')
    parser.add_argument('--periodic_bc', action='store_true')

    parser.add_argument('--test_components', type=bool, required=False, default=False)
    parser.add_argument('--visualize', type=bool, required=False, default=False)

    return parser


def _expand_dim_arg(shared_value, component_values, dims, name):
    if shared_value is not None:
        arr = np.asarray(shared_value)
        if arr.ndim == 0:
            return np.full(dims, int(arr), dtype=int)
        if arr.shape == (dims,):
            return arr.astype(int)
        raise ValueError(f"{name} must be a scalar or length-{dims} array")

    values = component_values[:dims]
    if all(value is not None for value in values):
        return np.array(values, dtype=int)

    joined = ",".join(f"{name}{i}" for i in range(dims))
    raise ValueError(f"Need to set either {name} or ({joined})")


def normalize_args(args):
    if args.d not in (2, 3):
        raise ValueError("dimension d must be 2 or 3")

    args.n = _expand_dim_arg(args.n, (args.n0, args.n1, args.n2), args.d, "n")
    args.p = _expand_dim_arg(args.p, (args.p0, args.p1, args.p2), args.d, "p")
    return args


def run_from_args(args):
    args = normalize_args(args)

    if args.d == 2:
        box = torch.tensor([[0, 0], [args.box_xlim, args.box_ylim]])
    else:
        box = torch.tensor([[0, 0, 0], [args.box_xlim, args.box_ylim, args.box_zlim]])
    box_geom = BoxGeometry(box)

    if args.ppw is not None:
        print("\n RUNNING PROBLEM WITH...")
    else:
        print("RUNNING PROBLEM WITH...")

    if args.disable_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print("CUDA available %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        print("--num cuda devices %d" % torch.cuda.device_count())
    if ((not torch.cuda.is_available()) and (args.sparse_assembly == 'reduced_gpu')):
        args.sparse_assembly = 'reduced_cpu'
        print("Changed sparse assembly to reduced_cpu")

    op, param_map, inv_param_map, curved_domain, kh, delta_t, num_timesteps = configure_pde_domain(args)

    p = args.p
    npan = args.n / (p - 2)
    if args.d == 2:
        a = np.array([args.box_xlim, args.box_ylim]) / (2 * npan)
    else:
        a = np.array([args.box_xlim, args.box_ylim, args.box_zlim]) / (2 * npan)

    print("p = ", p)
    print("a = ", a)

    dom = Domain_Driver(box_geom, op, kh, a, p=p, d=args.d, periodic_bc=args.periodic_bc)

    print(args.sparse_assembly)
    build_info = build_operator_with_info(dom, args, box_geom, kh)

    uu_dir, uu_sol, res, true_res, resloc_hps, toc_solve, forward_bdry_error, reverse_bdry_error, solve_info = run_solver(
        dom, args, curved_domain, kh, param_map, delta_t, num_timesteps
    )
    print(uu_sol.shape)

    if args.store_sol:
        print("\t--Storing solution")
        solve_info['xx'] = dom.hps.xx_tot
        solve_info['sol'] = uu_sol

    info = dict(build_info)
    info.update(solve_info)

    if args.pickle is not None:
        print("Pickling results to file %s" % args.pickle)
        with open(args.pickle, "wb+") as f:
            pickle.dump(info, f)

    if args.visualize:
        if args.d == 2:
            print("Warning: visualization for d=2 not yet implemented")
        else:
            visualize_problem(dom, curved_domain, param_map, uu_sol, p, kh)

    return {
        "args": args,
        "dom": dom,
        "box_geom": box_geom,
        "param_map": param_map,
        "inv_param_map": inv_param_map,
        "curved_domain": curved_domain,
        "kh": kh,
        "delta_t": delta_t,
        "num_timesteps": num_timesteps,
        "build_info": build_info,
        "solve_info": solve_info,
        "info": info,
        "uu_dir": uu_dir,
        "uu_sol": uu_sol,
        "res": res,
        "true_res": true_res,
        "resloc_hps": resloc_hps,
        "toc_solve": toc_solve,
        "forward_bdry_error": forward_bdry_error,
        "reverse_bdry_error": reverse_bdry_error,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_args(args)


if __name__ == "__main__":
    main()
