# hpsmultidomain
## Hierarchical Poincaré-Steklov Solver for PDEs

[![License](https://img.shields.io/github/license/hpslib/hpsmultidomain)](./LICENSE.md)
[![Top language](https://img.shields.io/github/languages/top/hpslib/hpsmultidomain)](https://www.python.org)
![Code size](https://img.shields.io/github/languages/code-size/hpslib/hpsmultidomain)
[![Latest commit](https://img.shields.io/github/last-commit/hpslib/hpsmultidomain)](https://github.com/annayesy/slabLU/commits/master)

Written by Joseph Kump and Anna Yesypenko

The Hierarchical Poincaré-Steklov (HPS) Solver is a high-performance computing solution designed to solve Partial Differential Equations (PDEs) on multidomain geometries. Leveraging advanced numerical methods and efficient parallel processing including GPUs, this solver is capable of handling complex PDE problems with high accuracy and computational efficiency.

## Features

- **Multidomain Discretization**: Supports solving PDEs on complex geometries divided into multiple domains.
- **Batch Processing**: Utilizes batch operations for efficient computation on both CPU and GPU.
- **Sparse Matrix Representation**: Employs sparse matrices to optimize memory usage and computation time.

<p align="center">
    <img src="https://github.com/hpslib/hpsmultidomain/blob/main/figures/gravity_helmholtz_low_res.png" width="49%"/> <img src="https://github.com/hpslib/hpsmultidomain/blob/main/figures/gravity_helmholtz_high_res.png" width="49%" /> 
</p>

<div style="display: flex; justify-content: center;">
    <p style="width: 50%; text-align: center; font-size: 90%;">
        Figures 1 and 2: Plots of solutions to the gravity Helmholtz equation, $\Delta u  + \kappa^2 (1-x_3) u = -1$ on $\Omega = [1.1, 2.1] \times [-1, 0] \times [-1.2, -0.2]$ with a zero DBC. A cross-section has been taken through the $x$-axis. We see consistent results across different $p$ and $h$.
    </p>
</div>

<p align="center">
    <img src="https://github.com/hpslib/hpsmultidomain/blob/main/figures/picture_sinusoidal_curve.png" width="49%"/> <img src="https://github.com/hpslib/hpsmultidomain/blob/main/figures/picture_annulus.png" width="49%" /> 
</p>

<div style="display: flex; justify-content: center;">
    <p style="width: 50%; text-align: center; font-size: 90%;">
        Figures 3 and 4: Plots of solutions to a constant-coefficient Helmholtz equation on different domain geometries.
    </p>
</div>

## Dependencies

- [PyTorch](https://pytorch.org/): For tensor operations and GPU acceleration.
- [NumPy](https://numpy.org/): For numerical operations.
- [SciPy](https://scipy.org/): For sparse matrix operations and linear algebra.
- [petsc4py](https://petsc.org/release/petsc4py/) (Optional): To use PETSc for sparse matrix operations. The solver can fall back to SciPy if petsc4py is not available. However, PETSc (particularly using the MUMPS direct solver) makes the code much faster.

## Example usage
For a 2D problem:
```
python hps/argparse_driver.py --pde poisson --domain square --bc log_dist --n 1000 --p 12 --d 2 --solver superLU
```
And for a 3D problem:
```
python hps/argparse_driver.py --pde poisson --domain square --bc log_dist --n 50 --p 12 --d 3 --solver MUMPS
```

## Notes
A series of command line arguments can be seen in `argparse_driver.py`. These include:
- `pde` to specify the partial differential equation to solve, such as `poisson` or `bfield_constant` (i.e. constant-coefficient Helmholtz equation)
- `domain` to specify domain shapes. `square` is the standard for rectangular domains, `curved` (sinusoidal curve) and `annulus` are alternatives.
- `bc` to supply a Dirichlet BC for the problem. Some of these comdbined with certain PDEs have analytic solutions, others do not.
- `d` for the domain dimension, either 2 or 3
- `p` for the discretization's polynomial order on each subdomain. There will be $p^d$ points per subdomain.
- `n` for the total number of discretization points along each axis. This can be either a single number (assuming a square discretization) or two/three numbers set to `n0`, `n1`, `n2` for non-square discretizations. For now, each axis must have a number of points equal to a multiple of (p-2)
- `solver` for the solver of choice for the sparse matrix factorization, such as `MUMPS` or `superLU`
- `box_xlim`/`box_ylim`/`box_zlim` to set the physical dimensions of the 2D or 3D domain (defaults to [0,1]^d)
- `kh` for the wavenumber in PDEs with wavenumbers such as `bfield_constant`
- `ppw` to specify the number of discretization points per wavelength in PDEs with wavenumbers such as `bfield_constant`. This automatically sets the wavenumber, so it is incompatible with `kh`
- `sparse_assembly` to specify whether operations with PyTorch tensors are done on GPUs or CPUs. If not specified then the code will check if a GPU is available and otherwise resort to CPU.
- `pickle` to specify a filepath to store results of the run
- `visualize` to plot the PDE solution

If you wish to run the solver multiple times for the same 3D problem (same PDE and BC) across a range of `p` and `n` values, you can run `run_models.py` with the problem specified. This calls `test_3d.py` to automatically piece together the commandline arguments and run the solver several times in a row.

Alternatively, you can call `test_hps_multidomain.py` or `test_hps_multidomain_curved.py` to test a few various 2D and 3D configurations.

## Associated Papers

The paper connected to this code can be found here:

[1] Kump, Joseph; Yesypenko, Anna; and Per-Gunnar Martinsson. "A Two-Level Direct Solver for the Hierarchical Poincaré-Steklov Method" arXiv preprint arXiv:2503.04033 (2025).
