# Hierarchical Poincaré-Steklov Solver for PDEs

The Hierarchical Poincaré-Steklov (HPS) Solver is a high-performance computing solution designed to solve Partial Differential Equations (PDEs) on multidomain geometries. Leveraging advanced numerical methods and efficient parallel processing, this solver is capable of handling complex PDE problems with high accuracy and computational efficiency.

The paper connected to this code can be found here: [https://arxiv.org/abs/2503.04033v1](https://arxiv.org/abs/2503.04033v1).

## Features

- **Multidomain Discretization**: Supports solving PDEs on complex geometries divided into multiple domains.
- **Batch Processing**: Utilizes batch operations for efficient computation on both CPU and GPU.
- **Sparse Matrix Representation**: Employs sparse matrices to optimize memory usage and computation time.

## Dependencies

- [PyTorch](https://pytorch.org/): For tensor operations and GPU acceleration.
- [NumPy](https://numpy.org/): For numerical operations.
- [SciPy](https://scipy.org/): For sparse matrix operations and linear algebra.
- [petsc4py](https://petsc.org/release/petsc4py/) (Optional): To use PETSc for sparse matrix operations. The solver can fall back to SciPy if petsc4py is not available. However, PETSc (particularly using the MUMPS direct solver) makes the code much faster.

Example usage.
For a 2D problem:
```
python hps/argparse_driver.py --pde poisson --domain square --bc log_dist --n 1000 --p 12 --d 2 --solver superLU
```
And for a 3D problem:
```
python hps/argparse_driver.py --pde poisson --domain square --bc log_dist --n 50 --p 12 --d 3 --solver MUMPS
```

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
