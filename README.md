# Hierarchical Poincaré-Steklov Solver for PDEs

The Hierarchical Poincaré-Steklov (HPS) Solver is a high-performance computing solution designed to solve Partial Differential Equations (PDEs) on multidomain geometries. Leveraging advanced numerical methods and efficient parallel processing, this solver is capable of handling complex PDE problems with high accuracy and computational efficiency.

## Features

- **Multidomain Discretization**: Supports solving PDEs on complex geometries divided into multiple domains.
- **Batch Processing**: Utilizes batch operations for efficient computation on both CPU and GPU.
- **Sparse Matrix Representation**: Employs sparse matrices to optimize memory usage and computation time.

## Dependencies

- [PyTorch](https://pytorch.org/): For tensor operations and GPU acceleration.
- [NumPy](https://numpy.org/): For numerical operations.
- [SciPy](https://scipy.org/): For sparse matrix operations and linear algebra.
- [PETSc](https://www.mcs.anl.gov/petsc/) (Optional): For parallel linear algebra operations. The solver can fall back to SciPy if PETSc is not available.

Example usage.
For a 2D problem:
```
python src/argparse_driver.py --pde poisson --domain square --bc log_dist --n 1000 --p 12 --d 2 --solver superLU
```
And for a 3D problem:
```
python src/argparse_driver.py --pde poisson --domain square --bc log_dist --n 50 --p 12 --d 3 --solver MUMPS
```
