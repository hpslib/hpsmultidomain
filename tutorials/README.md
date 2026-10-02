# hpsmultidomain tutorials

Start here if you want to use the solver directly instead of going through the
command-line driver.

## Setup

From the repository root, activate the environment containing PETSc/MUMPS and
install this repository plus the notebook tools:

```bash
conda activate petscenv
python -m pip install -e .
python -m pip install jupyter ipykernel
python -m ipykernel install --user --name petscenv --display-name "Python (petscenv)"
```

If your environment has another name, replace both occurrences of `petscenv`
in the activation and kernel-registration commands.

Confirm that this PETSc build includes MUMPS:

```bash
python -c "from petsc4py import PETSc; assert PETSc.Sys.hasExternalPackage('mumps'), 'PETSc was built without MUMPS'; print('PETSc + MUMPS available')"
```

Then start Jupyter from the repository root so the `tutorials` package is on
the Python path:

```bash
python -m jupyter lab
```

Open `tutorials/constant_coefficient_2d.ipynb` and select
`Python (petscenv)` if Jupyter does not select it automatically. See the main
README for a PETSc/MUMPS source-build recipe.

## Constant-coefficient Helmholtz tutorial

- `constant_coefficient_2d.ipynb`: notebook walkthrough for a 2D
  constant-coefficient Helmholtz problem on a 12-by-6 leaf decomposition. It
  defines the PDE coefficient and boundary excitation, constructs a `PDO_2d`,
  builds a `Domain_Driver`, requests MUMPS, solves for the unknown field, and
  displays it on a uniform Cartesian grid.
- `constant_coefficient_2d.py`: the same example as a plain Python script.
- `uniform_grid.py`: the leafwise interpolation helper used only for tutorial
  visualization.

Run the script version from the repository root:

```bash
python -m tutorials.constant_coefficient_2d
```

The example requests `solver_type = "mumps"`. With a PETSc installation that
passes the setup check above, the factorization uses MUMPS through PETSc.

## Known-solution checks

These short scripts place a Helmholtz Green's-function source outside the
domain, use its nonzero trace as Dirichlet data, solve with zero body load, and
print the relative solution error:

```bash
python -m tutorials.helmholtz_known_solution_2d
python -m tutorials.helmholtz_known_solution_3d
```

## Neumann boundary conditions

`neumann_known_solution_2d.py` solves a periodic channel (periodic in x,
Neumann on the walls in y) with a known solution and prints the error as `p`
grows:

```bash
python -m tutorials.neumann_known_solution_2d
```

Boundary types are set per axis or per face with `bc_types`, e.g.
`{"x": "periodic", "y": "neumann"}` or
`{"x_lo": "dirichlet", "x_hi": "neumann", "y": "neumann"}`. Things to know:

- Neumann data is the outward normal derivative du/dn (the plain derivative,
  whatever the coefficients), given at the Neumann points `solver.XX[solver.I_Ntot]`
  through `uu_neu` in `solve_dir_full` (`uu_neu_func` / `uu_neu_vec` in
  `solve`). `solver.normals_Ntot` holds their outward unit normals. Without
  data, du/dn = 0.
- 2D, box geometries and the statically condensed solver only.
- With no Dirichlet face, the operator needs a zeroth-order term `c`;
  otherwise the problem is singular and factorizing it raises an error.
- Gauss faces (operators with a `c12` term) whose face map has a kernel also
  need at least one Dirichlet face.
- Pass body loads as grid vectors (on `solver.XXfull`) for real-valued
  problems; a callable body load currently fails there.
