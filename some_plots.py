import numpy as np

import matplotlib.pyplot as plt

# Number of boxes along an axis (so the total is cubed)
# These are for p=12:
nBoxes = np.array([2, 3, 4, 5, 6, 8, 10, 12])
# What if we make this DOF?
#nBoxes = (10*nBoxes) ** 3

assemblyTime = [0.35, 1.04, 2.39, 4.79, 8.22, 19.38, 39.04, 77.88]

memory = [0.07, 0.24, 0.57, 1.12, 1.94, 4.59, 8.96, 15.49]

inversionTime = [0.03, 0.36, 1.83, 5.08, 13.00, 51.48, 168.02, 434.81]

solveTime = [0.11, 0.34, 0.84, 1.69, 3.04, 7.19, 14.55, 31.59]

leafSolveError = [5.06e-04, 9.12e-05, 1.49e-05, 6.35e-06, 5.95e-06, 9.16e-07, 6.17e-07, 1.57e-07]

interpolationError = [0.0154, 0.0160, 0.0005, 5.3633e-05, 1.4921e-05, 5.8685e-07, 2.2868e-07, 3.9676e-08]
"""
plt.plot(nBoxes, assemblyTime)
plt.title("Assembly Times, p=12")
plt.xlabel("Number of boxes along one axis")
plt.ylabel("Time to assemble sparse matrix")
plt.show()

plt.plot(nBoxes, memory)
plt.title("Sparse matrix memory, p=12")
plt.xlabel("Number of boxes along one axis")
plt.ylabel("Memory for sparse matrix (GB)")
plt.show()

plt.plot(nBoxes, inversionTime)
plt.title("Inversion Times, p=12")
plt.xlabel("Number of boxes along one axis")
plt.ylabel("Time to factorize LU for sparse matrix")
plt.show()

plt.plot(nBoxes, solveTime)
plt.title("Solve Times, p=12")
plt.xlabel("Number of boxes along one axis")
plt.ylabel("Time to solve for one RHS")
plt.show()

plt.plot(nBoxes, leafSolveError)
plt.title("Accuracy of leaf-level solve, p=12")
plt.xlabel("Number of boxes along one axis")
plt.ylabel("Relative error")
plt.show()

plt.plot(nBoxes, interpolationError)
plt.title("Accuracy of box interpolation, p=12")
plt.xlabel("Number of boxes along one axis (domain fixed)")
plt.ylabel("Relative error")
plt.show()
"""
nBoxes = (10*nBoxes) ** 3
plt.plot(nBoxes, assemblyTime)
plt.plot(nBoxes, memory)
plt.plot(nBoxes, inversionTime)
plt.plot(nBoxes, solveTime)
plt.title("Runtimes for Problem, p=12")
plt.xlabel("N = n^3")
plt.ylabel("Runtime (s) or memory (GB)")
plt.legend(["Assembly", "Memory", "LU", "Solve"])
plt.show()

#nBoxes = (10*nBoxes) ** 3

plt.semilogy(nBoxes, leafSolveError)
plt.semilogy(nBoxes, interpolationError)
plt.title("Accuracy for Laplace equation, p=12")
plt.xlabel("Degrees of freedom (nodes per box X number of boxes)")
plt.ylabel("Relative error")
plt.legend(["Leaf solve", "Box interpolation"])
plt.show()


# P refinement tests with Helmholtz, 10 ppw
N12 = np.array([20, 30, 40, 50, 60, 70, 80]) ** 3
trueError12 = [9.53e-04, 2.00e-04, 4.28e-05, 4.64e-05, 1.63e-04, 1.47e-04, 1.44e-04]
# Residual errors: ||A_loc u - f||
resError12 = [8.90e-11, 3.96e-10, 1.09e-09, 2.66e-09, 4.64e-09, 9.35e-09, 1.28e-08]

N18 = np.array([32, 48, 64, 80]) ** 3
trueError18 = [8.21e-04, 2.50e-04, 5.49e-04, 2.78e-03]
resError18 = [2.05e-09, 1.18e-08, 4.14e-08, 5.50e-08]

plt.semilogy(N12, trueError12, 'b-')
plt.semilogy(N18, trueError18, 'r-')
plt.semilogy(N12, resError12, 'b--')
plt.semilogy(N18, resError18, 'r--')
plt.title("Relative errors for Helmholtz, 10 ppw, fixed domain size")
plt.xlabel("N = n^3")
plt.ylabel("Relative Error")
plt.legend(["True, p=12", "True, p=18", "Residual, p=12", "Residual, p=18"])
plt.show()