import numpy as np
import torch
import pickle

import matplotlib.pyplot as plt

from relative_error_with_interpolation import relative_L2_error

#p_list = [9, 11, 13, 15, 17, 19, 21]
#p_list = [4, 6, 8, 10, 12, 14, 16, 18] #, 20]
nboxes_list = [2, 4, 8, 16, 32, 64, 128] #, 256]
#nboxes_list = [2, 6, 18, 54, 162]

p_list = [5, 7, 9, 11, 13, 15, 17] #, 19]
#nboxes_list = [2, 6, 18, 54, 162]
#nboxes_list = [8, 24, 72, 216]

base = nboxes_list[0] # number of boxes along axis for lowest res runs

solutions = []

kh        = 50
b         = 0
checkered = False
shifted   = False

directory = "data-helmholtz-"

if checkered:
    directory = directory + "checkerboard-"
    if shifted:
        directory = directory + "shifted-"

directory = directory + "kh" + str(kh) + "-b" + str(b)

for i, p in enumerate(p_list):
    n_list = [(p-2) * _ for _ in nboxes_list]
    solutions_p = []
    for j, n in enumerate(n_list):
        print(f"Running p={p}, n={n}...")

        file_path = directory + f"/helmholtz-p{p}-n{n}.pkl"

        # Open the file in read-binary mode ('rb')
        with open(file_path, 'rb') as file:
            # Load the data from the file
            data = pickle.load(file)
            solutions_p.append(data["sol"])
    solutions.append(solutions_p)

print("All runs complete.")

#print([(p**2 * (n // (p-2))**2, sol.shape) for (p, n, sol) in solutions])
"""
midpoints = torch.zeros(len(p_list), len(nboxes_list), base**2)

for i, p in enumerate(p_list):
    for j, nboxes in enumerate(nboxes_list):
        # How many x more boxes there are to a side than the default base
        sol = solutions[i][j]
        zoom = nboxes // base
        if zoom == 1:
            sol_reshaped = sol
        else:
            sol_reshaped = torch.reshape(sol, (-1, zoom, nboxes*p**2))
            sol_reshaped = sol_reshaped[:, (zoom//2), :]
            sol_reshaped = torch.reshape(sol_reshaped, (base, -1, zoom, p**2))
            sol_reshaped = sol_reshaped[:, :, (zoom//2), :]
            sol_reshaped = torch.flatten(sol_reshaped).unsqueeze(-1)

        midpoint = (p**2) // 2
        #midpoint_tuple = (p, nboxes, sol_reshaped[midpoint::p**2,:])
        #midpoints.append(midpoint_tuple)
        midpoints[i, j, :] = sol_reshaped[midpoint::p**2,:].squeeze(-1)

#hi_res = midpoints.pop(-1)
hi_res_midpoints = midpoints[-1, -1, :]

#print(hi_res)

rel_errors = torch.zeros(len(p_list), len(nboxes_list))

for i, p in enumerate(p_list):
    for j, nboxes in enumerate(nboxes_list):
        mp = midpoints[i, j, :]
        rel_error = torch.linalg.norm(mp - midpoints[i, -1, :]) / torch.linalg.norm(midpoints[i, -1, :])
        print("p = " + str(p) + ", nboxes = " + str(nboxes))
        print(rel_error)
        rel_errors[i, j] = rel_error.item()
"""


rel_errors = torch.zeros(len(p_list), len(nboxes_list))

for i, p in enumerate(p_list):
    for j, nboxes in enumerate(nboxes_list[:-1]):
        rel_error = relative_L2_error(solutions[i][j], solutions[i][-1], nboxes, nboxes_list[-1], p)
        print("p = " + str(p) + ", nboxes = " + str(nboxes) + ", nobxes_fine = " + str(nboxes_list[-1]))
        print(rel_error)
        rel_errors[i, j] = rel_error

print(rel_errors)

def fit_slope(h_vals, err_vals, n_tail=4):
    """Fit slope to the last n_tail points in log-log space."""
    log_h = np.log(h_vals[-n_tail:])
    log_e = np.log(err_vals[-n_tail:])
    slope, intercept = np.polyfit(log_h, log_e, 1)
    return slope, intercept

for i, p in enumerate(p_list):
    r, intercept = fit_slope([_**(-1) for _ in nboxes_list[:-1]], rel_errors[i,:-1], n_tail=3)
    plt.loglog(nboxes_list[:-1], rel_errors[i,:-1], label=f"p={p}, r={r:.2f}")
    #plt.loglog(nboxes_list[:-1], np.exp(intercept) * nboxes_list[:-1]**(-r), 'k--', alpha=0.5)
    print("p, r = ", p, r)

title_start = "Self-convergence of field, kh = "
if checkered:
    title_start = "Self-convergence of checkerboard field, kh = "
    if shifted:
        title_start = "Self-convergence of shifted checkerboard field, kh = "

plt.title(title_start + str(kh) + ", b = " + str(b))
plt.xlabel("h^{-1}")
plt.ylabel("l^2 error relative to h = 1/" + str(nboxes_list[-1]))
plt.legend()
plt.savefig(directory + "/convection-convergence.png")
plt.show()