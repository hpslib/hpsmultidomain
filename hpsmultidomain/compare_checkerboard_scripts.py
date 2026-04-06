import numpy as np
import torch
import pickle

import matplotlib.pyplot as plt

p_list = [9, 11, 13, 15, 17, 19, 21]
nboxes_list = [8, 24, 72, 216]

solutions = []

for i, p in enumerate(p_list):
    n_list = [(p-2) * _ for _ in nboxes_list]
    solutions_p = []
    for j, n in enumerate(n_list):
        print(f"Running p={p}, n={n}...")

        file_path = f"data-convection-checkerboard/checkerboard-p{p}-n{n}.pkl"

        # Open the file in read-binary mode ('rb')
        with open(file_path, 'rb') as file:
            # Load the data from the file
            data = pickle.load(file)
            solutions_p.append(data["sol"])
    solutions.append(solutions_p)

print("All runs complete.")

#print([(p**2 * (n // (p-2))**2, sol.shape) for (p, n, sol) in solutions])

base = 8 # number of boxes alongaxis for lowest res runs

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

nboxes_list = [8, 24, 72, 216]

rel_errors = torch.zeros(len(p_list), len(nboxes_list))

for i, p in enumerate(p_list):
    for j, nboxes in enumerate(nboxes_list):
        mp = midpoints[i, j, :]
        rel_error = torch.linalg.norm(mp - hi_res_midpoints) / torch.linalg.norm(hi_res_midpoints)
        print("p = " + str(p) + ", nboxes = " + str(nboxes))
        print(rel_error)
        rel_errors[i, j] = rel_error.item()


print(rel_errors)

def fit_slope(h_vals, err_vals, n_tail=4):
    """Fit slope to the last n_tail points in log-log space."""
    log_h = np.log(h_vals[-n_tail:])
    log_e = np.log(err_vals[-n_tail:])
    slope, intercept = np.polyfit(log_h, log_e, 1)
    return slope, intercept

for i, p in enumerate(p_list):
    if i == len(p_list)-1:
        r, intercept = fit_slope([_**(-1) for _ in nboxes_list[:-1]], rel_errors[i,:-1], n_tail=3)
        plt.loglog(nboxes_list[:-1], rel_errors[i,:-1], label="p=" + str(p))
        plt.loglog(nboxes_list, np.exp(intercept) * nboxes_list**(-r), 'k--', alpha=0.5)
    else:
        r, intercept = fit_slope([_**(-1) for _ in nboxes_list], rel_errors[i], n_tail=3)
        plt.loglog(nboxes_list, rel_errors[i], label="p=" + str(p))
        plt.loglog(nboxes_list, np.exp(intercept) * nboxes_list**(-r), 'k--', alpha=0.5)
    print("p, r = ", p, r)

plt.title("Self-convergence of checkerboard convection field")
plt.xlabel("h^{-1}")
plt.ylabel("l^2 error relative to p=21, h = 1/216")
plt.legend()
plt.savefig("checkerboard_convergence.pdf")
plt.show()