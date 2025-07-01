#import numpy as np

#import matplotlib.pyplot as plt

from test_3d import run_test_via_argparse

# Can set different parameters for a longer test here:
# ranges of p, domain shape, pde and bc of choice,
# and more (depending on the PDE you wish to solve)
p_list = [10,12,14,16,18,20,22]

domain = "square"
pde = "poisson"
bc = "log_dist"
ppw = None
kh = None
delta_t = None

output_path = "output/poisson_test/" # Folder needs to exist

for p in p_list:
    n_list = list(range(2*(p-2), 200, p-2))
    
    for n in n_list:
        print("\nRunning " + str(p) + ", " + str(n) + "\n")
        test_file = output_path + "/test_results_p_" + str(p) + "_n_" + str(n) + ".pkl"
        run_test_via_argparse(domain, pde, bc, n, p, ppw=ppw, kh=kh, delta_t=delta_t, num_timesteps=10, box_xlim=1.0, box_ylim=1.0, periodic_bc=False, components=False, store_sol=True, solver="mumps", assembly_type="reduced_gpu", pickle_loc=test_file)
