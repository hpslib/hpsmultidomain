#import numpy as np

#import matplotlib.pyplot as plt

from test_3d import run_test_via_argparse

# First we'll run a suite of Poisson problems for different n and p:
p_list = [8] #[6, 8, 10, 12, 14, 18, 22, 30]

domain = "square"
pde = "poisson"
bc = "log_dist"

for p in p_list:
    n_list = list(range(2*(p-2), min(100,11*(p-2)), p-2))

    for n in n_list:
        print(str(p) + ", " + str(n) + " done")
        test_file = "test_results/test_results_p_" + str(p) + "_n_" + str(n) + ".pkl"
        run_test_via_argparse(domain, pde, bc, n, p, box_xlim=1.0, box_ylim=1.0, periodic_bc=False, ppw=1, components=False, pickle_loc=test_file)
