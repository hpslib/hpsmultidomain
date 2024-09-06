#import numpy as np

#import matplotlib.pyplot as plt

from test_3d import run_test_via_argparse

# First we'll run a suite of Poisson problems for different n and p:
#p_list = [10, 12, 14, 16, 18, 20, 22]#, 24, 26]
#p_list = [30]
p_list = [10, 12, 14, 16, 18]

domain = "square"
pde = "mixed"
bc = "log_dist"
ppw = 0
kh = 0

for p in p_list:
    #n_list = list(range(2*(p-2), 100, p-2))
    n_list = [2*(p-2), 3*(p-2), 4*(p-2), 5*(p-2)]
    
    for n in n_list:
        print(str(p) + ", " + str(n) + " done")
        test_file = "test_mixed_improved/test_results_p_" + str(p) + "_n_" + str(n) + ".pkl"
        run_test_via_argparse(domain, pde, bc, n, p, ppw=ppw, kh=kh, box_xlim=1.0, box_ylim=1.0, periodic_bc=False, components=False, solver="mumps", assembly_type="reduced_gpu", pickle_loc=test_file)
