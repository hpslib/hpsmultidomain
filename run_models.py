#import numpy as np

#import matplotlib.pyplot as plt

from test_3d import run_test_via_argparse

# First we'll run a suite of Poisson problems for different n and p:
#p_list = [10, 12, 14, 18, 22]
p_list = [24]

domain = "square"
pde = "bfield_constant"
bc = "free_space"
ppw = 10

for p in p_list:
    n_list = list(range(2*(p-2), min(100, 100*(p-2)), p-2))
    #n_list = list(range(76, 100, 2))
    
    for n in n_list:
        print(str(p) + ", " + str(n) + " done")
        test_file = "test_helmholtz/test_results_p_" + str(p) + "_n_" + str(n) + ".pkl"
        run_test_via_argparse(domain, pde, bc, n, p, ppw=ppw, box_xlim=1.0, box_ylim=1.0, periodic_bc=False, components=False, solver="mumps", assembly_type="reduced_gpu", pickle_loc=test_file)
