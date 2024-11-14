#import numpy as np

#import matplotlib.pyplot as plt

from test_3d import run_test_via_argparse

# First we'll run a suite of Poisson problems for different n and p:
#p_list = [10, 12, 14, 16, 18, 20, 22]#, 24, 26]
#p_list = [30]
#p_list = [10, 12, 14, 16, 18]
p_list = [22]

domain = "annulus"
pde = "bfield_constant"
bc = "free_space"
ppw = 10
kh = 0
delta_t = None

output_path = "output/annulus_10ppw"

for p in p_list:
    #n_list = list(range(2*(p-2), 85, p-2))
    
    n_list = [2*(p-2), 3*(p-2), 4*(p-2)]
    
    for n in n_list:
        print("\nRunning " + str(p) + ", " + str(n) + "\n")
        test_file = output_path + "/test_results_p_" + str(p) + "_n_" + str(n) + ".pkl"
        run_test_via_argparse(domain, pde, bc, n, p, ppw=ppw, delta_t=delta_t, box_xlim=1.0, box_ylim=1.0, periodic_bc=False, components=False, solver="mumps", assembly_type="reduced_gpu", pickle_loc=test_file)
    
    """
    delta_t_list = [-1, -2, -3, -4, -5]

    for i in delta_t_list:
        dt = 10 ** i
        n  = 2*(p-2)
        print("\nRunning " + str(p) + ", " + str(dt) + "\n")
        test_file = output_path + "/test_results_p_" + str(p) + "_dt_" + str(dt) + ".pkl"
        run_test_via_argparse(domain, pde, bc, n, p, ppw=ppw, kh=kh, delta_t=dt, box_xlim=1.0, box_ylim=1.0, periodic_bc=False, components=False, solver="mumps", assembly_type="reduced_gpu", pickle_loc=test_file)
    """