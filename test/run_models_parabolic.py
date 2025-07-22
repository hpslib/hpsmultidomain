#import numpy as np

#import matplotlib.pyplot as plt

from test_3d import run_test_via_argparse

p_list = [10, 12, 14, 16]

domain = "square"
pde = "convection_diffusion"
bc = "convection_diffusion"
ppw = None
kh = None

output_path = "output/convection_diffusion"

for p in p_list:
    
    delta_t_list = [-1, -2, -3]

    for i in delta_t_list:
        dt = 10 ** i
        num_timesteps = 5 * (10 ** -i)
        n_list = [2*(p-2), 3*(p-2), 4*(p-2), 5*(p-2)]
        for n in n_list:
            print("\nRunning " + str(p) + ", " + str(dt) + "\n")
            test_file = output_path + "/test_results_p_" + str(p) + "_dt_" + str(dt) + "_n_" + str(n) + ".pkl"
            run_test_via_argparse(domain, pde, bc, n, p, ppw=ppw, kh=kh, delta_t=dt, box_xlim=1.0, box_ylim=1.0, periodic_bc=False, components=False, solver="mumps", assembly_type="reduced_gpu", pickle_loc=test_file)
    