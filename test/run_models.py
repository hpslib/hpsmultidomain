#import numpy as np

#import matplotlib.pyplot as plt

from test_3d import run_test_via_argparse

# First we'll run a suite of Poisson problems for different n and p:
p_list = [10, 12, 14, 16, 18, 20, 22]#, 24, 26]
#p_list = [30]
#p_list = [10, 12, 14, 16, 18]
#p_list = [22]

#p_list = [10,12,14,16,18,20,22]
#p_list = [18,20,22] 
#p_list = [20,22]
#p_list = [22]

#p_list = [9,11,13,15,17,19,21]

domain = "curved"
pde = "bfield_constant"
bc = "ones"
ppw = None
kh = 16
delta_t = None

output_path = "output/curved_gpu_bc1_kh16_0207"

for p in p_list:
    n_list = list(range(2*(p-2), 200, p-2))
    
    #n_list = [2*(p-2), 3*(p-2), 4*(p-2), 5*(p-2), 6*(p-2), 7*(p-2), 8*(p-2), 9*(p-2)]
    n_list = list(range(2*(p-2), 10*(p-2)+1, p-2))
    
    for n in n_list:
        print("\nRunning " + str(p) + ", " + str(n) + "\n")
        test_file = output_path + "/test_results_p_" + str(p) + "_n_" + str(n) + ".pkl"
        run_test_via_argparse(domain, pde, bc, n, p, kh=kh, delta_t=delta_t, box_xlim=1.0, box_ylim=1.0, periodic_bc=False, components=False, store_sol=True, solver="mumps", assembly_type="reduced_gpu", pickle_loc=test_file)
