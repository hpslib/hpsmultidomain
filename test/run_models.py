#import numpy as np

#import matplotlib.pyplot as plt

from test_3d import run_test_via_argparse

# First we'll run a suite of Poisson problems for different n and p:
#p_list = [10, 12, 14, 16, 18, 20, 22]#, 24, 26]
#p_list = [30]
#p_list = [10, 12, 14, 16, 18]
#p_list = [14]

p_list = [8, 10,12,14,16,18,20,22]
#p_list = [18,20,22] 
#p_list = [10, 12, 14]
#p_list = [6, 8, 10, 12, 14]

#p_list = [9,11,13,15,17,19,21]

domain = "square"
pde = "bfield_constant"
bc = "free_space"
ppw = 10
kh = None
delta_t = None
store_sol = False

output_path = "output/helmholtz_10ppw_no_sparse_mat_0510"

for p in p_list:
    n_list = list(range(5*(p-2), 260, p-2))

    #n_list = [152]
    #n_list = list(range(156, 210, p-2))
    #n_list = list(range(2*(p-2), 10*(p-2)+1, p-2))
    
    for n in n_list:
        print("\nRunning " + str(p) + ", " + str(n) + "\n")
        test_file = output_path + "/test_results_p_" + str(p) + "_n_" + str(n) + ".pkl"
        run_test_via_argparse(domain, pde, bc, n, p, ppw=ppw, kh=kh, delta_t=delta_t, box_xlim=1.0, box_ylim=1.0, periodic_bc=False, components=False, store_sol=store_sol, solver="mumps", assembly_type="reduced_gpu", pickle_loc=test_file)
