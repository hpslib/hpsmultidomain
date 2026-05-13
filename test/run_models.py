#import numpy as np

#import matplotlib.pyplot as plt

from test_3d import run_test_case

# Can set different parameters for a longer test here:
# ranges of p, domain shape, pde and bc of choice,
# and more (depending on the PDE you wish to solve)
#p_list = [10,12,14,16,18,20,22]
p_list = [10,14,18,22]

domain = "square"
pde = "bfield_constant"
bc = "free_space"
ppw = None
kh = 40
delta_t = None
disable_cuda = False

no_condense=False

output_path = "output/condensed_test_helmholtz_kh40" # Folder needs to exist

for p in p_list:
    n_list = list(range(2*(p-2), 100, (p-2)))
    
    for n in n_list:
        print("\nRunning " + str(p) + ", " + str(n) + "\n")
        test_file = output_path + "/test_results_p_" + str(p) + "_n_" + str(n) + ".pkl"
        run_test_case(domain=domain, pde=pde, bc=bc, n=n, p=p, ppw=ppw, kh=kh, delta_t=delta_t, num_timesteps=None, box_xlim=1.0, box_ylim=1.0, periodic_bc=False, components=False, store_sol=False, solver="mumps", disable_cuda=disable_cuda, sparse_assembly="reduced_gpu", no_condense=no_condense, pickle=test_file)
