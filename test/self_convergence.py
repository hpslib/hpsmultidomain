from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import torch

torch.set_printoptions(precision=16)

mypath      = "gpu_output/gravity_kh_12_0116/"
plotpath    = "plots/convection_diffusion/"
total_title = "Convection Diffusion with 10 timesteps:\n"

#p_list = [10]#, 12, 14, 16]
#p_list = [8, 10, 12, 14, 18, 22, 30]


p_list = np.array([9, 11, 13, 15, 17, 19, 21])

box_list = [2, 3, 4, 5, 6, 7, 8, 9]
p_lists = []

def make_p_results(mypath, p_list, box_list):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    #print(onlyfiles)
    sol_errors = []
    for boxes in box_list:
        box_indices = (p_list - 2) * boxes
        #print(box_indices)
        box_files = [] #[_ for _ in onlyfiles if "_n_" + str(p) + "_" in _]
        for i in range(len(box_indices)):
            box_files = box_files + [_ for _ in onlyfiles if "_n_" + str(box_indices[i]) in _ and "_p_" + str(p_list[i]) + "_" in _]

        p_indices = []
        computed_sols = []
        
        for filename in box_files:
            with open(mypath + "/" + filename, 'rb') as f:
                x = pickle.load(f)
                p_indices.append(x["p"])
                computed_sols.append(x["computed_sol"])

        n_indices = [_.shape[0] for _ in computed_sols]
        p_and_n = list(zip(p_indices, n_indices))

        # Find the central point in each box:
        #print([_ / (boxes**3) for _ in n_indices])
        #print(np.array(p_indices) ** 3)
        midpoints = [int((_ ** 3) / 2) for _ in p_indices]
        #print(box_files)
        #print(p_indices)
        #print(midpoints)

        # Now we want to take the midpoints of each box and compare:
        sol_reshaped = []
        for p, midpoint, sol, i in zip(p_indices, midpoints, computed_sols, list(range(len(p_indices)))):
            sol_reshaped.append(sol[midpoint:-1:p**3,0])
        
        sol_reshaped = torch.vstack(sol_reshaped)

        sol_reshaped_error = sol_reshaped - sol_reshaped[-1]
        sol_reshaped_error = torch.linalg.norm(sol_reshaped_error, dim=1) / torch.linalg.norm(sol_reshaped[-1])
        #print(sol_reshaped)

        sol_errors.append(sol_reshaped_error)
        p_lists.append(p_indices)

        """    
        p_result = dict(n=n)
        
        p_result = pd.DataFrame.from_dict(p_result)

        if pd.api.types.is_object_dtype(p_result['n']):
            p_result['n'] = p_result['n'].str[0]

        p_result.set_index('n', inplace=True)
        p_result.sort_index(inplace=True)
        p_results.append(pd.DataFrame.from_dict(p_result))
        """

    return sol_errors, p_lists, computed_sols, sol_reshaped

convergences, p_lists, computed_sols, sol_reshaped = make_p_results(mypath, p_list, box_list)

print(p_lists)

figsize = (8,6)
plt.rcParams['figure.figsize'] = [figsize[0],figsize[1]]
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'serif','size':14})
plt.rc('text.latex',preamble=r'\usepackage{amsfonts,bm}')

#p_list = p_list[:-1]
convergences = [_[:-1] for _ in convergences]

for box, convergence, p_indices in zip(box_list, convergences, p_lists):
    p_indices = p_indices[:-1]
    N_list = (np.array(p_indices)**3) * box**3
    #if convergence[-1] < 1e-16:
    #    N_list = N_list[:-1]
    #    convergence = convergence[:-1]
    plt.loglog(N_list, convergence)

legend = ["h = 1/" + str(_) for _ in box_list]

plt.legend(legend)
plt.title("p-Refinement of Gravity Helmholtz Equation")
plt.xlabel("N")
plt.ylabel("Relative Errpr")
plt.savefig("gravity_convergence_N.pdf")
plt.show()

for box, convergence, p_indices in zip(box_list, convergences, p_lists):
    p_indices = p_indices[:-1]
    plt.semilogy(p_indices, convergence)

plt.legend(legend)
plt.title("p-Refinement of Gravity Helmholtz Equation")
plt.xlabel("p")
plt.ylabel("Relative Error")
plt.savefig("gravity_convergence_p.pdf")
plt.show()


#print([a[int(b**3/2),0] for a, b in zip(computed_sols, p_lists[-1])])

#print(sol_reshaped)

#midpoint = int(9**3 / 2)
#print(midpoint)
#print(computed_sols[0][midpoint:-1:9**3,0])