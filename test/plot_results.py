from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

mypath      = "output/convection_diffusion/"
plotpath    = "plots/convection_diffusion/"
total_title = "Convection Diffusion with 10 timesteps:\n"

p_list = [10]#, 12, 14, 16]
#p_list = [8, 10, 12, 14, 18, 22, 30]


#p_list = [10, 12, 14, 16, 18, 20, 22]
#p_list = [6, 8, 10, 12]#, 14]

def make_p_results(mypath, p_list):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    p_results = []
    for p in p_list:
        p_files = [_ for _ in onlyfiles if "_p_" + str(p) + "_" in _]
        n = []
        delta_t = []
        toc_invert = []
        toc_build_dtn = []
        toc_leaf_solve = []
        sparse_solve_res = []
        true_res = []
        leaf_res = []

        #forward_bdry_error = []
        #reverse_bdry_error = []

        # Interpolation stuff
        GtC_error = []
        CtG_error = []
        GtC_cond = []
        CtG_cond = []
        neumann_tensor_error = []
        neumann_sparse_error = []
        dtn_cond = []
        for filename in p_files:
            with open(mypath + "/" + filename, 'rb') as f:
                x = pickle.load(f)
                n.append(x["n"])
                #delta_t.append(x["delta_t"])
                toc_invert.append(x["toc_build_blackbox"])
                toc_build_dtn.append(x["toc_assembly"])
                toc_leaf_solve.append(x["toc_solve_petsc"])
                sparse_solve_res.append(x["res_solve_petsc"])
                true_res.append(x["trueres_solve_petsc"])
                leaf_res.append(x["resloc_hps_solve_petsc"])
                #forward_bdry_error.append(x["forward_bdry_error"])
                #reverse_bdry_error.append(x["reverse_bdry_error"])


                # Interpolation stuff
                """
                GtC_error.append(x["GtC_error"])
                CtG_error.append(x["GtC_error"])
                GtC_cond.append(x["GtC_cond"])
                CtG_cond.append(x["CtG_cond"])
                neumann_tensor_error.append(x["neumann_tensor_error"])
                neumann_sparse_error.append(x["neumann_sparse_error"])
                dtn_cond.append(x["dtn_cond"])"""
            
        p_result = dict(n=n, toc_invert=toc_invert, toc_build_dtn=toc_build_dtn, toc_leaf_solve=toc_leaf_solve,
                        sparse_solve_res=sparse_solve_res, true_res=true_res, leaf_res=leaf_res)
                        #delta_t=delta_t)
                        #forward_bdry_error=forward_bdry_error, reverse_bdry_error=reverse_bdry_error,
                        #GtC_error=GtC_error, CtG_error=CtG_error, GtC_cond=GtC_cond,CtG_cond=CtG_cond, #INTERPOLATION
                        #neumann_tensor_error=neumann_tensor_error, neumann_sparse_error=neumann_sparse_error, dtn_cond=dtn_cond)
        
        p_result = pd.DataFrame.from_dict(p_result)

        if pd.api.types.is_object_dtype(p_result['n']):
            p_result['n'] = p_result['n'].str[0]
            
        p_result.set_index('n', inplace=True)
        p_result.sort_index(inplace=True)
        p_results.append(pd.DataFrame.from_dict(p_result))

    return p_results

def make_plot(p_list, p_results, field, title, xlabel, ylabel, type="plot"):
    legend = []
    for i in range(len(p_list)):
        if type=="plot":
            plt.plot(p_results[i].index, p_results[i][field])
        if type=="semilogy":
            plt.semilogy(p_results[i].index, p_results[i][field])
        if type=="loglog":
            plt.loglog(p_results[i].index, p_results[i][field])
        legend.append("p = " + str(p_list[i]))

    plt.title(title)
    plt.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(plotpath + field + ".png")
    plt.show()

"""
legend = []
for i in range(len(p_list)):
    plt.loglog(p_results[i].index**3, p_results[i]["toc_invert"])
    legend.append("p = " + str(p_list[i]))

plt.loglog(p_results[i].index**3, 1.5*p_results[i].index**3)
legend.append("N = N^3/2")

plt.title("Poisson: time to factorize sparse matrix (log plot)")
plt.legend(legend)
plt.xlabel("N")
plt.ylabel("seconds")
plt.savefig("plots_poisson/log_toc_invert.png")
plt.show()
"""

p_results = make_p_results(mypath, p_list)


#make_plot(p_list, p_results, "toc_invert", total_title + "time to factorize sparse matrix", "N", "seconds")
#make_plot(p_list, p_results, "toc_build_dtn", total_title + "time to assemble batched DtN maps", "N", "seconds")
#make_plot(p_list, p_results, "toc_leaf_solve", total_title + "time to solve batched leaf operators", "N", "seconds")



#make_plot(p_list, p_results, "sparse_solve_res", total_title + "residual of sparse system solve for boundaries", "N", "relative error", type="loglog")
#make_plot(p_list, p_results, "true_res", total_title + "residual of total result", "delta T", "relative error", type="loglog")
#make_plot(p_list, p_results, "leaf_res", total_title + "residual of leaf computations", "N", "relative error", type="loglog")
"""
make_plot(p_list, p_results, "forward_bdry_error", total_title + "error when applying box bdries to sparse mat", "N", "relative error", type="loglog")
make_plot(p_list, p_results, "reverse_bdry_error", total_title + "residual of box boundaries after sparse solve", "N", "relative error", type="loglog")

# Interpolation:
make_plot(p_list, p_results, "GtC_error", total_title + "error of leaf Gauss to Cheb interpolation", "N", "relative error", type="loglog")
make_plot(p_list, p_results, "CtG_error", total_title + "error of leaf Cheb to Gauss interpolation", "N", "relative error", type="loglog")
make_plot(p_list, p_results, "GtC_cond", total_title + "cond of leaf Gauss to Cheb interpolation", "N", "condition #", type="loglog")
make_plot(p_list, p_results, "CtG_cond", total_title + "cond of leaf Cheb to Gauss interpolation", "N", "condition #", type="loglog")
make_plot(p_list, p_results, "neumann_tensor_error", total_title + "applying DtNs to Gaussian Dirichlet data", "N", "relative error", type="loglog")
make_plot(p_list, p_results, "neumann_sparse_error", total_title + "applying sparse mat (from DtN) to Gaussian Dirichlet data", "N", "relative error", type="loglog")
make_plot(p_list, p_results, "dtn_cond", total_title + "condtion # of a DtN map", "N", "condition #", type="loglog")
"""

# Here we'll create a figure plot:
def plot_paired_results(p_list1, p_list2, path1, path2, subtitle1, subtitle2, title, filename):
    figsize = (16,6)
    p_results_poisson = make_p_results(path1, p_list1)
    p_results_helmholtz = make_p_results(path2, p_list2)

    plt.rcParams['figure.figsize'] = [figsize[0],figsize[1]]
    plt.rc('text',usetex=True)
    plt.rc('font',**{'family':'serif','size':14})
    plt.rc('text.latex',preamble=r'\usepackage{amsfonts,bm}')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title)
    for i in range(len(p_list1)):
        ax1.loglog(p_results_poisson[i].index**3, p_results_poisson[i]["true_res"])
    ax1.legend(["p = " + str(_) for _ in p_list1])
    ax1.set_xlabel("N")
    ax1.set_ylabel("Relative Error")
    ax1.set_title(subtitle1)

    for i in range(len(p_list2)):
        ax2.loglog(p_results_helmholtz[i].index**3, p_results_helmholtz[i]["true_res"])
    ax2.legend(["p = " + str(_) for _ in p_list2])
    ax2.set_xlabel("N")
    ax2.set_title(subtitle2)

    plt.savefig(filename)
    plt.show()

p_list_poisson   = [6,8,10,12,14]
p_list_helmholtz = [10,12,14,16,18,20,22]

path_poisson   = "output/poisson/"
path_helmholtz = "output/test_helmholtz/"

subtitle1 = "Poisson Equation"
subtitle2 = "Helmholtz Equation, 10 PPW"
title     = "Relative Errors for Poisson and Helmholtz Equation"
filename  = "poisson_helmholtz_accuracy.pdf"
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, filename)

path_kh16 = "gpu_output/helmholtz_gpu_kh16_1208/"
path_kh30 = "gpu_output/helmholtz_gpu_kh30_1208/"
subtitle1 = "K = 16"
subtitle2 = "K = 30"
title     = "Relative Errors for Helmholtz Equation with Fixed K"
filename  = "helmholtz_kh_accuracy_gpu.pdf"
plot_paired_results(p_list_helmholtz, p_list_helmholtz, path_kh16, path_kh30, subtitle1, subtitle2, title, filename)