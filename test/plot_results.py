from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

mypath      = "output/convection_diffusion"
plotpath    = "plots/convection_diffusion/"
total_title = "Convection-diffusion:\n"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#p_list = [8, 10, 12, 14, 16, 18, 20]
#p_list = [8, 10, 12, 14, 18, 22, 30]
p_list = [10, 12, 14, 16, 18]

p_results = []

for p in p_list:
    p_files = [_ for _ in onlyfiles if "_p_" + str(p) + "_" in _]
    n = []
    toc_invert = []
    toc_build_dtn = []
    toc_leaf_solve = []
    sparse_solve_res = []
    true_res = []
    leaf_res = []

    forward_bdry_error = []
    reverse_bdry_error = []

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
            toc_invert.append(x["toc_build_blackbox"])
            toc_build_dtn.append(x["toc_assembly"])
            toc_leaf_solve.append(x["toc_solve_petsc"])
            sparse_solve_res.append(x["res_solve_petsc"].item())
            true_res.append(x["trueres_solve_petsc"])
            leaf_res.append(x["resloc_hps_solve_petsc"])
            forward_bdry_error.append(x["forward_bdry_error"])
            reverse_bdry_error.append(x["reverse_bdry_error"])


            # Interpolation stuff
            GtC_error.append(x["GtC_error"])
            CtG_error.append(x["GtC_error"])
            GtC_cond.append(x["GtC_cond"])
            CtG_cond.append(x["CtG_cond"])
            neumann_tensor_error.append(x["neumann_tensor_error"])
            neumann_sparse_error.append(x["neumann_sparse_error"])
            dtn_cond.append(x["dtn_cond"])
        
    p_result = dict(n=n, toc_invert=toc_invert, toc_build_dtn=toc_build_dtn, toc_leaf_solve=toc_leaf_solve,
                    sparse_solve_res=sparse_solve_res, true_res=true_res, leaf_res=leaf_res,
                    forward_bdry_error=forward_bdry_error, reverse_bdry_error=reverse_bdry_error,
                    GtC_error=GtC_error, CtG_error=CtG_error, GtC_cond=GtC_cond,CtG_cond=CtG_cond, #INTERPOLATION
                    neumann_tensor_error=neumann_tensor_error, neumann_sparse_error=neumann_sparse_error, dtn_cond=dtn_cond)
    
    p_result = pd.DataFrame.from_dict(p_result)
    p_result.set_index('n', inplace=True)
    p_result.sort_index(inplace=True)
    p_results.append(pd.DataFrame.from_dict(p_result))

def make_plot(field, title, xlabel, ylabel, type="plot"):
    legend = []
    for i in range(len(p_list)):
        if type=="plot":
            plt.plot(p_results[i].index**3, p_results[i][field])
        if type=="semilogy":
            plt.semilogy(p_results[i].index**3, p_results[i][field])
        if type=="loglog":
            plt.loglog(p_results[i].index**3, p_results[i][field])
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


#make_plot("toc_invert", total_title + "time to factorize sparse matrix", "N", "seconds")
#make_plot("toc_build_dtn", total_title + "time to assemble batched DtN maps", "N", "seconds")
#make_plot("toc_leaf_solve", total_title + "time to solve batched leaf operators", "N", "seconds")

make_plot("sparse_solve_res", total_title + "residual of sparse system solve for boundaries", "N", "relative error", type="loglog")
make_plot("true_res", total_title + "residual of total result", "N", "relative error", type="loglog")
make_plot("leaf_res", total_title + "residual of leaf computations", "N", "relative error", type="loglog")

make_plot("forward_bdry_error", total_title + "error when applying box bdries to sparse mat", "N", "relative error", type="loglog")
make_plot("reverse_bdry_error", total_title + "residual of box boundaries after sparse solve", "N", "relative error", type="loglog")

# Interpolation:
make_plot("GtC_error", total_title + "error of leaf Gauss to Cheb interpolation", "N", "relative error", type="loglog")
make_plot("CtG_error", total_title + "error of leaf Cheb to Gauss interpolation", "N", "relative error", type="loglog")
make_plot("GtC_cond", total_title + "cond of leaf Gauss to Cheb interpolation", "N", "condition #", type="loglog")
make_plot("CtG_cond", total_title + "cond of leaf Cheb to Gauss interpolation", "N", "condition #", type="loglog")
make_plot("neumann_tensor_error", total_title + "applying DtNs to Gaussian Dirichlet data", "N", "relative error", type="loglog")
make_plot("neumann_sparse_error", total_title + "applying sparse mat (from DtN) to Gaussian Dirichlet data", "N", "relative error", type="loglog")
make_plot("dtn_cond", total_title + "condtion # of a DtN map", "N", "condition #", type="loglog")
