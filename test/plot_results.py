from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

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
        toc_system_solve = []
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
                toc_system_solve.append(x["toc_solve_petsc"]) # Old
                toc_leaf_solve.append(x["toc_solve_leaf"])
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
                        sparse_solve_res=sparse_solve_res, true_res=true_res, leaf_res=leaf_res,
                        toc_system_solve=toc_system_solve)
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



# Here we'll create a figure plot:
def plot_paired_results(p_list1, p_list2, path1, path2, subtitle1, subtitle2, title, ylabel, data_col, filename, type="loglog"):
    figsize = (16,6)
    p_results_poisson = make_p_results(path1, p_list1)
    p_results_helmholtz = make_p_results(path2, p_list2)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    plt.rcParams['figure.figsize'] = [figsize[0],figsize[1]]
    plt.rc('text',usetex=True)
    plt.rc('font',**{'family':'serif','size':18})
    plt.rc('text.latex',preamble=r'\usepackage{amsfonts,bm}')

    #fig.suptitle(title)
    for i in range(len(p_list1)):
        if type == "loglog":
            ax1.loglog(p_results_poisson[i].index**3, p_results_poisson[i][data_col], '.-')
        elif type == "plot":
            ax1.plot(p_results_poisson[i].index**3, p_results_poisson[i][data_col], '.-')
        elif type == "semilogy":
            ax1.semilogy(p_results_poisson[i].index**3, p_results_poisson[i][data_col], '.-')
        else:
            raise ValueError("Type needs to be loglog, plot, or semilogy")
    #ax1.loglog(p_results_poisson[-1].index**3, (p_results_poisson[-1].index)**6)
    ax1.legend(["$p$ = " + str(_) for _ in p_list1])# + ["trend for N**{1.5} scaling"])
    ax1.set_xlabel("$N$")
    ax1.set_ylabel(ylabel)
    #ax1.set_title(subtitle1)
    ax1.grid(True)

    for i in range(len(p_list2)):
        #if i == 0 and type == "loglog":
        #    ax2.loglog(p_results_helmholtz[i].index[1:]**3, p_results_helmholtz[i][data_col][1:])
        if type == "loglog":
            ax2.loglog(p_results_helmholtz[i].index**3, p_results_helmholtz[i][data_col], '.-')
        elif type == "plot":
            ax2.plot(p_results_helmholtz[i].index**3, p_results_helmholtz[i][data_col], '.-')
        elif type == "semilogy":
            ax2.semilogy(p_results_helmholtz[i].index**3, p_results_helmholtz[i][data_col], '.-')
        else:
            raise ValueError("Type needs to be loglog, plot, or semilogy")
    #ax2.loglog(p_results_helmholtz[-1].index**3, (p_results_helmholtz[-1].index)**6)
    ax2.legend(["$p$ = " + str(_) for _ in p_list2])# + ["trend for N**{1.5} scaling"])
    ax2.set_xlabel("$N$")
    #ax2.set_title(subtitle2)
    ax2.grid(True)

    plt.savefig(filename)
    plt.show()

# Here we'll create a figure plot:
def plot_trio_results(p_list1, p_list2, p_list3,
                      path1, path2, path3,
                      subtitle1, subtitle2, subtitle3,
                      title, ylabel,
                      data_col, filename, type="loglog"):
    figsize = (24,6)
    p_results_poisson = make_p_results(path1, p_list1)
    p_results_helmholtz = make_p_results(path2, p_list2)
    p_results_3 = make_p_results(path3, p_list3)


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    plt.rcParams['figure.figsize'] = [figsize[0],figsize[1]]
    plt.rc('text',usetex=True)
    plt.rc('font',**{'family':'serif','size':18})
    plt.rc('text.latex',preamble=r'\usepackage{amsfonts,bm}')
    fig.suptitle(title)
    for i in range(len(p_list1)):
        if type == "loglog":
            ax1.loglog(p_results_poisson[i].index**3, p_results_poisson[i][data_col], '.-')
        elif type == "plot":
            ax1.plot(p_results_poisson[i].index**3, p_results_poisson[i][data_col], '.-')
        elif type == "semilogy":
            ax1.semilogy(p_results_poisson[i].index**3, p_results_poisson[i][data_col], '.-')
        else:
            raise ValueError("Type needs to be loglog, plot, or semilogy")
    #ax1.loglog(p_results_poisson[-1].index**3, (p_results_poisson[-1].index)**6)
    ax1.legend(["$p$ = " + str(_) for _ in p_list1])# + ["trend for N**{1.5} scaling"])
    ax1.set_xlabel("$N$")
    ax1.set_ylabel(ylabel)
    ax1.set_title(subtitle1)
    ax1.grid(True)

    for i in range(len(p_list2)):
        #if i == 0 and type == "loglog":
        #    ax2.loglog(p_results_helmholtz[i].index[1:]**3, p_results_helmholtz[i][data_col][1:])
        if type == "loglog":
            ax2.loglog(p_results_helmholtz[i].index**3, p_results_helmholtz[i][data_col], '.-')
        elif type == "plot":
            ax2.plot(p_results_helmholtz[i].index**3, p_results_helmholtz[i][data_col], '.-')
        elif type == "semilogy":
            ax2.semilogy(p_results_helmholtz[i].index**3, p_results_helmholtz[i][data_col], '.-')
        else:
            raise ValueError("Type needs to be loglog, plot, or semilogy")
    #ax2.loglog(p_results_helmholtz[-1].index**3, (p_results_helmholtz[-1].index)**6)
    ax2.legend(["$p$ = " + str(_) for _ in p_list2])# + ["trend for N**{1.5} scaling"])
    ax2.set_xlabel("$N$")
    ax2.set_title(subtitle2)
    ax2.grid(True)

    for i in range(len(p_list3)):
        #if i == 0 and type == "loglog":
        #    ax2.loglog(p_results_helmholtz[i].index[1:]**3, p_results_helmholtz[i][data_col][1:])
        if type == "loglog":
            ax3.loglog(p_results_3[i].index**3, p_results_3[i][data_col], '.-')
        elif type == "plot":
            ax3.plot(p_results_3[i].index**3, p_results_3[i][data_col], '.-')
        elif type == "semilogy":
            ax3.semilogy(p_results_3[i].index**3, p_results_3[i][data_col], '.-')
        else:
            raise ValueError("Type needs to be loglog, plot, or semilogy")
    #ax3.loglog(p_results_helmholtz[-1].index**3, (p_results_helmholtz[-1].index)**6)
    ax3.legend(["$p$ = " + str(_) for _ in p_list2])# + ["trend for N**{1.5} scaling"])
    ax3.set_xlabel("$N$")
    ax3.set_title(subtitle3)
    ax3.grid(True)

    plt.savefig(filename)
    plt.show()

p_list_poisson   = [6, 8, 10, 12, 14]
p_list_helmholtz = [10,12,14,16,18,20,22]


path_poisson   = "gpu_output/poisson_0401/"
path_helmholtz = "gpu_output/helmholtz_10ppw_0401/"

p_results_poisson = make_p_results(path_poisson, p_list_poisson)


subtitle1 = "Poisson Equation"
subtitle2 = "Helmholtz Equation, 10 Points per Wavelength"
title     = "Relative Errors for Poisson and Helmholtz Equation"
ylabel    = "Relative Error"
filename  = "poisson_helmholtz_accuracy_gpu.pdf"
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, ylabel, "true_res", filename)
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, ylabel, "true_res", filename)


title     = "Matrix factorization time for Poisson and Helmholtz Equation"
ylabel    = "Seconds"
filename  = "poisson_helmholtz_factor_time_gpu.pdf"
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, ylabel, "toc_invert", filename)

title     = "DtN build time for Poisson and Helmholtz Equation"
filename  = "poisson_helmholtz_DtN_time_gpu.pdf"
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, ylabel, "toc_build_dtn", filename, type="plot")

title     = "Leaf solve time for Poisson and Helmholtz Equation"
filename  = "poisson_helmholtz_leaf_time_gpu.pdf"
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, ylabel, "toc_leaf_solve", filename, type="plot")

title     = "Factorized system solve time for Poisson and Helmholtz Equation"
filename  = "poisson_helmholtz_system_solve_time_gpu.pdf"
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, ylabel, "toc_system_solve", filename)