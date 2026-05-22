from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


def make_p_results(mypath, p_list, sparse_diag=False):
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

        sparse_mem = []
        factorized_mem = []

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
                if sparse_diag:
                    toc_build_dtn[-1] += x["toc_sparse_arrange"]

                toc_system_solve.append(x["toc_solve_petsc"]) # Old
                toc_leaf_solve.append(x["toc_solve_leaf"])
                sparse_solve_res.append(x["res_solve_petsc"])
                true_res.append(x["trueres_solve_petsc"])
                leaf_res.append(x["resloc_hps_solve_petsc"])
                #forward_bdry_error.append(x["forward_bdry_error"])
                #reverse_bdry_error.append(x["reverse_bdry_error"])

                sparse_mem.append(x["sparse_mem"])
                factorized_mem.append(x["factorized_mem"])

                # Interpolation stuff
                """
                GtC_error.append(x["GtC_error"])
                CtG_error.append(x["GtC_error"])
                GtC_cond.append(x["GtC_cond"])
                CtG_cond.append(x["CtG_cond"])
                neumann_tensor_error.append(x["neumann_tensor_error"])
                neumann_sparse_error.append(x["neumann_sparse_error"])
                dtn_cond.append(x["dtn_cond"])"""

        toc_total_build = np.array(toc_build_dtn) + np.array(toc_invert)
        toc_total_solve = np.array(toc_system_solve) + np.array(toc_leaf_solve)
            
        p_result = dict(n=n, toc_invert=toc_invert, toc_build_dtn=toc_build_dtn, toc_leaf_solve=toc_leaf_solve,
                        sparse_solve_res=sparse_solve_res, true_res=true_res, leaf_res=leaf_res,
                        toc_system_solve=toc_system_solve,
                        sparse_mem=sparse_mem, factorized_mem=factorized_mem,
                        toc_total_build=toc_total_build, toc_total_solve=toc_total_solve)
                        #delta_t=delta_t)
                        #forward_bdry_error=forward_bdry_error, reverse_bdry_error=reverse_bdry_error,
                        #GtC_error=GtC_error, CtG_error=CtG_error, GtC_cond=GtC_cond,CtG_cond=CtG_cond, #INTERPOLATION
                        #neumann_tensor_error=neumann_tensor_error, neumann_sparse_error=neumann_sparse_error, dtn_cond=dtn_cond)
        
        p_result = pd.DataFrame.from_dict(p_result)

        if pd.api.types.is_object_dtype(p_result['n']):
            p_result['n'] = p_result['n'].str[0]

        p_result.set_index('n', inplace=True)
        p_result.sort_index(inplace=True)

        # Postprocessing on factorized memory:
        p_result = postprocess_factorized_memory(p_result)

        p_results.append(pd.DataFrame.from_dict(p_result))

    return p_results

# This is needed since some matrices get too large and # nonzeros is stored as int32
def postprocess_factorized_memory(p_result):

    overflow = 0

    col = p_result.columns.get_loc("factorized_mem")

    for i in range(len(p_result)):
        if (i != 0) and ((p_result.iat[i, col] + overflow * 2**32) < p_result.iat[i - 1, col]):
            overflow += 1
        p_result.iat[i, col] += overflow * 2**32

    p_result["factorized_mem"] = (p_result["factorized_mem"] * 8 + p_result["factorized_mem"] * 4) / 1e9

    return p_result


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
    p_results_poisson = make_p_results(path1, p_list1, sparse_diag=True)
    p_results_helmholtz = make_p_results(path2, p_list2)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

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

    ax2.sharey(ax1)
    ax2.sharex(ax1)

    plt.savefig(filename)
    plt.show()

# Here we'll create a figure plot:
def plot_trio_results(p_list1, p_list2, p_list3,
                      path1, path2, path3,
                      subtitle1, subtitle2, subtitle3,
                      title, ylabel,
                      data_col, filename, type="loglog"):
    figsize = (24,6)
    p_results_poisson = make_p_results(path1, p_list1, sparse_diag=True)
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


# Here we'll create a figure plot:
def plot_trio_results_new(p_list1, p_list2, path1, path2, subtitle1, subtitle2, subtitle3, title, ylabel1, ylabel2, ylabel3, data_col1, data_col2, data_col3, filename, type="loglog"):
    figsize = (25,6)
    p_results_poisson = make_p_results(path1, p_list1, sparse_diag=True)
    p_results_helmholtz = make_p_results(path2, p_list2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(wspace=0.2)
    #fig.suptitle(title)

    plt.rcParams['figure.figsize'] = [figsize[0],figsize[1]]
    plt.rc('text',usetex=True)
    plt.rc('font',**{'family':'serif','size':22})
    plt.rc('text.latex',preamble=r'\usepackage{amsfonts,bm}')

    plot_fn = {"loglog": ax1.loglog, "plot": ax1.plot, "semilogy": ax1.semilogy}.get(type)
    if plot_fn is None:
        raise ValueError("Type needs to be loglog, plot, or semilogy")
    
    for i in range(len(p_list1)):
        plot_fn(p_results_poisson[i].index**3, p_results_poisson[i][data_col1], '.-', label="$p$ = " + str(p_list1[i]) + " (two-level)")

    ax1.set_prop_cycle(None)

    for i in range(len(p_list2)):
        plot_fn(p_results_helmholtz[i].index**3, p_results_helmholtz[i][data_col1], '.--', label="$p$ = " + str(p_list2[i]) + " (unreduced)")
    
    #ax1.legend(["$p$ = " + str(_) for _ in p_list1] + ["$p$ = " + str(_) for _ in p_list2]) #, loc='upper left', bbox_to_anchor=(1, 1))
    ax1.set_xlabel("$N$")
    ax1.set_ylabel(ylabel1)
    ax1.set_title(subtitle1)
    ax1.grid(True)

    plot_fn = {"loglog": ax2.loglog, "plot": ax2.plot, "semilogy": ax2.semilogy}.get(type)
    if plot_fn is None:
        raise ValueError("Type needs to be loglog, plot, or semilogy")

    for i in range(len(p_list1)):
        plot_fn(p_results_poisson[i].index**3, p_results_poisson[i][data_col2], '.-')

    ax2.set_prop_cycle(None)

    for i in range(len(p_list2)):
        plot_fn(p_results_helmholtz[i].index**3, p_results_helmholtz[i][data_col2], '.--')
    
    #ax2.legend(["$p$ = " + str(_) for _ in p_list2])
    ax2.set_xlabel("$N$")
    ax2.set_ylabel(ylabel2)
    ax2.set_title(subtitle2)
    ax2.grid(True)

    plot_fn = {"loglog": ax3.loglog, "plot": ax3.plot, "semilogy": ax3.semilogy}.get(type)
    if plot_fn is None:
        raise ValueError("Type needs to be loglog, plot, or semilogy")

    for i in range(len(p_list1)):
        plot_fn(p_results_poisson[i].index**3, p_results_poisson[i][data_col3], '.-')

    ax3.set_prop_cycle(None)

    for i in range(len(p_list2)):
        plot_fn(p_results_helmholtz[i].index**3, p_results_helmholtz[i][data_col3], '.--')
    
    #ax3.legend(["$p$ = " + str(_) for _ in p_list2])
    ax3.set_xlabel("$N$")
    ax3.set_ylabel(ylabel3)
    ax3.set_title(subtitle3)
    ax3.grid(True)

    #labels = ["$p$ = " + str(_) for _ in p_list1] + ["$p$ = " + str(_) for _ in p_list2]

    handles, labels = ax1.get_legend_handles_labels()

    num_p = len(p_list_poisson)

    # Interleave: [solid_p8, dashed_p8, solid_p10, dashed_p10, ...]
    interleaved_handles = [h for pair in zip(handles[:num_p], handles[num_p:]) for h in pair]
    interleaved_labels  = [l for pair in zip(labels[:num_p],  labels[num_p:])  for l in pair]

    fig.legend(interleaved_handles, interleaved_labels,
            loc='upper center', bbox_to_anchor=(0.5, 0), ncols=num_p)

    plt.savefig(filename, bbox_inches='tight')
    plt.show()

# CPU only comparison
"""
p_list_poisson   = [10, 14, 18, 22]
p_list_helmholtz = p_list_poisson

path_poisson   = "output/condensed_test_helmholtz_kh40_gpu/"
path_helmholtz = "output/not_condensed_test_helmholtz_kh40/"
"""

# With GPU:

p_list_poisson   = [10, 14, 18, 22] #[5, 10, 15, 20]
p_list_helmholtz = p_list_poisson

path_poisson   = "condense_or_no_output/condensed_test_helmholtz_kh40_gpu_sparse_diag/"
path_helmholtz = "condense_or_no_output/no_condensed_test_helmholtz_kh40_gpu/"


p_results_poisson = make_p_results(path_poisson, p_list_poisson)

"""
subtitle1 = "Statically Condensed"
subtitle2 = "Not Statically Condensed"
title     = "Relative Errors for GPU Helmholtz Equation"
ylabel    = "Relative Error"
filename  = "output/compared_accuracy.png"
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, ylabel, "true_res", filename)
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, ylabel, "true_res", filename)


title     = "Matrix factorization time for GPU Helmholtz Equation"
ylabel    = "Seconds"
filename  = "output/factor_time.png"
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, ylabel, "toc_invert", filename)

title     = "Prefactor Assembly time for GPU Helmholtz Equation"
filename  = "output/DtN_time.png"
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, ylabel, "toc_build_dtn", filename, type="plot")

title     = "Leaf solve time for GPU Helmholtz Equation"
filename  = "output/leaf_time.png"
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, ylabel, "toc_leaf_solve", filename, type="plot")

title     = "Factorized system solve time for GPU Helmholtz Equation"
filename  = "output/system_solve_time.png"
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, ylabel, "toc_system_solve", filename)

title     = "Total build time for GPU Helmholtz Equation"
ylabel    = "Seconds"
filename  = "output/total_build_time.png"
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, ylabel, "toc_total_build", filename)

title     = "Total system solve time for GPU Helmholtz Equation"
filename  = "output/total_solve_time.png"
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, ylabel, "toc_total_solve", filename)

title     = "Sparse system memory for GPU Helmholtz Equation"
filename  = "output/sparse_system_memory.png"
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, "Memory (GB)", "sparse_mem", filename, type="plot")


title     = "Factorized system memory for GPU Helmholtz Equation"
filename  = "output/factorized_system_memory.png"
plot_paired_results(p_list_poisson, p_list_helmholtz, path_poisson, path_helmholtz, subtitle1, subtitle2, title, "Memory (GB)", "factorized_mem", filename, type="plot")
"""


title = "Comparison of Statically-Condensed to Non-Condensed Solver"
subtitle1 = "Build Stage Time"
subtitle2 = "Solve Stage Time"
subtitle3 = "Factorized System Memory"
ylabel1 = "Seconds"
ylabel2 = "Seconds"
ylabel3 = "GB"
data_col1 = "toc_total_build"
data_col2 = "toc_total_solve"
data_col3 = "factorized_mem"
filename = "new-condensed-not-condensed-comp.png"
plot_trio_results_new(p_list_poisson, p_list_poisson, path_poisson, path_helmholtz, subtitle1, subtitle2, subtitle3, title, ylabel1, ylabel2, ylabel3, data_col1, data_col2, data_col3, filename, type="loglog")
plot_trio_results_new(p_list_poisson, p_list_poisson, path_poisson, path_helmholtz, subtitle1, subtitle2, subtitle3, title, ylabel1, ylabel2, ylabel3, data_col1, data_col2, data_col3, filename, type="loglog")


"""
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
"""