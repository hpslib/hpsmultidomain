from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

mypath = "test_interpolation_operator"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
kh_list   = [0, 2, 4, 6, 8, 10, 12, 14, 16]
p_list    = [8, 10, 12, 14, 16, 18, 20, 22]

kh_results = []

for kh in kh_list:
    kh_files = [_ for _ in onlyfiles if "_kh_" + str(kh) + "." in _]
    GtC_error     = []
    GtC_cond      = []
    CtG_error     = []
    CtG_cond      = []
    redundant_var = []
    for p in p_list:
        pkh_file = [_ for _ in kh_files if "_p_" + str(p) + "_" in _][0]

        with open(mypath + "/" + pkh_file, 'rb') as f:
            x = pickle.load(f)
            GtC_error.append(x["GtC_error"].item())
            GtC_cond.append(x["GtC_cond"])
            CtG_error.append(x["CtG_error"].item())
            CtG_cond.append(x["CtG_cond"])
            redundant_var.append(x["redundant_var"])

    kh_result = dict(p=p_list, GtC_error=GtC_error, GtC_cond=GtC_cond, CtG_error=CtG_error,
                    CtG_cond=CtG_cond, redundant_var=redundant_var)
    
    kh_result = pd.DataFrame.from_dict(kh_result)
    kh_result.set_index('p', inplace=True)
    kh_result.sort_index(inplace=True)
    kh_results.append(pd.DataFrame.from_dict(kh_result))

def make_plot(field, title, xlabel, ylabel, type="plot"):
    legend = []
    for i in range(len(kh_list)):
        if type=="plot":
            plt.plot(kh_results[i].index, kh_results[i][field])
        if type=="semilogy":
            plt.semilogy(kh_results[i].index, kh_results[i][field])
        if type=="loglog":
            plt.loglog(kh_results[i].index, kh_results[i][field])
        legend.append("kh = " + str(kh_list[i]))

    plt.title(title)
    plt.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig("plots_interpolation/" + field + ".png")
    plt.show()

make_plot("GtC_error", "Gaussian-to-Chebyshev Interpolation Error, domain size [1/2]x[1/2]", "p", "relative error", type="semilogy")
make_plot("CtG_error", "Chebyshev-to-Gaussian Interpolation Error, domain size [1/2]x[1/2]", "p", "relative error", type="semilogy")
make_plot("GtC_cond", "Gaussian-to-Chebyshev Condition Number, domain size [1/2]x[1/2]", "p", "cond #", type="semilogy")
make_plot("CtG_cond", "Chebyshev-to-Gaussian Condition Number, domain size [1/2]x[1/2]", "p", "cond#", type="semilogy")
make_plot("redundant_var", "Biggest error between repeated edge/corner values after interpolation", "p", "relative error", type="semilogy")