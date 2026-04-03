import numpy as np
import torch

import matplotlib.pyplot as plt
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'serif','size':18})
plt.rc('text.latex',preamble=r'\usepackage{amsfonts,bm}')

# This is a function that produces a plot of the solution to the given PDE
# Several variables reuqire manual editing to get the best visual results
# - such as center, interior, and the x, y, and z bounds

def visualize_problem(dom, curved_domain, param_map, uu_sol, p, kh=0, d=2, n=0, f=None):

    center=np.array([0,0,0])

    xx = dom.hps.grid_xx.flatten(start_dim=0,end_dim=-2)
    #xx = dom.hps.grid_ext

    if curved_domain:
        xx = param_map(xx)

    sequence_containing_x_vals = xx[:,0] - center[0]
    sequence_containing_y_vals = xx[:,1] - center[1]
    if d==3:
        sequence_containing_z_vals = xx[:,2] - center[2]

    result = uu_sol.flatten() #uu_sol[:,Jx].flatten()

    max_result = torch.linalg.norm(result, ord=np.inf)

    #h = round(a[0] * 2, 2)

    import matplotlib.pyplot as plt
    plt.rc('text',usetex=True)
    plt.rc('font',**{'family':'serif','size':18})
    plt.rc('text.latex',preamble=r'\usepackage{amsfonts,bm}')

    fig = plt.figure(figsize=(12, 12))
    if d==2:
        ax = fig.add_subplot()
        sc = ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, c=result, marker='.', cmap="seismic", vmin=-max_result, vmax=max_result)
    if d==3:
        ax = fig.add_subplot(projection='3d')
        #ax.view_init(azim=-30)
        #ax.view_init(elev=5, azim=-5)
        ax.view_init(elev=95, azim=-90)
        sc = ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals, c=result, marker='.', cmap="seismic", vmin=-max_result, vmax=max_result)
        ax.set_xticks([-1.0, 3.0])
        ax.set_xticklabels(["-1.0", "3.0"])
        ax.set_yticks([-3.0, 1.0])
        ax.set_yticklabels(["-3.0", "1.0"])
        ax.set_zticks([-.9, -.7])
        ax.set_zticklabels(["-.9", "-.7"])

    #ax.set_xticks([1.4, 1.6])
    #ax.set_xticklabels(["1.4", "1.6"])

    #plt.title("Helmholtz Equation: $\kappa = $" + str(kh) + ", $p = $" + str(p) + ", $h$ = 0.625 x 0.5 x 0.5")
    plt.title("Checkerboard convection-diffusion steady state, p = " + str(p[0]) + ", n = " + str(n))
    plt.xlabel("x")
    plt.ylabel("  y")
    plt.colorbar(sc, shrink=0.5)
    plt.rcParams['figure.figsize'] = [14, 6]
    #plt.savefig("3D-domain-faces-annulus-p18-h16x2x2.png")
    plt.savefig("data-convection-checkerboard-p-refine/convection-checkerboard-p" + str(p[0]) + "-" + str(n) + ".png")
    #plt.show()

    """
    if f is not None:
        fig2 = plt.figure(figsize=(12, 12))
        ax2 = fig2.add_subplot()
        sc2 = ax2.scatter(sequence_containing_x_vals, sequence_containing_y_vals, c=f(xx), marker='.', cmap="seismic") #, vmin=-max_result, vmax=max_result)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(sc2, shrink=0.5)
        plt.rcParams['figure.figsize'] = [14, 6]
        plt.show()
    """
