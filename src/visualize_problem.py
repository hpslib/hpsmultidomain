import numpy as np
import torch

import matplotlib.pyplot as plt
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'serif','size':18})
plt.rc('text.latex',preamble=r'\usepackage{amsfonts,bm}')

# This is a function that produces a plot of the solution to the given PDE
# Several variables reuqire manual editing to get the best visual results
# - such as center, interior, and the x, y, and x bounds

def visualize_problem(dom, curved_domain, param_map, uu_sol, p, kh=0):

    center=np.array([-1.1,+1.,+1.2])

    xx = dom.hps.grid_xx.flatten(start_dim=0,end_dim=-2)
    #xx = dom.hps.grid_ext

    #print(xx)
    tol = 1e-8
    interior0 = torch.logical_and((xx[:, 0] > 0.3 + tol), (xx[:, 0] < 0.5 + tol))
    #interior1 = torch.logical_and((xx[:, 1] > 0 + tol), (xx[:, 1] < 1 - tol))
    interior2 = torch.logical_and((xx[:, 2] > 0.3 + tol), (xx[:, 2] < 0.5 + tol))

    interior = interior2 #torch.logical_and(torch.logical_and(interior0, interior1), interior2)

    if curved_domain:
        xx = param_map(xx)

    sequence_containing_x_vals = xx[:,0] - center[0]
    sequence_containing_y_vals = xx[:,1] - center[1]
    sequence_containing_z_vals = xx[:,2] - center[2]

    norms = np.sqrt(sequence_containing_x_vals*sequence_containing_x_vals
                    + sequence_containing_y_vals*sequence_containing_y_vals
                    + sequence_containing_z_vals*sequence_containing_z_vals)

    Jx = torch.tensor(dom.hps.H.JJ.Jxreorder)

    result = uu_sol.flatten() #uu_sol[:,Jx].flatten()

    max_result = torch.linalg.norm(result, ord=np.inf)


    # Eliminates the domain exterior points
    sequence_containing_x_vals = sequence_containing_x_vals[interior]
    sequence_containing_y_vals = sequence_containing_y_vals[interior]
    sequence_containing_z_vals = sequence_containing_z_vals[interior]
    result = result[interior]

    #h = round(a[0] * 2, 2)

    import matplotlib.pyplot as plt
    plt.rc('text',usetex=True)
    plt.rc('font',**{'family':'serif','size':18})
    plt.rc('text.latex',preamble=r'\usepackage{amsfonts,bm}')

    fig = plt.figure(figsize=(12, 12))
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

    plt.title("Helmholtz Equation: $\kappa = $" + str(kh) + ", $p = $" + str(p) + ", $h$ = 0.625 x 0.5 x 0.5")
    plt.xlabel("x")
    plt.ylabel("  y")
    plt.colorbar(sc, shrink=0.5)
    plt.rcParams['figure.figsize'] = [14, 6]
    #plt.savefig("3D-domain-faces-annulus-p18-h16x2x2.png")
    plt.show()
