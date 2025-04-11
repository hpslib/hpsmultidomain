import numpy as np
from scipy.sparse import block_diag

from hps_leaf_disc import *

import matplotlib.pyplot as plt

# Plan:
# Create dense blocks corresponding to A_II, A_IB, and N
# Arrange them reasonably
# Make sparsity pattern

#
# Let's start with just 5 boxes in a row, arranged in 1D
#
d = 2
p = 6
nboxes = 3 # number of boxes along one dimension
kh = 3
a = 1 / nboxes

zz,Ds,JJ,hmin = leaf_discretization_2d(a,p)
Nx, _ = get_diff_ops(Ds,JJ,d)

# We'll assume 2D constant-coefficient Helmholtz
A_block = Ds.D11 + Ds.D22 + kh * np.eye(Ds.D11.shape[0], Ds.D11.shape[0])
#A_block = A_block[JJ.Jc,:]

# For each block:
# 1. Eliminate points on domain boundary
# 2. Remove points that are redundant
# 3. Set non-redundant domain points to Neumann derivatives

# First let's set Jl and Jd equal to Neumann derivatives:
A_block[JJ.Jl] = Nx[:p-2]
A_block[JJ.Jd] = Nx[2*(p-2):3*(p-2)]
A_block[JJ.Ju] = Nx[3*(p-2):4*(p-2)]

# Now let's delete Jr rows:
A_block = A_block[:-p] #np.delete(A_block, JJ.Jr, axis=0)
#A_block = np.delete(A_block, JJ.Ju, axis=1)

# Also delete the bottom two corner nodes:
A_block = np.delete(A_block, (0, p-1), axis=0)
# Here we delete corners from columns:
A_block = np.delete(A_block, (0, p-1, -p, -1), axis=1)

print(A_block.shape)
print(Nx.shape)

A_row = block_diag([A_block] * nboxes)
A_row = A_row.toarray()

box_size = p**2 - 4
edge_no_corner = p-2

# What remains: shift Jr columns to next box, then add Nr from previous box to current one

# Add Nr from previous box to current one:
Nx_no_corner = np.delete(Nx, (0,p-1,-p,-1), axis=1)
for j in range(1, nboxes):
    A_row[j*(box_size-edge_no_corner):j*(box_size-edge_no_corner)+edge_no_corner,(j-1)*box_size:j*box_size] = Nx_no_corner[edge_no_corner:2*edge_no_corner]

# Shift Jr columns to next box:
for j in range(1, nboxes):
    A_row[:,j*box_size-edge_no_corner:j*box_size] += A_row[:,j*box_size:j*box_size+edge_no_corner]


# Delete redundant columns:
for j in range(nboxes-1, 0, -1):
    A_row = np.delete(A_row, range(j*box_size,j*box_size+edge_no_corner), axis=1)


# Now delete that firstsection corresponding to far left edge:
A_row = A_row[edge_no_corner:,edge_no_corner:]
A_row = A_row[:,:-edge_no_corner]

plt.spy(A_row)
plt.show()

#
# Now let's create the entire sparsity matrix with multiple rows.
# First we need interior rows with down and up replaced with N
#
#A_row_int = 





A = block_diag([A_row] * nboxes)
A = A.toarray()

plt.spy(A)
plt.show()