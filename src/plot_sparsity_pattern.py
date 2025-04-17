import numpy as np
from scipy.sparse import block_diag
import scipy.linalg as la

from hps_leaf_disc import *

import matplotlib.pyplot as plt
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'serif','size':14})
plt.rc('text.latex',preamble=r'\usepackage{amsfonts,bm}')

# Plan:
# Create dense blocks corresponding to A_II, A_IB, and N
# Arrange them reasonably
# Make sparsity pattern

#
# Let's start with just 5 boxes in a row, arranged in 1D
#
d = 2
p = 6
nboxes = 4 # number of boxes along one dimension
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

# Set A_block to be blue:
A_block_true = A_block
A_block = np.abs(A_block)

# First let's set Jl and Jd equal to Neumann derivatives:
A_block[JJ.Jl] = -np.abs(Nx[:p-2])
A_block[JJ.Jr] = np.abs(Nx[p-2:2*(p-2)])
A_block[JJ.Jd] = -np.abs(Nx[2*(p-2):3*(p-2)])
A_block[JJ.Ju] = np.abs(Nx[3*(p-2):4*(p-2)])

A_block_true[JJ.Jl] = Nx[:p-2]
A_block_true[JJ.Jd] = Nx[2*(p-2):3*(p-2)]
A_block_true[JJ.Ju] = 0

# Now let's delete Jr rows:
A_block = A_block[:-p]
A_block_true = A_block_true[:-p]

# Also delete the bottom two corner nodes:
A_block = np.delete(A_block, (0, p-1), axis=0)
A_block_true = np.delete(A_block_true, (0, p-1), axis=0)
# Here we delete corners from columns:
A_block = np.delete(A_block, (0, p-1, -p, -1), axis=1)
A_block_true = np.delete(A_block_true, (0, p-1, -p, -1), axis=1)

print(A_block.shape)
print(Nx.shape)

A_row = block_diag([A_block] * nboxes)
A_row = A_row.toarray()

A_row_true = block_diag([A_block_true] * nboxes)
A_row_true = A_row_true.toarray()

box_size = p**2 - 4
edge_no_corner = p-2

# What remains: shift Jr columns to next box, then add Nr from previous box to current one

# Add Nr from previous box to current one:
Nx_no_corner = np.delete(Nx, (0,p-1,-p,-1), axis=1)
for j in range(1, nboxes):
    A_row[j*(box_size-edge_no_corner):j*(box_size-edge_no_corner)+edge_no_corner,(j-1)*box_size:j*box_size] = -np.abs(Nx_no_corner[edge_no_corner:2*edge_no_corner])
    A_row_true[j*(box_size-edge_no_corner):j*(box_size-edge_no_corner)+edge_no_corner,(j-1)*box_size:j*box_size] = Nx_no_corner[edge_no_corner:2*edge_no_corner]

# Shift Jr columns to next box:
for j in range(1, nboxes):
    A_row[:,j*box_size-edge_no_corner:j*box_size] += A_row[:,j*box_size:j*box_size+edge_no_corner]
    A_row_true[:,j*box_size-edge_no_corner:j*box_size] += A_row_true[:,j*box_size:j*box_size+edge_no_corner]


# Delete redundant columns:
for j in range(nboxes-1, 0, -1):
    A_row = np.delete(A_row, range(j*box_size,j*box_size+edge_no_corner), axis=1)
    A_row_true = np.delete(A_row_true, range(j*box_size,j*box_size+edge_no_corner), axis=1)


# Now delete that firstsection corresponding to far left edge:
A_row = A_row[edge_no_corner:,edge_no_corner:]
A_row = A_row[:,:-edge_no_corner]

A_row_true = A_row_true[edge_no_corner:,edge_no_corner:]
A_row_true = A_row_true[:,:-edge_no_corner]

A_row[A_row > 0] = 1.0
A_row[A_row < 0] = -1.0

A_row = np.minimum(A_row, A_row.T)

A_row_max = np.max(np.abs(A_row))
A_row_true_max = np.max(np.abs(A_row_true))
plt.imshow(A_row, cmap='bwr', vmin=-A_row_max, vmax=A_row_max)
plt.savefig("A_naive_denoted.pdf")
plt.show()

plt.imshow(A_row_true, cmap='seismic', vmin=-0.1*A_row_true_max, vmax=0.1*A_row_true_max)
plt.colorbar()
plt.savefig("A_naive_values.pdf")
plt.show()

#
# Now let's create the entire sparsity matrix with multiple rows.
#
nrows = 2

A_offdiag = A_row.copy()
A_offdiag[A_offdiag > 0] = 0

plt.imshow(A_offdiag, cmap='bwr', vmin=-A_row_max, vmax=A_row_max)
plt.show()

A = block_diag([A_row] * nrows)
A = A.toarray()

A[:A_row.shape[0],A_row.shape[0]:] = A_offdiag
A[A_row.shape[0]:,:A_row.shape[0]] = A_offdiag.T

plt.imshow(A, cmap='bwr', vmin=-A_row_max, vmax=A_row_max)
plt.show()






"""
print(A_row.shape)

A_row_positive = A_row[(A_row > 0).any(axis=1)]
A_row_negative = A_row[(A_row <= 0).all(axis=1)]

A_row_true_positive = A_row_true[(A_row > 0).any(axis=1)]
A_row_true_negative = A_row_true[(A_row <= 0).all(axis=1)]

print(A_row_positive.shape, A_row_negative.shape)

A_row = np.concatenate((A_row_positive, A_row_negative))
A_row_true = np.concatenate((A_row_true_positive, A_row_true_negative))

# Now sort by columns:
A_row = A_row.T
A_row_positive = A_row[(A_row > 0).any(axis=1)]
A_row_negative = A_row[(A_row <= 0).all(axis=1)]

A_row_true = A_row_true.T
A_row_true_positive = A_row_true[(A_row > 0).any(axis=1)]
A_row_true_negative = A_row_true[(A_row <= 0).all(axis=1)]

print(A_row_positive.shape, A_row_negative.shape)

A_row = np.concatenate((A_row_positive, A_row_negative))
A_row_true = np.concatenate((A_row_true_positive, A_row_true_negative))
A_row = A_row.T
A_row_true = A_row_true.T

print(A_row.shape)
plt.imshow(A_row, cmap='bwr', vmin=-A_row_max, vmax=A_row_max)
plt.savefig("A_rearranged_denoted.pdf")
plt.show()

plt.imshow(A_row_true, cmap='seismic', vmin=-0.1*A_row_true_max, vmax=0.1*A_row_true_max)
plt.colorbar()
plt.savefig("A_rearranged_values.pdf")
plt.show()

#
# Now let's apply Gaussian Elimination to A_row
#

top_left = A_row_positive.shape[0]
A_ii_size = top_left // nboxes

A_ii = A_row[:A_ii_size,:A_ii_size]
A_ii_inv = np.linalg.inv(A_ii)

A_ii_true = A_row_true[:A_ii_size,:A_ii_size]
A_ii_inv_true = np.linalg.inv(A_ii_true)

for i in range(nboxes):
    A_row[i*A_ii_size:(i+1)*A_ii_size,:] = A_ii_inv @ A_row[i*A_ii_size:(i+1)*A_ii_size,:]
    A_row_true[i*A_ii_size:(i+1)*A_ii_size,:] = A_ii_inv_true @ A_row_true[i*A_ii_size:(i+1)*A_ii_size,:]

plt.imshow(A_row, cmap='bwr', vmin=-A_row_max, vmax=A_row_max)
plt.show()

A_row_true_max = np.max(np.abs(A_row_true))
plt.imshow(A_row_true, cmap='seismic', vmin=-0.1*A_row_true_max, vmax=0.1*A_row_true_max)
plt.colorbar()
plt.savefig("A_inverted_values.pdf")
plt.show()

#
# Now we apply "Gaussian Elimination" to get L, D, and U
#
# L:
L = A_row_true.copy()
L[:top_left,top_left:] = 0
L[top_left:,top_left:] = np.eye(A_row_true.shape[0] - top_left)

N = L[top_left:,:top_left].copy()
N[N != 0] = -1
L[top_left:,:top_left] = N

plt.imshow(L, cmap='bwr', vmin=-1, vmax=1)
#plt.imshow(L, cmap='seismic', vmin=-0.1*A_row_true_max, vmax=0.1*A_row_true_max)
#plt.colorbar()
plt.savefig("L_denoted.pdf")
plt.show()

# D:
D = A_row_true.copy()
D[:top_left,top_left:]  = 0
D[top_left:,:top_left]  = 0
D[top_left:,top_left:] -= A_row_true[top_left:,:top_left] @ A_row_true[:top_left,top_left:]

T = D[top_left:,top_left:].copy()
T[T != 0] = -1
D[top_left:,top_left:] = T

plt.imshow(D, cmap='bwr', vmin=-1, vmax=1)
#plt.imshow(D, cmap='bwr', vmin=-0.1*A_row_true_max, vmax=0.1*A_row_true_max)
#plt.colorbar()
plt.savefig("D_denoted.pdf")
plt.show()

# U:
U = A_row_true.copy()
U[top_left:,:top_left] = 0
U[top_left:,top_left:] = np.eye(A_row_true.shape[0] - top_left)

S = U[:top_left,top_left:].copy()
S[S != 0] = -1
U[:top_left,top_left:] = S

plt.imshow(U, cmap='bwr', vmin=-1, vmax=1)
#plt.imshow(U, cmap='seismic', vmin=-0.1*A_row_true_max, vmax=0.1*A_row_true_max)
#plt.colorbar()
plt.savefig("U_denoted.pdf")
plt.show()

"""