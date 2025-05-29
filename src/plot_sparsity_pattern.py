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
p = 8
nrows = 2 # Total number of rows
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

# First let's set Jl, Jr, Jd, Ju equal to Neumann derivatives:
A_block[JJ.Jl] = Nx[:p-2]
A_block[JJ.Jr] = Nx[p-2:2*(p-2)]
A_block[JJ.Jd] = Nx[2*(p-2):3*(p-2)]
A_block[JJ.Ju] = Nx[3*(p-2):4*(p-2)]

# Then we mask with 1 (interior) and -1 (boundary) points
A_block[A_block != 0] = 1
A_block[JJ.Jl] *= -1
A_block[JJ.Jr] *= -1
A_block[JJ.Jd] *= -1
A_block[JJ.Ju] *= -1

# Here we denote the corner nodes as 0. We will delete them later:
A_block[0,:]   = 0 #*= -2
A_block[p-1,:] = 0 #*= -2
A_block[-p,:]  = 0 #*= -2
A_block[-1,:]  = 0 #*= -2

A_block[:,0]   = 0 #*= -2
A_block[:,p-1] = 0 #*= -2
A_block[:,-p]  = 0 #*= -2
A_block[:,-1]  = 0 #*= -2

# Now we set boundary columns equal to -1 too:
A_block = np.minimum(A_block, A_block.T)

#plt.imshow(A_block, cmap='bwr', vmin=-2, vmax=2)
#plt.show()

A_row = block_diag([A_block] * nboxes)
A_row = A_row.toarray()

#plt.imshow(A_row, cmap='bwr', vmin=-2, vmax=2)
#plt.show()

# Now we need to "shift" Jl to Jr from previous box:
for j in range(1, nboxes):
    # Rows:
    A_row[j*(p**2)-p:j*(p**2)]   += A_row[j*(p**2):j*(p**2)+p]
    # Then columns:
    A_row[j*(p**2):(j+1)*(p**2),j*(p**2)-p:j*(p**2)] += A_row[j*(p**2):(j+1)*(p**2),j*(p**2):j*(p**2)+p]

#plt.imshow(A_row, cmap='bwr', vmin=-2, vmax=2)
#plt.show()


# Optional for now: set redundant rows and columns correlating to Jl to 0:
for j in range(nboxes):
    A_row[j*(p**2):j*(p**2)+p]   = 0
    A_row[:,j*(p**2):j*(p**2)+p] = 0

# Plus set right side of last box to 0:
A_row[-p:] = 0
A_row[:,-p:] = 0

#plt.imshow(A_row, cmap='bwr', vmin=-2, vmax=2)
#plt.show()


# Next step: stack two of these rows together

A = block_diag([A_row] * nrows)
A = A.toarray()

#plt.imshow(A, cmap='bwr', vmin=-2, vmax=2)
#plt.show()

# We need to set the off-diagonal that has the interaction between down and up Neumann faces.
# Let's do this with the shift approach from above:
for j in range(nboxes*(nrows-1)):
    A[JJ.Ju + j*(p**2)] += A[JJ.Jd + (j+nboxes)*(p**2)]
    A[:,JJ.Ju + j*(p**2)] += A[:,JJ.Jd + (j+nboxes)*(p**2)]

# Correct a redundant stacking here:
A[A==-2] = -1

#plt.imshow(A, cmap='bwr', vmin=-2, vmax=2)
#plt.show()

# Now let's set all down faces to 0:
for j in range(nboxes*nrows):
    A[JJ.Jd + j*(p**2)]   = 0
    A[:,JJ.Jd + j*(p**2)] = 0

# And let's set up faces from top row (last row) to 0:
for j in range(nboxes*(nrows-1), nboxes*nrows):
    A[JJ.Ju + j*(p**2)]   = 0
    A[:,JJ.Ju + j*(p**2)] = 0

#plt.imshow(A, cmap='bwr', vmin=-2, vmax=2)
#plt.show()

# Finally to create A, let's delete all rows of only 0s. These are redundant entries we do not
# need to compute:
zero_rows = np.where(np.all(A == 0, axis=1))[0]
zero_cols = np.where(np.all(A.T == 0, axis=1))[0]

print(zero_rows == zero_cols) # Sanity check

A = np.delete(A, zero_rows, axis=0)
A = np.delete(A, zero_cols, axis=1)

plt.imshow(A, cmap='bwr', vmin=-1, vmax=1)
plt.savefig("A_naive_denoted.pdf")
plt.show()

#
# With A now created, let's permute it
#

# First by rows:
A_positive = A[(A > 0).any(axis=1)]
A_negative = A[(A <= 0).all(axis=1)]

A = np.concatenate((A_positive, A_negative))

# Now by columns:
A = A.T
A_positive = A[(A > 0).any(axis=1)]
A_negative = A[(A <= 0).all(axis=1)]

A = np.concatenate((A_positive, A_negative))
A = A.T


plt.imshow(A, cmap='bwr', vmin=-1, vmax=1)
plt.savefig("A_rearranged_denoted.pdf")
plt.show()

# Now we can set up the block LDU decomposition:
I = A_positive.shape[0]

A_II = A[:I,:I]
A_IB = A[:I,I:]
N_I  = A[I:,:I]
N_B  = A[I:,I:]

S = -np.linalg.inv(A_II) @ A_IB

L = np.zeros(A.shape)
L[I:,:I]  = N_I # Get structure of N_I
L[L != 0] = -1 # Set N_I * A_II^-1 to -1
L = L + np.eye(A.shape[0])

plt.imshow(L, cmap='bwr', vmin=-1, vmax=1)
plt.savefig("L_denoted.pdf")
plt.show()

T = N_I @ S + N_B
T[T != 0] = -1

D = np.zeros(A.shape)
D[:I,:I] = np.eye(A_II.shape[0])
D[I:,I:] = T

plt.imshow(D, cmap='bwr', vmin=-1, vmax=1)
plt.savefig("D_denoted.pdf")
plt.show()

Aii = np.zeros(A.shape)
Aii[:I,:I] = A_II
Aii[I:,I:] = np.eye(N_B.shape[0])

plt.imshow(Aii, cmap='bwr', vmin=-1, vmax=1)
plt.savefig("Aii_denoted.pdf")
plt.show()

U = np.zeros(A.shape)
U[:I,I:]  = -S
U[U != 0] = -1
U = U + np.eye(A.shape[0])

plt.imshow(U, cmap='bwr', vmin=-1, vmax=1)
plt.savefig("U_denoted.pdf")
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