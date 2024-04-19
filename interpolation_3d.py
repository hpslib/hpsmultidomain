import numpy as np
from collections import namedtuple
from time import time
import numpy.polynomial.chebyshev as cheb_py
import scipy.linalg

from numpy.polynomial  import legendre
from scipy.interpolate import RegularGridInterpolator, interpn
from hps_leaf_disc     import cheb


# let p be the degree of polynomial and our domian be [-a,a]:
p = 8
q = p # let Chebyshev and Gaussian node counts be the same:
a = 1.0

_, croots = cheb(p-1)
croots = np.flip(croots)

lcoeff = np.zeros(q+1)
lcoeff[-1] = 1
lroots = a * legendre.legroots(lcoeff)

# This gives us the p^2 X 2 array of our xy coordinates in Gaussian nodes
croots2d = np.array([np.repeat(croots, p), np.hstack([croots]*p)])
lroots2d = np.array([np.repeat(lroots, q), np.hstack([lroots]*q)])
"""
print(croots)
print(lroots)
print(croots2d)
print(lroots2d)
"""
# Idea: use interpn on the range of basis functions for croots2D to generate the columns of our interpolation matrix:
# What to do with specific corners and edges (ask about corner assignment in 2d)?
cpoints = (croots, croots) # tuple of our 2D Chebyshev points
values = np.zeros((p,p))

Interp_loc = []

for i in range(p):
    for j in range(p):
        values[:,:] = 0
        values[i,j] = 1
        Interp_loc.append(interpn(cpoints, values, lroots2d.T, method='linear'))

Interp_loc = np.column_stack(Interp_loc)
print("Interpolation matrix is ")
print(Interp_loc)
print(Interp_loc.shape)

# Seems to work, can experiment with different methods (linear, cubic, pchip, etc.)

# For no-corners: need to assign 2 edges and 1-2 corner points to each face.
# For Gaussian: steal idea from Fortunato paper to fix boundary discontinuities
