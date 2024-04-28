import numpy as np
from collections import namedtuple
from time import time
import numpy.polynomial.chebyshev as cheb_py
import scipy.linalg

from numpy.polynomial  import legendre
from numpy.polynomial.polynomial  import polyvander2d
from scipy.interpolate import RegularGridInterpolator, interpn
from hps_leaf_disc     import cheb


# let p be the degree of polynomial and our domian be [-a,a]:
p = 5
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

def get_loc_interp_3d(p, q, a):
    """
    Computes local interpolation matrices from Chebyshev points.
    
    Parameters:
    - q: The degree of the Chebyshev polynomial for interpolation
    - q: The degree of the Gaussian polynomial for interpolation
    
    Returns:
    - Interp_loc: Local interpolation matrix
    - err: Norm of the interpolation error
    - cond: Condition number of the interpolation matrix
    """
    _, croots  = cheb(p-1)
    croots     = a * np.flip(croots)
    croots2d   = np.array([np.repeat(croots, p), np.hstack([croots]*p)])
    lcoeff     = np.zeros(q+1)
    lcoeff[-1] = 1

    lroots   = a * legendre.legroots(lcoeff)
    lroots2d = np.array([np.repeat(lroots, q), np.hstack([lroots]*q)])
    cpoints  = (croots, croots) # tuple of our 2D Chebyshev points
    values   = np.zeros((p,p))

    """
    Interp_loc = []
    for i in range(p):
        for j in range(p):
            values[:,:] = 0
            values[i,j] = 1
            Interp_loc.append(interpn(cpoints, values, lroots2d.T, method='linear'))

    Interp_loc = np.column_stack(Interp_loc)
    """

    # Vandermonde-based approach:
    Vc = polyvander2d(croots2d[0], croots2d[1], (p,p))
    Vl = polyvander2d(lroots2d[0], lroots2d[1], (q,q))

    Interp_loc = np.linalg.lstsq(Vl.T,Vc.T,rcond=None)[0].T

    cond = np.linalg.cond(Interp_loc)
    # TODO: get err
    return Interp_loc,cond

Interp_loc, cond = get_loc_interp_3d(p, q, a)
print("Interpolation matrix is ")
print(Interp_loc)
print(Interp_loc.shape)
print(cond)

# Seems to work, can experiment with different methods (linear, cubic, pchip, etc.)

# For no-corners: need to assign 2 edges and 1-2 corner points to each face.
# For Gaussian: steal idea from Fortunato paper to fix boundary discontinuities

# Idea: given Gaussian data on the boundaries, we develop a projector similar to Fortunato
# to convert it into Gaussian data that is continuous on the corners. We then apply this to
# our Gaussian data when making it Chebyshev
# Steps:
# 1. Build "naive" interpolation map that acts on all six faces, converting from Gaussian to Chebyshev
# 2. Build "B" that enforces continuity on the edge points (12 edges, including 8 corners)
# 3. SVD B and get V tilde
# 4. Apply V tilde to right side of interpolation matrix (or left?). Profit.

Interp_mat = scipy.linalg.block_diag(*np.repeat(np.expand_dims(Interp_loc,0),6,axis=0))

print(Interp_mat)
print(Interp_mat.shape)

print(lroots2d[0])
print(lroots2d[1])

Vl = polyvander2d(lroots2d[0], lroots2d[1], (q,q))

print(Vl.shape)

# Determine what indices in Jxreorder are redundant...