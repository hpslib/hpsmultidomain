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
p = 10
q = 8
a = 1.0

# TODO: figure out interpolation with Chebyshev / Gaussian (or other orthogonal) expansion
# coefficients, NOT monomial bases

