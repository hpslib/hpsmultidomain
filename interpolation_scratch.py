import numpy as np

from numpy.polynomial  import legendre
from numpy.polynomial.polynomial import polyvander2d
from numpy.polynomial.chebyshev import chebvander2d
from numpy.polynomial.legendre import legvander2d

import matplotlib.pyplot as plt

def cheb(p):
    """
    Computes the Chebyshev differentiation matrix and Chebyshev points for a given degree p.
    
    Parameters:
    - p: The polynomial degree
    
    Returns:
    - D: The Chebyshev differentiation matrix
    - x: The Chebyshev points
    """
    x = np.cos(np.pi * np.arange(p+1) / p)
    c = np.concatenate((np.array([2]), np.ones(p-1), np.array([2])))
    c = np.multiply(c,np.power(np.ones(p+1) * -1, np.arange(p+1)))
    X = x.repeat(p+1).reshape((-1,p+1))
    dX = X - X.T
    # create the off diagonal entries of D
    D = np.divide(np.outer(c,np.divide(np.ones(p+1),c)), dX + np.eye(p+1))
    D = D - np.diag(np.sum(D,axis=1))
    return D,x

# Ideas:
# Do two Lagrange transfroms in 1D, first x then y
# Or avoid interpolating to Gaussian altogether... mayber shift just edge nodes ever so slightly off the edges?
# Try both of these

def get_legendre_row(x, input_row):
    output_row = np.zeros(len(input_row))
    return output_row

def get_loc_interp_3d(p, q, a, l):
    """
    Computes local interpolation matrices from Chebyshev points.
    
    Parameters:
    - p: The degree of the Chebyshev polynomial for interpolation
    - q: The degree of the Gaussian polynomial for interpolation
    
    Returns:
    - Interp_loc: Local interpolation matrix
    - err: Norm of the interpolation error
    - cond: Condition number of the interpolation matrix
    """
    _, croots  = cheb(p-1)
    croots     = np.flip(croots)
    croots2d   = np.array([np.repeat(croots, p), np.hstack([croots]*p)])
    lcoeff     = np.zeros(q+1)
    lcoeff[-1] = 1

    #print(get_legendre_row(0, croots))

    lroots   = legendre.legroots(lcoeff)
    #lroots[1:-1] = croots[1:-1]
    lroots2d = np.array([np.repeat(lroots, q), np.hstack([lroots]*q)])

    # Vandermonde-based approach with Chebyshev expansion coefficients:
    ChebVc = chebvander2d(croots2d[0], croots2d[1], (l,l))
    ChebVl = chebvander2d(lroots2d[0], lroots2d[1], (l,l))

    # Vandermonde-based approach with Gaussian expansion coefficients:
    GaussVc = legvander2d(croots2d[0], croots2d[1], (l,l))
    GaussVl = legvander2d(lroots2d[0], lroots2d[1], (l,l))

    Interp_loc_CtG = np.linalg.lstsq(ChebVc.T,ChebVl.T,rcond=None)[0].T
    Interp_loc_GtC = np.linalg.lstsq(GaussVl.T,GaussVc.T,rcond=None)[0].T

    condGtC = np.linalg.cond(Interp_loc_GtC)
    condCtG = np.linalg.cond(Interp_loc_CtG)

    print(np.linalg.cond(ChebVc), np.linalg.cond(ChebVl), np.linalg.cond(GaussVc), np.linalg.cond(GaussVl))

    return Interp_loc_GtC,Interp_loc_CtG,condGtC,condCtG,croots2d,lroots2d

# Define a function:
def known_function(xx):
    return np.sin(np.pi * xx[0,:]) + xx[1,:]

print("For a sample function sin(pi x) + y on domain [-1,1]^2:")


p_list = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
a = 0.25

cheb_error  = []
gauss_error = []
cheb_cond   = []
gauss_cond  = []
for p in p_list:

    q = p-4; l = p#min(p,q) + 20
    Interp_loc_GtC,Interp_loc_CtG,condGtC,condCtG,croots2d,lroots2d = get_loc_interp_3d(p, q, a, l)

    true_cheb  = known_function(croots2d)
    true_gauss = known_function(lroots2d)

    interp_cheb  = Interp_loc_GtC @ true_gauss
    interp_gauss = Interp_loc_CtG @ true_cheb
    #print("\n p = " + str(p) + ":\n")
    #print("The relative error between the true Chebyshev nodes and interpolated from Gaussian is:")
    cheb_error.append(np.linalg.norm(interp_cheb - true_cheb) / np.linalg.norm(true_cheb))
    #print("The relative error between the true Gaussian nodes and interpolated from Chebyshev is:")
    gauss_error.append(np.linalg.norm(interp_gauss - true_gauss) / np.linalg.norm(true_gauss))
    #print("And the condtion numbers are:")
    cheb_cond.append(condGtC)
    gauss_cond.append(condCtG)

plt.semilogy(p_list, cheb_error)
plt.semilogy(p_list, gauss_error)
plt.title("Errors of interpolated vs true values, l=" + str(l))
plt.xlabel("p and q")
plt.ylabel("relative error")
plt.legend(["Gaussian-to-Chebyshev", "Chebyshev-to-Gaussian"])
plt.savefig("plots_interpolation/2Derrors_" + str(l) + "q_is_pm1.png")
plt.show()

plt.semilogy(p_list, cheb_cond)
plt.semilogy(p_list, gauss_cond)
plt.title("Condition numbers of 2D interpolation operators, l=" + str(l))
plt.xlabel("p and q")
plt.ylabel("condition number")
plt.legend(["Gaussian-to-Chebyshev", "Chebyshev-to-Gaussian"])
plt.savefig("plots_interpolation/2Dcond_" + str(l) + "q_is_pm1.png")
plt.show()


"""
# Let's do 1D then extend to 2D:

from numpy.polynomial.chebyshev import chebfit, chebval
p = 10
a = 0.25
q = 8

_, croots = cheb(p-1)
croots    = np.flip(croots)

lcoeff     = np.zeros(q)
lcoeff[-1] = 1
lroots   = legendre.legroots(lcoeff)

# We'll avoid vandermonde matrices altogether, building a Chebyshev-to-Gaussian
# interpolation directly, column by column:
legvals = []

for i in range(croots.shape[0]):
    y       = np.zeros(croots.shape)
    y[i]    = 1
    coeffs  = chebfit(croots, y, p-1)
    legvals.append(chebval(lroots, coeffs))

CtG1D = np.vstack(legvals).T

print(np.linalg.cond(CtG1D))

# Now let's test this with a couple functions:
def sq(x):
    return x*x

print(sq(a*croots))
print(sq(a*lroots))
print(CtG1D @ sq(a*croots))
"""