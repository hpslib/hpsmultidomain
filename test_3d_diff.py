
# Let's test some differential operators in 3D

import numpy as np

from collections import namedtuple

Ds_3d  = namedtuple('Ds_3d',  ['D11', 'D22', 'D33', 'D12', 'D13', 'D23', 'D1', 'D2', 'D3'])

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

###
# Given polynomial order p and element size a,
# Returns chebyshev collocation points on [-a,a]^3 and corresponding differentiation operators.
###
def cheb_3d(a,p):
    D,xvec = cheb(p-1) # should this be p? We need p+1 points for a degree p polynomial
    xvec = a * np.flip(xvec)

    # TODO: get the proper components D1, D2, D3, D11, D22, D33, D12, D13, D23
    # Note that Kronecker product is associative
    # Could replace some of these with np.eye(p**2)
    D = (1/a) * D
    I = np.eye(p)
    D1 = -np.kron(D, np.kron(I, I))
    D2 = -np.kron(I, np.kron(D, I))
    D3 = -np.kron(I, np.kron(I, D))
    Dsq = D @ D
    D11 = np.kron(Dsq, np.kron(I, I))
    D22 = np.kron(I, np.kron(Dsq, I))
    D33 = np.kron(I, np.kron(I, Dsq))

    D12 = np.kron(D, np.kron(D, I))
    D13 = np.kron(D, np.kron(I, D))
    D23 = np.kron(I, np.kron(D, D))

    zz1 = np.repeat(xvec,p**2)
    zz2 = np.repeat(xvec,p**2).reshape(-1,p).T.flatten()
    zz3 = np.repeat(xvec,p**2).reshape(-1,p**2).T.flatten()
    zz  = np.vstack((zz1,zz2,zz3))
    Ds  = Ds_3d(D1=D1, D2=D2, D3=D3, D11=D11, D22=D22, D33=D33, D12=D12, D13=D13, D23=D23)

    return zz, Ds

a = 0.1
p = 12

zz, Ds = cheb_3d(a,p)
k = 1

def u_true(zz):
    return np.exp(k*zz[0]) + np.sin(k*zz[1]) * np.cos(k*zz[2])
    #return np.exp(zz[2]) + np.sin(zz[1]) * np.cos(zz[0])

def du1_true(zz):
    return k*np.exp(k*zz[0])
    #return np.sin(zz[1]) * -np.sin(zz[0])

def du2_true(zz):
    return k*np.cos(k*zz[1]) * np.cos(k*zz[2])
    #return np.cos(zz[1]) * np.cos(zz[0])

def du3_true(zz):
    return np.sin(k*zz[1]) * -k*np.sin(k*zz[2])
    #return np.exp(zz[2])

u = u_true(zz)

du1 = Ds.D1 @ u
du2 = Ds.D2 @ u
du3 = Ds.D3 @ u

du1_t = du1_true(zz)
du2_t = du2_true(zz)
du3_t = du3_true(zz)

du1_rel_error = np.linalg.norm(du1 - du1_t) / np.linalg.norm(du1_t)
du2_rel_error = np.linalg.norm(du2 - du2_t) / np.linalg.norm(du2_t)
du3_rel_error = np.linalg.norm(du3 - du3_t) / np.linalg.norm(du3_t)

print("Relative error for du1 is:" + str(du1_rel_error))
print("Relative error for du2 is:" + str(du2_rel_error))
print("Relative error for du3 is:" + str(du3_rel_error))

# Now let's try second order derivatives:

def du11_true(zz):
    return (k**2)*np.exp(k*zz[0])
    #return np.sin(zz[1]) * -np.cos(zz[0])

def du22_true(zz):
    return -(k**2)*np.sin(k*zz[1]) * np.cos(k*zz[2])
    #return -np.sin(zz[1]) * np.cos(zz[0])

def du33_true(zz):
    return np.sin(k*zz[1]) * -(k**2)*np.cos(k*zz[2])
    #return np.exp(zz[2])

u = u_true(zz)

du11 = Ds.D11 @ u
du22 = Ds.D22 @ u
du33 = Ds.D33 @ u

du11_t = du11_true(zz)
du22_t = du22_true(zz)
du33_t = du33_true(zz)

du11_rel_error = np.linalg.norm(du11 - du11_t) / np.linalg.norm(du11_t)
du22_rel_error = np.linalg.norm(du22 - du22_t) / np.linalg.norm(du22_t)
du33_rel_error = np.linalg.norm(du33 - du33_t) / np.linalg.norm(du33_t)

print("Relative error for du11 is:" + str(du11_rel_error))
print("Relative error for du22 is:" + str(du22_rel_error))
print("Relative error for du33 is:" + str(du33_rel_error))

# Lastly, let's try mixed derivatives:

def du12_true(zz):
    return 0*np.exp(k*zz[0])

def du13_true(zz):
    return 0*np.exp(k*zz[0])

def du23_true(zz):
    return (k**2)*np.cos(k*zz[1]) * -np.sin(k*zz[2])

u = u_true(zz)

du12 = Ds.D12 @ u
du13 = Ds.D13 @ u
du23 = Ds.D23 @ u

du12_t = du12_true(zz)
du13_t = du13_true(zz)
du23_t = du23_true(zz)

du12_rel_error = np.linalg.norm(du12 - du12_t) / (np.linalg.norm(du12_t) + 1)
du13_rel_error = np.linalg.norm(du13 - du13_t) / (np.linalg.norm(du13_t) + 1)
du23_rel_error = np.linalg.norm(du23 - du23_t) / (np.linalg.norm(du23_t) + 1)

print("Relative error for du12 is:" + str(du12_rel_error))
print("Relative error for du13 is:" + str(du13_rel_error))
print("Relative error for du23 is:" + str(du23_rel_error))