import numpy as np
from collections import namedtuple
#import scipy
from time import time
import scipy.linalg

from numpy.polynomial  import legendre
from numpy.polynomial.polynomial import polyvander2d
from numpy.polynomial.chebyshev import chebvander, chebvander2d
from numpy.polynomial.legendre import legvander, legvander2d
from scipy.interpolate import interpn

from scipy.linalg import null_space

# Define named tuples for storing partial differential operators (PDOs) and differential schemes (Ds)
# for both 2D and 3D problems, along with indices (JJ) for domain decomposition.
Pdo_2d   = namedtuple('Pdo_2d',['c11','c22','c12', 'c1','c2','c'])
Ds_2d    = namedtuple('Ds_2d', ['D11','D22','D12','D1','D2'])
JJ_2d    = namedtuple('JJ_2d', ['Jl','Jr','Jd','Ju','Jx', 'Jlc', 'Jrc', 'Jdc',
                               'Juc', 'Jxreorder', 'Jxunique', 'Jc', 'Jtot',
                               'unique_in_reorder'])

Pdo_3d = namedtuple('Pdo_3d', ['c11', 'c22', 'c33', 'c12', 'c13', 'c23', 'c1', 'c2', 'c3', 'c'])
Ds_3d  = namedtuple('Ds_3d',  ['D11', 'D22', 'D33', 'D12', 'D13', 'D23', 'D1', 'D2', 'D3'])
JJ_3d  = namedtuple('JJ_3d',  ['Jl', 'Jr', 'Jd', 'Ju', 'Jb', 'Jf', 'Jx', 'Jlc', 'Jrc', 'Jdc',
                               'Juc', 'Jbc', 'Jfc', 'Jxreorder', 'Jxunique', 'Jc', 'Jtot',
                               'unique_in_reorder'])

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

def gauss(p):
    """
    Computes the Gaussian points for a given degree p.
    
    Parameters:
    - p: The polynomial degree
    
    Returns:
    - x: The Gaussian points
    """
    lcoeff     = np.zeros(p+1)
    lcoeff[-1] = 1

    return legendre.legroots(lcoeff)

def diag_mult(diag,M):
    """
    Performs multiplication of a diagonal matrix (represented by a vector) with a matrix.
    
    Parameters:
    - diag: A vector representing the diagonal of a diagonal matrix
    - M: A matrix to be multiplied
    
    Returns:
    - The result of the diagonal matrix multiplied by M
    """
    return (diag * M.T).T

def get_loc_interp(x_cheb, x_cheb_nocorners, q):
    """
    Computes local interpolation matrices from Chebyshev points.
    
    Parameters:
    - x_cheb: Chebyshev points including corner points
    - x_cheb_nocorners: Chebyshev points excluding corner points
    - q: The degree of the polynomial for interpolation
    
    Returns:
    - Interp_loc: Local interpolation matrix
    - err: Norm of the interpolation error
    - cond: Condition number of the interpolation matrix
    """
    Vpoly_cheb = chebvander(x_cheb,q)
    Vpoly_nocorner = chebvander(x_cheb_nocorners,q)

    Interp_loc = np.linalg.lstsq(Vpoly_nocorner.T,Vpoly_cheb.T,rcond=None)[0].T
    err  = np.linalg.norm(Interp_loc @ Vpoly_nocorner - Vpoly_cheb)
    cond = np.linalg.cond(Interp_loc) 
    return Interp_loc,err,cond

def get_loc_interp_2d(p, q, l):
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

    #print(get_legendre_row(0, croots))

    lcoeff     = np.zeros(q+1)
    lcoeff[-1] = 1
    lroots     = legendre.legroots(lcoeff)

    # Vandermonde-based approach:

    # Vandermonde-based approach with Chebyshev expansion coefficients:
    ChebVc = chebvander(croots, l)
    ChebVl = chebvander(lroots, l)

    # Vandermonde-based approach with Gaussian expansion coefficients:
    GaussVc = legvander(croots, q)
    GaussVl = legvander(lroots, q)

    Interp_loc_CtG = np.linalg.lstsq(ChebVc.T,ChebVl.T,rcond=None)[0].T
    Interp_loc_GtC = np.linalg.lstsq(GaussVl.T,GaussVc.T,rcond=None)[0].T

    condGtC = np.linalg.cond(Interp_loc_GtC)
    condCtG = np.linalg.cond(Interp_loc_CtG)
    return Interp_loc_GtC,Interp_loc_CtG,condGtC,condCtG

def get_loc_interp_3d(p, q, l):
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

    # Vandermonde-based approach:

    # Vandermonde-based approach with Chebyshev expansion coefficients:
    ChebVc = chebvander2d(croots2d[0], croots2d[1], (l,l))
    ChebVl = chebvander2d(lroots2d[0], lroots2d[1], (l,l))

    # Vandermonde-based approach with Gaussian expansion coefficients:
    GaussVc = legvander2d(croots2d[0], croots2d[1], (q,q))
    GaussVl = legvander2d(lroots2d[0], lroots2d[1], (q,q))

    Interp_loc_CtG = np.linalg.lstsq(ChebVc.T,ChebVl.T,rcond=None)[0].T
    Interp_loc_GtC = np.linalg.lstsq(GaussVl.T,GaussVc.T,rcond=None)[0].T

    condGtC = np.linalg.cond(Interp_loc_GtC)
    condCtG = np.linalg.cond(Interp_loc_CtG)
    return Interp_loc_GtC,Interp_loc_CtG,condGtC,condCtG

#################################### 2D discretization ##########################################

def cheb_2d(a,p):
    D,xvec = cheb(p-1)
    xvec1 = a[0] * np.flip(xvec)
    xvec2 = a[1] * np.flip(xvec)

    D_axis1 = (1/a[0]) * D
    D_axis2 = (1/a[1]) * D
    I = np.eye(p)

    D1 = -np.kron(D_axis1,I)
    D2 = -np.kron(I,D_axis2)

    D11 = np.kron(D_axis1 @ D_axis1, I)
    D22 = np.kron(I, D_axis2 @ D_axis2)
    D12 = np.kron(D_axis1, D_axis2)

    zz1 = np.repeat(xvec1,p)
    zz2 = np.repeat(xvec2,p).reshape(-1,p).T.flatten()
    zz = np.vstack((zz1,zz2))
    Ds = Ds_2d(D1= D1, D2= D2, D11= D11, D22= D22, D12= D12)
    return zz, Ds 

def gauss_2d(a,p):
    """
    Given polynomial order p and element size a,
    returns Gaussian collocation points on [-a,a]^3
    """
    xvec  = gauss(p) # should this be p? We need p+1 points for a degree p polynomial
    xvec1 = a[0] * xvec
    xvec2 = a[1] * xvec

    # TODO: get the proper components D1, D2, D3, D11, D22, D33, D12, D13, D23
    # Note that Kronecker product is associative
    # Could replace some of these with np.eye(p**2)

    zz1 = np.repeat(xvec1,p)
    zz2 = np.repeat(xvec2,p).reshape(-1,p).T.flatten()
    zz  = np.vstack((zz1,zz2))

    return zz

def leaf_discretization_2d(a,p,q):
    """
    Performs leaf-level discretization for a 3D domain.
    Returns the discretization points, interpolated surface points,
    index arrays for parts of the disc., and the correct operators.
    """
    zz,Ds = cheb_2d(a,p)
    zzG   = gauss_2d(a,q)

    hmin0 = zz[0,p] - zz[0,0]
    hmin1 = zz[1,1] - zz[1,0]

    # Jl, Jr, Jd, Ju are RLDU as expected, with no corners
    # Jc is interior, Jx is all boundaries without corners
    # Jb, Jf are back and front
    Jc0   = np.abs(zz[0,:]) < a[0] - 0.5*hmin0
    Jc1   = np.abs(zz[1,:]) < a[1] - 0.5*hmin1
    Jl    = np.argwhere(np.logical_and(zz[0,:] < - a[0] + 0.5 * hmin0, Jc1))
    Jl    = Jl.copy().reshape((p-2),)
    Jr    = np.argwhere(np.logical_and(zz[0,:] > + a[0] - 0.5 * hmin0, Jc1))
    Jr    = Jr.copy().reshape((p-2),)
    Jd    = np.argwhere(np.logical_and(zz[1,:] < - a[1] + 0.5 * hmin1, Jc0))
    Jd    = Jd.copy().reshape((p-2),)
    Ju    = np.argwhere(np.logical_and(zz[1,:] > + a[1] - 0.5 * hmin1, Jc0))
    Ju    = Ju.copy().reshape((p-2),)

    Jc    = np.argwhere(np.logical_and(Jc0, Jc1))
    Jc    = Jc.copy().reshape((p-2)**2,)
    Jx    = np.concatenate((Jl,Jr,Jd,Ju))

    Jl_corner    = np.argwhere(zz[0,:] < - a[0] + 0.5 * hmin0)
    Jl_corner    = Jl_corner.copy().reshape(p,)
    Jr_corner    = np.argwhere(zz[0,:] > + a[0] - 0.5 * hmin0)
    Jr_corner    = Jr_corner.copy().reshape(p,)
    Jd_corner    = np.argwhere(zz[1,:] < - a[1] + 0.5 * hmin1)
    Jd_corner    = Jd_corner.copy().reshape(p,)
    Ju_corner    = np.argwhere(zz[1,:] > + a[1] - 0.5 * hmin1)
    Ju_corner    = Ju_corner.copy().reshape(p,)

    # TODO: figure out corners / switch to Legendre for this
    Jxreorder = np.concatenate((Jl_corner,Jr_corner,Jd_corner,Ju_corner))
    Jxunique, unique_in_reorder  = np.unique(Jxreorder, return_index=True)
    Jtot  = np.concatenate((Jx,Jc))

    # Take only necessary surface values for Gaussian nodes:
    a_gauss0 = np.max(zzG[0,:])
    a_gauss1 = np.max(zzG[1,:])
    Jl_gauss = np.argwhere(zzG[0,:] < - a_gauss0 + 0.5 * hmin0)
    Jl_gauss = Jl_gauss.copy().reshape(q,)
    Jr_gauss = np.argwhere(zzG[0,:] > + a_gauss0 - 0.5 * hmin0)
    Jr_gauss = Jr_gauss.copy().reshape(q,)
    Jd_gauss = np.argwhere(zzG[1,:] < - a_gauss1 + 0.5 * hmin1)
    Jd_gauss = Jd_gauss.copy().reshape(q,)
    Ju_gauss = np.argwhere(zzG[1,:] > + a_gauss1 - 0.5 * hmin1)
    Ju_gauss = Ju_gauss.copy().reshape(q,)
    Jgauss = np.concatenate((Jl_gauss,Jr_gauss,Jd_gauss,Ju_gauss))
    zzG = zzG.T[Jgauss,:]
    # Need to do a little surface cleaning to make sure the faces of our Gaussian box 
    # line up with the faces of our Chebyshev box:
    zzG[:q,0]      = -a[0]
    zzG[q:2*q,0]   =  a[0]
    zzG[2*q:3*q,1] = -a[1]
    zzG[3*q:4*q,1] =  a[1]

    JJ    = JJ_2d(Jl= Jl, Jr= Jr, Ju= Ju, Jd= Jd, Jx=Jx,
                  Jlc=Jl_corner, Jrc=Jr_corner, Jdc=Jd_corner, Juc=Ju_corner,
                  Jxreorder=Jxreorder, Jxunique=Jxunique, Jc=Jc, Jtot=Jtot,
                  unique_in_reorder = unique_in_reorder)

    hmin = np.min([hmin0, hmin1])
    
    return zz,Ds,JJ,hmin,zzG

#################################### 3D discretization ##########################################

def cheb_3d(a,p):
    """
    Given polynomial order p and element size a,
    returns chebyshev collocation points on [-a,a]^3 and corresponding differentiation operators.
    """
    D,xvec = cheb(p-1) # should this be p? We need p+1 points for a degree p polynomial
    xvec1 = a[0] * np.flip(xvec)
    xvec2 = a[1] * np.flip(xvec)
    xvec3 = a[2] * np.flip(xvec)

    D_axis1 = (1/a[0]) * D
    D_axis2 = (1/a[1]) * D
    D_axis3 = (1/a[2]) * D

    I = np.eye(p)
    D1 = -np.kron(D_axis1, np.kron(I, I))
    D2 = -np.kron(I, np.kron(D_axis2, I))
    D3 = -np.kron(I, np.kron(I, D_axis3))

    D11 = np.kron(D_axis1 @ D_axis1, np.kron(I, I))
    D22 = np.kron(I, np.kron(D_axis2 @ D_axis2, I))
    D33 = np.kron(I, np.kron(I, D_axis3 @ D_axis3))

    D12 = np.kron(D_axis1, np.kron(D_axis2, I))
    D13 = np.kron(D_axis1, np.kron(I, D_axis3))
    D23 = np.kron(I, np.kron(D_axis2, D_axis3))

    zz1 = np.repeat(xvec1,p**2)
    zz2 = np.repeat(xvec2,p**2).reshape(-1,p).T.flatten()
    zz3 = np.repeat(xvec3,p**2).reshape(-1,p**2).T.flatten()
    zz  = np.vstack((zz1,zz2,zz3))
    Ds  = Ds_3d(D1=D1, D2=D2, D3=D3, D11=D11, D22=D22, D33=D33, D12=D12, D13=D13, D23=D23)

    return zz, Ds

def gauss_3d(a,p):
    """
    Given polynomial order p and element size a,
    returns Gaussian collocation points on [-a,a]^3
    """
    xvec  = gauss(p) # should this be p? We need p+1 points for a degree p polynomial
    xvec1 = a[0] * xvec
    xvec2 = a[1] * xvec
    xvec3 = a[2] * xvec

    # TODO: get the proper components D1, D2, D3, D11, D22, D33, D12, D13, D23
    # Note that Kronecker product is associative
    # Could replace some of these with np.eye(p**2)

    zz1 = np.repeat(xvec1,p**2)
    zz2 = np.repeat(xvec2,p**2).reshape(-1,p).T.flatten()
    zz3 = np.repeat(xvec3,p**2).reshape(-1,p**2).T.flatten()
    zz  = np.vstack((zz1,zz2,zz3))

    return zz

def leaf_discretization_3d(a,p,q):
    """
    Performs leaf-level discretization for a 3D domain.
    Returns the discretization points, interpolated surface points,
    index arrays for parts of the disc., and the correct operators.
    """
    zz,Ds = cheb_3d(a,p)
    zzG   = gauss_3d(a,q)

    hmin0 = zz[0,p**2] - zz[0,0]
    hmin1 = zz[1,p]    - zz[1,0]
    hmin2 = zz[2,1]    - zz[2,0]

    # Jl, Jr, Jd, Ju are RLDU as expected, with no corners
    # Jc is interior, Jx is all boundaries without corners
    # Jb, Jf are back and front
    Jc0   = np.abs(zz[0,:]) < a[0] - 0.5*hmin0
    Jc1   = np.abs(zz[1,:]) < a[1] - 0.5*hmin1
    Jc2   = np.abs(zz[2,:]) < a[2] - 0.5*hmin2
    Jl    = np.argwhere(np.logical_and(zz[0,:] < - a[0] + 0.5 * hmin0,
                                            np.logical_and(Jc1,Jc2)))
    Jl    = Jl.copy().reshape((p-2)**2,)
    Jr    = np.argwhere(np.logical_and(zz[0,:] > + a[0] - 0.5 * hmin0,
                                            np.logical_and(Jc1,Jc2)))
    Jr    = Jr.copy().reshape((p-2)**2,)
    Jd    = np.argwhere(np.logical_and(zz[1,:] < - a[1] + 0.5 * hmin1,
                                            np.logical_and(Jc0,Jc2)))
    Jd    = Jd.copy().reshape((p-2)**2,)
    Ju    = np.argwhere(np.logical_and(zz[1,:] > + a[1] - 0.5 * hmin1,
                                            np.logical_and(Jc0,Jc2)))
    Ju    = Ju.copy().reshape((p-2)**2,)
    Jb    = np.argwhere(np.logical_and(zz[2,:] < - a[2] + 0.5 * hmin2,
                                            np.logical_and(Jc0,Jc1)))
    Jb    = Jb.copy().reshape((p-2)**2,)
    Jf    = np.argwhere(np.logical_and(zz[2,:] > + a[2] - 0.5 * hmin2,
                                            np.logical_and(Jc0,Jc1)))
    Jf    = Jf.copy().reshape((p-2)**2,)
    Jc    = np.argwhere(np.logical_and(Jc0, np.logical_and(Jc1,Jc2)))
    Jc    = Jc.copy().reshape((p-2)**3,)
    Jx    = np.concatenate((Jl,Jr,Jd,Ju,Jb,Jf))

    Jl_corner    = np.argwhere(zz[0,:] < - a[0] + 0.5 * hmin0)
    Jl_corner    = Jl_corner.copy().reshape(p**2,)
    Jr_corner    = np.argwhere(zz[0,:] > + a[0] - 0.5 * hmin0)
    Jr_corner    = Jr_corner.copy().reshape(p**2,)
    Jd_corner    = np.argwhere(zz[1,:] < - a[1] + 0.5 * hmin1)
    Jd_corner    = Jd_corner.copy().reshape(p**2,)
    Ju_corner    = np.argwhere(zz[1,:] > + a[1] - 0.5 * hmin1)
    Ju_corner    = Ju_corner.copy().reshape(p**2,)
    Jb_corner    = np.argwhere(zz[2,:] < - a[2] + 0.5 * hmin2)
    Jb_corner    = Jb_corner.copy().reshape(p**2,)
    Jf_corner    = np.argwhere(zz[2,:] > + a[2] - 0.5 * hmin2)
    Jf_corner    = Jf_corner.copy().reshape(p**2,)

    # TODO: figure out corners / switch to Legendre for this
    Jxreorder = np.concatenate((Jl_corner,Jr_corner,Jd_corner,Ju_corner,Jb_corner,Jf_corner))
    Jxunique, unique_in_reorder  = np.unique(Jxreorder, return_index=True)
    Jtot  = np.concatenate((Jx,Jc))

    # Take only necessary surface values for Gaussian nodes:
    a_gauss0 = np.max(zzG[0,:])
    a_gauss1 = np.max(zzG[1,:])
    a_gauss2 = np.max(zzG[2,:])
    Jl_gauss = np.argwhere(zzG[0,:] < - a_gauss0 + 0.5 * hmin0)
    Jl_gauss = Jl_gauss.copy().reshape(q**2,)
    Jr_gauss = np.argwhere(zzG[0,:] > + a_gauss0 - 0.5 * hmin0)
    Jr_gauss = Jr_gauss.copy().reshape(q**2,)
    Jd_gauss = np.argwhere(zzG[1,:] < - a_gauss1 + 0.5 * hmin1)
    Jd_gauss = Jd_gauss.copy().reshape(q**2,)
    Ju_gauss = np.argwhere(zzG[1,:] > + a_gauss1 - 0.5 * hmin1)
    Ju_gauss = Ju_gauss.copy().reshape(q**2,)
    Jb_gauss = np.argwhere(zzG[2,:] < - a_gauss2 + 0.5 * hmin2)
    Jb_gauss = Jb_gauss.copy().reshape(q**2,)
    Jf_gauss = np.argwhere(zzG[2,:] > + a_gauss2 - 0.5 * hmin2)
    Jf_gauss = Jf_gauss.copy().reshape(q**2,)
    Jgauss = np.concatenate((Jl_gauss,Jr_gauss,Jd_gauss,Ju_gauss,Jb_gauss,Jf_gauss))
    zzG = zzG.T[Jgauss,:]
    # Need to do a little surface cleaning to make sure the faces of our Gaussian box 
    # line up with the faces of our Chebyshev box:
    zzG[:q**2,0]         = -a[0]
    zzG[q**2:2*q**2,0]   =  a[0]
    zzG[2*q**2:3*q**2,1] = -a[1]
    zzG[3*q**2:4*q**2,1] =  a[1]
    zzG[4*q**2:5*q**2,2] = -a[2]
    zzG[5*q**2:,2]       =  a[2]

    JJ    = JJ_3d(Jl= Jl, Jr= Jr, Ju= Ju, Jd= Jd, Jb= Jb, Jf=Jf, Jx=Jx,
                  Jlc=Jl_corner, Jrc=Jr_corner, Jdc=Jd_corner,
                  Juc=Ju_corner, Jbc=Jb_corner, Jfc=Jf_corner,
                  Jxreorder=Jxreorder, Jxunique=Jxunique, Jc=Jc, Jtot=Jtot,
                  unique_in_reorder = unique_in_reorder)

    hmin = np.min([hmin0, hmin1, hmin2])
    
    return zz,Ds,JJ,hmin,zzG

def get_diff_ops(Ds,JJ,d):
    """
    Forms Neumann derivative matrix.
    """
    if (d == 2):
        Nl = Ds.D1[JJ.Jl]
        Nr = Ds.D1[JJ.Jr]
        Nd = Ds.D2[JJ.Jd]
        Nu = Ds.D2[JJ.Ju]

        Nlc = Ds.D1[JJ.Jlc]
        Nrc = Ds.D1[JJ.Jrc]
        Ndc = Ds.D2[JJ.Jdc]
        Nuc = Ds.D2[JJ.Juc]

        Nx  = np.concatenate((-Nl,+Nr,-Nd,+Nu))
        Nxc = np.concatenate((-Nlc,+Nrc,-Ndc,+Nuc))
    else: # Need to include corners here...
        Nl = Ds.D1[JJ.Jl]
        Nr = Ds.D1[JJ.Jr]
        Nd = Ds.D2[JJ.Jd]
        Nu = Ds.D2[JJ.Ju]
        Nb = Ds.D3[JJ.Jb]
        Nf = Ds.D3[JJ.Jf]

        Nlc = Ds.D1[JJ.Jlc]
        Nrc = Ds.D1[JJ.Jrc]
        Ndc = Ds.D2[JJ.Jdc]
        Nuc = Ds.D2[JJ.Juc]
        Nbc = Ds.D3[JJ.Jbc]
        Nfc = Ds.D3[JJ.Jfc]

        Nx  = np.concatenate((-Nl,+Nr,-Nd,+Nu,-Nb,+Nf))
        Nxc = np.concatenate((-Nlc,+Nrc,-Ndc,+Nuc,-Nbc,+Nfc))
    return Nx, Nxc

#################################### HPS Object ##########################################

class HPS_Disc:
    def __init__(self,a,p,q,d):
        """
        Initializes the HPS discretization class.
        
        Parameters:
        - a: Half the size of the computational domain
        - p: The polynomial degree for Chebyshev discretization
        - d: Dimension of the problem (2 or 3)
        """

        self._discretize(a,p,q,d)
        self.a = a; self.p = p; self.q = q; self.d = d
        self._get_interp_mat()
        
    def _discretize(self,a,p,q,d):
        if d == 2:
            self.zz, self.Ds, self.JJ, self.hmin, self.zzG = leaf_discretization_2d(a, p, q)
        else:
            self.zz, self.Ds, self.JJ, self.hmin, self.zzG = leaf_discretization_3d(a, p, q) # Need to figure out a
        Nx, Nxc = get_diff_ops(self.Ds, self.JJ, d)
        self.Nx = Nx; self.Nxc = Nxc
        
    def _get_interp_mat(self):
        """
        Interpolation from data on Ix to Ix_reorder (aka Chebyshev to dropped corners / gaussian)
        """
        
        p = self.p; q = self.q; a = self.a
        
        if self.d==2:
            tic = time()
            l = p
            Interp_loc_GtC,Interp_loc_CtG,err,cond = get_loc_interp_2d(p, q, l)
            self.Interp_mat         = scipy.linalg.block_diag(*np.repeat(np.expand_dims(Interp_loc_GtC,0),4,axis=0))
            self.Interp_mat_reverse = scipy.linalg.block_diag(*np.repeat(np.expand_dims(Interp_loc_CtG,0),4,axis=0))

            # Form averaging operator P. Currently we have two approaches, one is local averaging (P)
            # and the other is orthogonal projection (Pnew):
            P = np.eye(self.Interp_mat.shape[0])
            B = np.eye(self.Interp_mat.shape[0])
            for i in range(self.Interp_mat.shape[0]):
                elem = self.JJ.Jxreorder[i]
                where = np.argwhere(self.JJ.Jxreorder == elem)
                P[i,where] = 1 / len(where)
                B[i,where] = 1

            B       = B[self.JJ.unique_in_reorder,:]
            Bsum    = np.sum(B, axis=1)
            # Identify the corner points:
            indexer = np.argwhere(Bsum > 1)
            B       = B[indexer.T[0]]

            # Try this instead:
            V_null = null_space(B)
            Pnew = V_null @ np.transpose(V_null)
            
            # Apply this to our interpolation matrix to ensure continuity at corner nodes:
            self.Interp_mat        = Pnew @ self.Interp_mat    # with redundant corners
            self.Interp_mat_unique = self.Interp_mat[self.JJ.unique_in_reorder,:] # without redundant corners

            toc = time() - tic
            print ("--Interp_mat has GtC condition number %5.5f, CtG condition number %5.5e, and time to calculate %12.5f"\
                % (cond,err,toc))

        else:
            tic = time()
            l = p #min(p,q) + 10
            Interp_loc_GtC,Interp_loc_CtG,err,cond = get_loc_interp_3d(p, q, l)
            self.Interp_mat         = scipy.linalg.block_diag(*np.repeat(np.expand_dims(Interp_loc_GtC,0),6,axis=0))
            self.Interp_mat_reverse = scipy.linalg.block_diag(*np.repeat(np.expand_dims(Interp_loc_CtG,0),6,axis=0))

            # Form averaging operator P. Currently we have two approaches, one is local averaging (P)
            # and the other is orthogonal projection (Pnew):
            P = np.eye(self.Interp_mat.shape[0])
            B = np.eye(self.Interp_mat.shape[0])
            for i in range(self.Interp_mat.shape[0]):
                elem = self.JJ.Jxreorder[i]
                where = np.argwhere(self.JJ.Jxreorder == elem)
                P[i,where] = 1 / len(where)
                B[i,where] = 1

            B       = B[self.JJ.unique_in_reorder,:]
            Bsum    = np.sum(B, axis=1)
            indexer = np.argwhere(Bsum > 1)
            B       = B[indexer.T[0]]

            # Identify the corner points:
            Bsum = np.sum(B, axis=1)
            indexer = np.argwhere(Bsum > 2)
            indexer = indexer.T[0]

            # Convert the corner points into two rows each:
            for index in indexer:
                where = np.argwhere(B[index] > 0)
                where = where.T[0]
                # Add an extra row to B:
                B = np.vstack((B, np.zeros((1, B.shape[1]))))
                # Now set the new row to two entries, and zero out one of the entries in the first corner row:
                B[-1, where[0]] = 1
                B[-1, where[2]] = 1
                B[index, where[1]] = 0

            # And lastly... convert the second nonzero of every row to -1 so it makes sense:
            for index in range(B.shape[0]):
                where = np.argwhere(B[index] > 0)
                where = where.T[0]
                B[index, where[1]] = -1

            # Try this instead:
            V_null = null_space(B)
            Pnew = V_null @ np.transpose(V_null)
            
            # Apply this to our interpolation matrix to ensure continuity at corner nodes:
            self.Interp_mat        = Pnew @ self.Interp_mat    # with redundant corners
            self.Interp_mat_unique = self.Interp_mat[self.JJ.unique_in_reorder,:] # without redundant corners

            toc = time() - tic
            print ("--Interp_mat has GtC condition number %5.5f, CtG condition number %5.5e, and time to calculate %12.5f"\
                % (cond,err,toc))
