import numpy as np
from collections import namedtuple
import scipy
from time import time
import numpy.polynomial.chebyshev as cheb_py
import scipy.linalg

from numpy.polynomial  import legendre
from numpy.polynomial.polynomial import polyvander2d
from scipy.interpolate import interpn

# Define named tuples for storing partial differential operators (PDOs) and differential schemes (Ds)
# for both 2D and 3D problems, along with indices (JJ) for domain decomposition.
Pdo_2d   = namedtuple('Pdo_2d',['c11','c22','c12', 'c1','c2','c'])
Ds_2d    = namedtuple('Ds_2d', ['D11','D22','D12','D1','D2'])
JJ_2d    = namedtuple('JJ_2d', ['Jl','Jr','Jd','Ju','Jx','Jc','Jxreorder'])

Pdo_3d = namedtuple('Pdo_3d', ['c11', 'c22', 'c33', 'c12', 'c13', 'c23', 'c1', 'c2', 'c3', 'c'])
Ds_3d  = namedtuple('Ds_3d',  ['D11', 'D22', 'D33', 'D12', 'D13', 'D23', 'D1', 'D2', 'D3'])
JJ_3d  = namedtuple('JJ_3d',  ['Jl', 'Jr', 'Jd', 'Ju', 'Jb', 'Jf', 'Jx', 'Jxreorder', 'Jc', 'Jtot'])

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
    Vpoly_cheb = cheb_py.chebvander(x_cheb,q)
    Vpoly_nocorner = cheb_py.chebvander(x_cheb_nocorners,q)

    Interp_loc = np.linalg.lstsq(Vpoly_nocorner.T,Vpoly_cheb.T,rcond=None)[0].T
    err  = np.linalg.norm(Interp_loc @ Vpoly_nocorner - Vpoly_cheb)
    cond = np.linalg.cond(Interp_loc) 
    return Interp_loc,err,cond

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

    
    Interp_loc = []
    for i in range(p):
        for j in range(p):
            values[:,:] = 0
            values[i,j] = 1
            Interp_loc.append(interpn(cpoints, values, lroots2d.T, method='pchip'))

    #Interp_loc = np.column_stack(Interp_loc)
    

    # Vandermonde-based approach:
    Vc = polyvander2d(croots2d[0], croots2d[1], (p,p))
    Vl = polyvander2d(lroots2d[0], lroots2d[1], (q,q))

    Interp_loc = np.linalg.lstsq(Vl.T,Vc.T,rcond=None)[0].T

    cond = np.linalg.cond(Interp_loc)
    # TODO: get err
    err = 3.14159
    return Interp_loc,err,cond

#################################### 2D discretization ##########################################

def cheb_2d(a,p):
    D,xvec = cheb(p-1)
    xvec = a * np.flip(xvec)
    D = (1/a) * D
    I = np.eye(p)
    D1 = -np.kron(D,I)
    D2 = -np.kron(I,D)
    Dsq = D @ D
    D11 = np.kron(Dsq,I)
    D22 = np.kron(I,Dsq)
    D12 = np.kron(D,D)

    zz1 = np.repeat(xvec,p)
    zz2 = np.repeat(xvec,p).reshape(-1,p).T.flatten()
    zz = np.vstack((zz1,zz2))
    Ds = Ds_2d(D1= D1, D2= D2, D11= D11, D22= D22, D12= D12)
    return zz, Ds 

def leaf_discretization_2d(a,p):
    zz,Ds = cheb_2d(a,p)
    hmin  = zz[1,1] - zz[0,1]

    Jc0   = np.abs(zz[0,:]) < a - 0.5*hmin
    Jc1   = np.abs(zz[1,:]) < a - 0.5*hmin
    Jl    = np.argwhere(np.logical_and(zz[0,:] < - a + 0.5 * hmin,Jc1))
    Jl    = Jl.copy().reshape(p-2,)
    Jr    = np.argwhere(np.logical_and(zz[0,:] > + a - 0.5 * hmin,Jc1))
    Jr    = Jr.copy().reshape(p-2,)
    Jd    = np.argwhere(np.logical_and(zz[1,:] < - a + 0.5 * hmin,Jc0))
    Jd    = Jd.copy().reshape(p-2,)
    Ju    = np.argwhere(np.logical_and(zz[1,:] > + a - 0.5 * hmin,Jc0))
    Ju    = Ju.copy().reshape(p-2,)
    Jc    = np.argwhere(np.logical_and(Jc0,Jc1))
    Jc    = Jc.copy().reshape((p-2)**2,)
    Jx    = np.concatenate((Jl,Jr,Jd,Ju))

    Jcorner = np.setdiff1d(np.arange(p**2),np.hstack((Jc,Jx)))

    Jl_corner = np.hstack((Jcorner[0],   Jl))
    Ju_corner = np.hstack((Jcorner[0+1], Ju))
    Jr_corner = np.hstack((Jcorner[0+3], np.flip(Jr,0)))
    Jd_corner = np.hstack((Jcorner[0+2], np.flip(Jd,0)))

    Jxreorder = np.hstack((Jl_corner,Ju_corner,Jr_corner,Jd_corner))

    JJ    = JJ_2d(Jl= Jl, Jr= Jr, Ju= Ju, Jd= Jd, 
              Jx= Jx, Jc= Jc, Jxreorder=Jxreorder)
    return zz,Ds,JJ,hmin

#################################### 3D discretization ##########################################

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

###
# Given polynomial order p and element size a,
# Returns Gaussian collocation points on [-a,a]^3
###
def gauss_3d(a,p):
    xvec = gauss(p) # should this be p? We need p+1 points for a degree p polynomial
    xvec = a * xvec

    # TODO: get the proper components D1, D2, D3, D11, D22, D33, D12, D13, D23
    # Note that Kronecker product is associative
    # Could replace some of these with np.eye(p**2)

    zz1 = np.repeat(xvec,p**2)
    zz2 = np.repeat(xvec,p**2).reshape(-1,p).T.flatten()
    zz3 = np.repeat(xvec,p**2).reshape(-1,p**2).T.flatten()
    zz  = np.vstack((zz1,zz2,zz3))

    return zz

def leaf_discretization_3d(a,p):
    """
    Performs leaf-level discretization for a 3D domain.
    """
    zz,Ds = cheb_3d(a,p)
    zzG   = gauss_3d(a,p)
    hmin  = zz[2,1] - zz[2,0]

    # Jl, Jr, Jd, Ju are RLDU as expected, with no corners
    # Jc is interior, Jx is all boundaries without corners
    # Jb, Jf are back and front
    Jc0   = np.abs(zz[0,:]) < a - 0.5*hmin
    Jc1   = np.abs(zz[1,:]) < a - 0.5*hmin
    Jc2   = np.abs(zz[2,:]) < a - 0.5*hmin
    Jl    = np.argwhere(np.logical_and(zz[0,:] < - a + 0.5 * hmin,
                                            np.logical_and(Jc1,Jc2)))
    Jl    = Jl.copy().reshape((p-2)**2,)
    Jr    = np.argwhere(np.logical_and(zz[0,:] > + a - 0.5 * hmin,
                                            np.logical_and(Jc1,Jc2)))
    Jr    = Jr.copy().reshape((p-2)**2,)
    Jd    = np.argwhere(np.logical_and(zz[1,:] < - a + 0.5 * hmin,
                                            np.logical_and(Jc0,Jc2)))
    Jd    = Jd.copy().reshape((p-2)**2,)
    Ju    = np.argwhere(np.logical_and(zz[1,:] > + a - 0.5 * hmin,
                                            np.logical_and(Jc0,Jc2)))
    Ju    = Ju.copy().reshape((p-2)**2,)
    Jb    = np.argwhere(np.logical_and(zz[2,:] < - a + 0.5 * hmin,
                                            np.logical_and(Jc0,Jc1)))
    Jb    = Jb.copy().reshape((p-2)**2,)
    Jf    = np.argwhere(np.logical_and(zz[2,:] > + a - 0.5 * hmin,
                                            np.logical_and(Jc0,Jc1)))
    Jf    = Jf.copy().reshape((p-2)**2,)
    Jc    = np.argwhere(np.logical_and(Jc0, np.logical_and(Jc1,Jc2)))
    Jc    = Jc.copy().reshape((p-2)**3,)
    Jx    = np.concatenate((Jl,Jr,Jd,Ju,Jb,Jf))

    Jl_corner    = np.argwhere(zz[0,:] < - a + 0.5 * hmin)
    Jl_corner    = Jl_corner.copy().reshape(p**2,)
    Jr_corner    = np.argwhere(zz[0,:] > + a - 0.5 * hmin)
    Jr_corner    = Jr_corner.copy().reshape(p**2,)
    Jd_corner    = np.argwhere(zz[1,:] < - a + 0.5 * hmin)
    Jd_corner    = Jd_corner.copy().reshape(p**2,)
    Ju_corner    = np.argwhere(zz[1,:] > + a - 0.5 * hmin)
    Ju_corner    = Ju_corner.copy().reshape(p**2,)
    Jb_corner    = np.argwhere(zz[2,:] < - a + 0.5 * hmin)
    Jb_corner    = Jb_corner.copy().reshape(p**2,)
    Jf_corner    = np.argwhere(zz[2,:] > + a - 0.5 * hmin)
    Jf_corner    = Jf_corner.copy().reshape(p**2,)

    # TODO: figure out corners / switch to Legendre for this
    Jxreorder = np.concatenate((Jl_corner,Jr_corner,Jd_corner,Ju_corner,Jb_corner,Jf_corner))
    Jtot  = np.concatenate((Jx,Jc))
    JJ    = JJ_3d(Jl= Jl, Jr= Jr, Ju= Ju, Jd= Jd, Jb= Jb,
                  Jf=Jf, Jx=Jx, Jxreorder=Jxreorder, Jc=Jc, Jtot=Jtot)
    return zz,Ds,JJ,hmin,zzG

def get_diff_ops(Ds,JJ,d):
    if (d == 2):
        Nl = Ds.D1[JJ.Jl]
        Nr = Ds.D1[JJ.Jr]
        Nd = Ds.D2[JJ.Jd]
        Nu = Ds.D2[JJ.Ju]

        Nx = np.concatenate((-Nl,+Nr,-Nd,+Nu))
    else:
        Nl = Ds.D1[JJ.Jl][:,JJ.Jtot]
        Nr = Ds.D1[JJ.Jr][:,JJ.Jtot]
        Nd = Ds.D2[JJ.Jd][:,JJ.Jtot]
        Nu = Ds.D2[JJ.Ju][:,JJ.Jtot]
        Nb = Ds.D3[JJ.Jb][:,JJ.Jtot]
        Nf = Ds.D3[JJ.Jf][:,JJ.Jtot]

        Nx = np.concatenate((-Nl,+Nr,-Nd,+Nu,-Nb,+Nf))
    return Nx

#################################### HPS Object ##########################################

class HPS_Disc:
    def __init__(self,a,p,d):
        """
        Initializes the HPS discretization class.
        
        Parameters:
        - a: Half the size of the computational domain
        - p: The polynomial degree for Chebyshev discretization
        - d: Dimension of the problem (2 or 3)
        """
        self._discretize(a,p,d)
        self.a = a; self.p = p; self.d = d
        self._get_interp_mat()
        
    def _discretize(self,a,p,d):
        if d == 2:
            self.zz, self.Ds, self.JJ, self.hmin = leaf_discretization_2d(a, p)
        else:
            self.zz, self.Ds, self.JJ, self.hmin, self.zzG = leaf_discretization_3d(a, p)
        self.Nx = get_diff_ops(self.Ds, self.JJ, d)
        
    ## Interpolation from data on Ix to Ix_reorder
    def _get_interp_mat(self):
        
        p = self.p; a = self.a
        
        if self.d==2:
            x_cheb = self.zz[-1,:p-1] + a
            x_cheb_nocorners  = self.zz[-1,1:p-1] + a
            
            cond_min = 3.25; cond_max = 3.5; err_tol = 2e-8
            q = p+5; tic = time()
            while (True):
                Interp_loc,err,cond = get_loc_interp(x_cheb,x_cheb_nocorners,q)
                
                if ((cond < cond_min) or ((err > err_tol) and (cond < cond_max)) ):
                    break
                else:
                    q += 10
            toc = time() - tic

            #print(Interp_loc)
            #print(Interp_loc.shape)

            # TODO: for 3d, we need to extend this from edge-to-edge to face-to-face. Given p, the final size should be
            # 6*(p-2)^2 + 12*(p) - 8 X 6*(p-2)^2
            
            # Expand the dimensions of Interp_loc once then repeat it four times
            Interp_mat_chebfleg = scipy.linalg.block_diag(*np.repeat(np.expand_dims(Interp_loc,0),4,axis=0))
            #print(Interp_mat_chebfleg.shape)
            # Reorder the columns
            perm = np.hstack((np.arange(p-2),\
                            np.arange(p-2)+3*(p-2),\
                            np.flip(np.arange(p-2)+1*(p-2),0),\
                            np.flip(np.arange(p-2)+2*(p-2),0)
                            ))
            #print(perm)
            perm = np.argsort(perm)
            #print(perm)
            self.Interp_mat = Interp_mat_chebfleg[:,perm]
            #print(self.Interp_mat.shape)
            print ("--Interp_mat required lstsqfit of q=%d, condition number %5.5f with error %5.5e and time to calculate %12.5f"\
                % (q,cond,err,toc))
        else:
            tic = time()
            Interp_loc,err,cond = get_loc_interp_3d(p, p, a)
            self.Interp_mat = scipy.linalg.block_diag(*np.repeat(np.expand_dims(Interp_loc,0),6,axis=0))

            # Form B, then projection operator P = VV^*
            """
            u, c = np.unique(self.JJ.Jxreorder, return_counts=True)
            dup = u[c > 1]
            B = []
            for elem in dup:
                where = np.argwhere(self.JJ.Jxreorder == elem)
                Brows = np.zeros((len(where)-1, self.JJ.Jxreorder.shape[0]))
                Brows[:,where[0]] = 1
                for index in range(len(where[1:])):
                    Brows[index,where[1+index]] = -1
                B.append(Brows)
            
            B = np.vstack(B)
            _, _, Vh = np.linalg.svd(B, full_matrices=True)
            null_rank = B.shape[1] - B.shape[0]
            Vh = Vh[:,-null_rank:]
            P = Vh @ Vh.T
            #print(B)
            #print(B.shape)
            #print(Vh.shape, P.shape)
            """
            P = np.eye(self.Interp_mat.shape[0])
            for i in range(self.Interp_mat.shape[0]):
                elem = self.JJ.Jxreorder[i]
                where = np.argwhere(self.JJ.Jxreorder == elem)
                P[i,where] = 1 / len(where)

            #print(P)
            
            # Apply this to our interpolation matrix to ensure continuity at corner nodes:
            self.Interp_mat = P @ self.Interp_mat

            toc = time() - tic
            print ("--Interp_mat has condition number %5.5f with error %5.5e and time to calculate %12.5f"\
                % (cond,err,toc))
