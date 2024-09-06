import numpy as np
from collections import namedtuple
#import scipy
from time import time
import numpy.polynomial.chebyshev as cheb_py
import scipy.linalg

from numpy.polynomial  import legendre
from numpy.polynomial.polynomial import polyvander2d
from numpy.polynomial.chebyshev import chebvander2d
from numpy.polynomial.legendre import legvander2d
from scipy.interpolate import interpn

from scipy.linalg import null_space

# Define named tuples for storing partial differential operators (PDOs) and differential schemes (Ds)
# for both 2D and 3D problems, along with indices (JJ) for domain decomposition.
Pdo_2d   = namedtuple('Pdo_2d',['c11','c22','c12', 'c1','c2','c'])
Ds_2d    = namedtuple('Ds_2d', ['D11','D22','D12','D1','D2'])
JJ_2d    = namedtuple('JJ_2d', ['Jl','Jr','Jd','Ju','Jx','Jc','Jxreorder'])

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
    Vpoly_cheb = cheb_py.chebvander(x_cheb,q)
    Vpoly_nocorner = cheb_py.chebvander(x_cheb_nocorners,q)

    Interp_loc = np.linalg.lstsq(Vpoly_nocorner.T,Vpoly_cheb.T,rcond=None)[0].T
    err  = np.linalg.norm(Interp_loc @ Vpoly_nocorner - Vpoly_cheb)
    cond = np.linalg.cond(Interp_loc) 
    return Interp_loc,err,cond

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
    croots     = a * np.flip(croots)
    croots2d   = np.array([np.repeat(croots, p), np.hstack([croots]*p)])
    lcoeff     = np.zeros(q+1)
    lcoeff[-1] = 1

    #print(get_legendre_row(0, croots))

    lroots   = a * legendre.legroots(lcoeff)
    #lroots[1:-1] = croots[1:-1]
    lroots2d = np.array([np.repeat(lroots, q), np.hstack([lroots]*q)])

    # Vandermonde-based approach:

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
    return Interp_loc_GtC,Interp_loc_CtG,condGtC,condCtG

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

def leaf_discretization_3d(a,p,q):
    """
    Performs leaf-level discretization for a 3D domain.
    """
    zz,Ds = cheb_3d(a,p)
    zzG   = gauss_3d(a,q)
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
    Jxunique, unique_in_reorder  = np.unique(Jxreorder, return_index=True)
    Jtot  = np.concatenate((Jx,Jc))

    # Take only necessary surface values for Gaussian nodes:
    a_gauss = np.max(np.concatenate((zzG[0,:], zzG[1,:], zzG[2,:])))
    Jl_gauss = np.argwhere(zzG[0,:] < - a_gauss + 0.5 * hmin)
    Jl_gauss = Jl_gauss.copy().reshape(q**2,)
    Jr_gauss = np.argwhere(zzG[0,:] > + a_gauss - 0.5 * hmin)
    Jr_gauss = Jr_gauss.copy().reshape(q**2,)
    Jd_gauss = np.argwhere(zzG[1,:] < - a_gauss + 0.5 * hmin)
    Jd_gauss = Jd_gauss.copy().reshape(q**2,)
    Ju_gauss = np.argwhere(zzG[1,:] > + a_gauss - 0.5 * hmin)
    Ju_gauss = Ju_gauss.copy().reshape(q**2,)
    Jb_gauss = np.argwhere(zzG[2,:] < - a_gauss + 0.5 * hmin)
    Jb_gauss = Jb_gauss.copy().reshape(q**2,)
    Jf_gauss = np.argwhere(zzG[2,:] > + a_gauss - 0.5 * hmin)
    Jf_gauss = Jf_gauss.copy().reshape(q**2,)
    Jgauss = np.concatenate((Jl_gauss,Jr_gauss,Jd_gauss,Ju_gauss,Jb_gauss,Jf_gauss))
    zzG = zzG.T[Jgauss,:]
    # Need to do a little surface cleaning to make sure the faces of our Gaussian box 
    # line up with the faces of our Chebyshev box:
    zzG[:q**2,0]         = -a
    zzG[q**2:2*q**2,0]   =  a
    zzG[2*q**2:3*q**2,1] = -a
    zzG[3*q**2:4*q**2,1] =  a
    zzG[4*q**2:5*q**2,2] = -a
    zzG[5*q**2:,2]       =  a

    JJ    = JJ_3d(Jl= Jl, Jr= Jr, Ju= Ju, Jd= Jd, Jb= Jb, Jf=Jf, Jx=Jx,
                  Jlc=Jl_corner, Jrc=Jr_corner, Jdc=Jd_corner,
                  Juc=Ju_corner, Jbc=Jb_corner, Jfc=Jf_corner,
                  Jxreorder=Jxreorder, Jxunique=Jxunique, Jc=Jc, Jtot=Jtot,
                  unique_in_reorder = unique_in_reorder)
    return zz,Ds,JJ,hmin,zzG

def get_diff_ops(Ds,JJ,d):
    if (d == 2):
        Nl = Ds.D1[JJ.Jl]
        Nr = Ds.D1[JJ.Jr]
        Nd = Ds.D2[JJ.Jd]
        Nu = Ds.D2[JJ.Ju]

        Nx  = np.concatenate((-Nl,+Nr,-Nd,+Nu))
        Nxc = Nx
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
            self.zz, self.Ds, self.JJ, self.hmin = leaf_discretization_2d(a, p)
        else:
            self.zz, self.Ds, self.JJ, self.hmin, self.zzG = leaf_discretization_3d(a, p, q)
        Nx, Nxc = get_diff_ops(self.Ds, self.JJ, d)
        self.Nx = Nx; self.Nxc = Nxc
        
    ## Interpolation from data on Ix to Ix_reorder
    def _get_interp_mat(self):
        
        p = self.p; q = self.q; a = self.a
        
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
            l = min(p,q) + 20
            Interp_loc_GtC,Interp_loc_CtG,err,cond = get_loc_interp_3d(p, q, a, l)
            self.Interp_mat         = scipy.linalg.block_diag(*np.repeat(np.expand_dims(Interp_loc_GtC,0),6,axis=0))
            self.Interp_mat_reverse = scipy.linalg.block_diag(*np.repeat(np.expand_dims(Interp_loc_CtG,0),6,axis=0))

            # Form averaging operator P
            P = np.eye(self.Interp_mat.shape[0])
            B = np.eye(self.Interp_mat.shape[0])
            for i in range(self.Interp_mat.shape[0]):
                elem = self.JJ.Jxreorder[i]
                where = np.argwhere(self.JJ.Jxreorder == elem)
                P[i,where] = 1 / len(where)
                B[i,where] = 1

            B = B[self.JJ.unique_in_reorder,:]
            Bsum = np.sum(B, axis=1)
            indexer = np.argwhere(Bsum > 1)

            B = B[indexer.T[0]]

            # PROBLEM WITH B: CORNERS. CORNERS ARE CURRENTLY UNDERDETERMINED... THEY NEED TWO ROWS EACH, NOT ONE
            # BACK TO BUILDING IT ROW BY ROW

            # Pretty sure this B is right... next is finding its null space
            """_, _, Vh = np.linalg.svd(B, full_matrices=True)
            V = np.transpose(Vh)
            # The nullspace of B is the last _ columns of V:
            null_rank = 6*p**2 - 12*p + 8
            V_tilde = V[:,-null_rank:]
            Pnew = V_tilde @ np.transpose(V_tilde)"""

            # Try this instead:
            V_null = null_space(B)
            Pnew = V_null @ np.transpose(V_null)

            print(V_null.shape, Pnew.shape)
            
            # Apply this to our interpolation matrix to ensure continuity at corner nodes:
            self.Interp_mat        = Pnew @ self.Interp_mat    # with redundant corners
            self.Interp_mat_unique = self.Interp_mat[self.JJ.unique_in_reorder,:] # without redundant corners

            toc = time() - tic
            print ("--Interp_mat has GtC condition number %5.5f, CtG condition number %5.5e, and time to calculate %12.5f"\
                % (cond,err,toc))
