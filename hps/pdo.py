import torch
torch.set_default_dtype(torch.double)
import numpy as np
import warnings

class PDO_2d:
    """
    Represents a 2-dimensional Partial Differential Operator (PDO) with coefficients for the PDE terms.
    
    Parameters:
    c11, c22, c12: Coefficients for the second-order partial derivatives.
    c1, c2: Coefficients for the first-order derivatives.
    c: Coefficient for the zeroth-order term (source term).
    """
    def __init__(self, c11, c22, c12=None, c1= None, c2 = None, c = None):
        self.c11, self.c22 = c11, c22
        self.c12 = c12
        self.c1, self.c2 = c1, c2
        self.c = c

class PDO_3d:
    """
    Represents a 3-dimensional Partial Differential Operator (PDO) with coefficients for the PDE terms.
    
    Parameters:
    c11, c22, c33, c12, c13, c23: Coefficients for the second-order partial derivatives.
    c1, c2, c3: Coefficients for the first-order derivatives.
    c: Coefficient for the zeroth-order term (source term).
    """
    def __init__(self, c11, c22, c33, c12=None, c13=None, c23=None, c1=None, c2=None, c3=None, c = None):
        self.c11, self.c22, self.c33 = c11, c22, c33
        self.c12, self.c13, self.c23 = c12, c13, c23
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.c = c


        
def ones(xxloc):
    """
    Returns a tensor of ones for a given set of locations. Used for creating constant coefficient functions.
    
    Parameters:
    xxloc: n x d matrix of locations of n points in d dimensions.
    
    Returns:
    A tensor of ones with shape n.
    """
    return torch.ones(xxloc.shape[0], device=xxloc.device)

def const(c=1):
    """
    Returns a function that generates a tensor of constants for a given set of locations.
    
    Parameters:
    c: The constant value to use. Defaults to 1.
    
    Returns:
    A function that takes `xxloc`, an n x d matrix of locations, and returns a tensor of constants c.
    """
    return lambda xxloc: c * torch.ones(xxloc.shape[0], device=xxloc.device)


################################################################################
#################### parameter map for curved domains ##########################

def cii_func(parameter_map,yi_d1, yi_d2, yi_d3=None):

    if ((yi_d1 is None) and (yi_d2 is None) and (yi_d3 is None)):
        cii = None
    else:

        def cii(xx):
            yy = parameter_map(xx)

            result = 0
            if (yi_d1 is not None):
                result += yi_d1(yy)**2
            if (yi_d2 is not None):
                result += yi_d2(yy)**2
            if (yi_d3 is not None):
                result += yi_d3(yy)**2
            return result
    return cii

def ci_func(parameter_map,yi_d1d1, yi_d2d2, yi_d3d3=None):

    if ((yi_d1d1 is None) and (yi_d2d2 is None) and (yi_d3d3 is None)):
        ci = None
    else:

        def ci(xx):
            yy = parameter_map(xx)

            result = 0
            if (yi_d1d1 is not None):
                result -= yi_d1d1(yy)
            if (yi_d2d2 is not None):
                result -= yi_d2d2(yy)
            if (yi_d3d3 is not None):
                result -= yi_d3d3(yy)
            return result
    return ci

def cij_func(parameter_map,yi_d1, yj_d1, yi_d2, yj_d2, yi_d3=None, yj_d3=None):

    bool_expr1 = (yi_d1 is not None) and (yj_d1 is not None)
    bool_expr2 = (yi_d2 is not None) and (yj_d2 is not None)
    bool_expr3 = (yi_d3 is not None) and (yj_d3 is not None)
    if ((not bool_expr1) and (not bool_expr2) and (not bool_expr3)):
        cij = None
    else:

        def cij(xx):
            yy = parameter_map(xx)

            result = 0
            if (bool_expr1):
                result += 2*torch.mul(yi_d1(yy),yj_d1(yy))
            if (bool_expr2):
                result += 2*torch.mul(yi_d2(yy),yj_d2(yy))
            if (bool_expr3):
                result += 2*torch.mul(yi_d3(yy),yj_d3(yy))
            return result
    return cij

def cj_func(parameter_map, trace_JB, J, A, B, j):

    # HERE: check if final result is None. Unlikely.
    bool_expr = True
    if not bool_expr:
        cj = None
    else:
        def cj(xx):
            yy = parameter_map(xx)
            result = 0
            for k in range(3):
                #JBk = J @ B[k] # make function valued
                tr_JBk = trace_JB[k](yy) # make function valued
                for i in range(3):
                    # partial wrt Yk of g^ij
                    dYk_gij  = B[k][i][0](yy) * A[j][0](yy) + A[i][0](yy) * B[k][j][0](yy)
                    dYk_gij += B[k][i][1](yy) * A[j][1](yy) + A[i][1](yy) * B[k][j][1](yy)
                    dYk_gij += B[k][i][2](yy) * A[j][2](yy) + A[i][2](yy) * B[k][j][2](yy)
                    # g^ij itself:
                    gij      = A[i][0](yy)*A[j][0](yy) + A[i][1](yy)*A[j][1](yy) + A[i][2](yy)*A[j][2](yy)
                    result  += J[k][i](yy) * (dYk_gij - gij * tr_JBk)
            return result
    return cj

# Helper to get the trace of JB:
def get_traceJB(J, B):
    traceJB = []
    for k in range(3):
        def make(k):
            def tr_JBK(yy):
                result = 0
                for i in range(3):
                    for j in range(3):
                        result += J[i][j](yy) * B[k][j][i](yy)
                return result
            return tr_JBK
        traceJB.append(make(k))
    return traceJB

def build_J_from_A(A):
    """
    yi_dj[i][m](Y) = ∂x_i/∂Y_m   (i,m in {0,1,2})
    Returns:
      detA(Y), cofA[i][j](Y), J[k][i](Y) = (A^{-1})_{k i} = cofA[i][k](Y) / detA(Y)
    All are function-valued, no grids stored.
    """
    # pull out A_ij as callables for readability
    a11, a12, a13 = A[0][0], A[0][1], A[0][2]
    a21, a22, a23 = A[1][0], A[1][1], A[1][2]
    a31, a32, a33 = A[2][0], A[2][1], A[2][2]

    # determinant
    def detA(Y):
        A11,A12,A13 = a11(Y), a12(Y), a13(Y)
        A21,A22,A23 = a21(Y), a22(Y), a23(Y)
        A31,A32,A33 = a31(Y), a32(Y), a33(Y)
        return A11*(A22*A33 - A23*A32) - A12*(A21*A33 - A23*A31) + A13*(A21*A32 - A22*A31)

    # cofactors (NOT transposed)
    def C11(Y): return  +(a22(Y)*a33(Y) - a23(Y)*a32(Y))
    def C12(Y): return  -(a21(Y)*a33(Y) - a23(Y)*a31(Y))
    def C13(Y): return  +(a21(Y)*a32(Y) - a22(Y)*a31(Y))
    def C21(Y): return  -(a12(Y)*a33(Y) - a13(Y)*a32(Y))
    def C22(Y): return  +(a11(Y)*a33(Y) - a13(Y)*a31(Y))
    def C23(Y): return  -(a11(Y)*a32(Y) - a12(Y)*a31(Y))
    def C31(Y): return  +(a12(Y)*a23(Y) - a13(Y)*a22(Y))
    def C32(Y): return  -(a11(Y)*a23(Y) - a13(Y)*a21(Y))
    def C33(Y): return  +(a11(Y)*a22(Y) - a12(Y)*a21(Y))

    cofA = [[C11, C12, C13],
            [C21, C22, C23],
            [C31, C32, C33]]

    # J = A^{-1} = (cof(A))^T / det(A) ⇒ J[k][i] = cofA[i][k] / detA
    J = [[None]*3 for _ in range(3)]
    for k in range(3):
        for i in range(3):
            # bind i,k now to avoid closure bugs
            def make_J(i=i, k=k):
                def Jki(Y):
                    d = detA(Y)
                    return cofA[i][k](Y) / d
                return Jki
            J[k][i] = make_J(i, k)

    return detA, cofA, J



#####################################################################################

def pdo_param_2d(kh, bfield, z1, z2, y1, y2, y1_d1=None, y1_d2=None, y2_d1=None, y2_d2=None,\
       y1_d1d1=None, y1_d2d2=None, y2_d1d1=None, y2_d2d2=None):

    """
    Configures a PDO for variable-coefficient PDEs on custom domains by specifying parameter maps
    and their derivatives. This function sets up the PDO based on geometric transformations
    from a reference domain to the target curved domain.

    We solve variable-coefficient PDEs on curved domains Psi by solving
    on a reference rectangle Omega.
    parameter_map (z_func) : given point on Omega, maps to point on Psi
    y_func is maps points on Psi to points on Omega
    partial derivatives are taken with respect to y = (y_1,y_2)
    
    Parameters:
    kh: Wave number or parameter related to the equation's physical properties.
    bfield: Function defining the magnetic field or other spatially varying properties within the domain.
    z1, z2: Functions defining the forward parameter map from the reference domain to the curved domain.
    y1, y2: Functions defining the inverse parameter map from the curved domain to the reference domain.
    y1_d1, y1_d2, y2_d1, y2_d2: First derivatives of the y mapping functions.
    y1_d1d1, y1_d2d2, y2_d1d1, y2_d2d2: Second derivatives of the y mapping functions.
    
    Returns:
    A configured PDO object along with the parameter map and inverse parameter map functions.
    """
    
    
    def parameter_map(xx):
        ZZ = xx.clone()
        ZZ[:,0] = z1(xx)
        ZZ[:,1] = z2(xx)
        return ZZ
    
    def inv_parameter_map(xx):
        YY = xx.clone()
        YY[:,0] = y1(xx)
        YY[:,1] = y2(xx)
        return YY

    
    c11 = cii_func(parameter_map,y1_d1,y1_d2)
    c22 = cii_func(parameter_map,y2_d1,y2_d2)
    c1  = ci_func (parameter_map,y1_d1d1,y1_d2d2)
    c2  = ci_func (parameter_map,y2_d1d1,y2_d2d2)
    c12 = cij_func(parameter_map,y1_d1,y2_d1,y1_d2,y2_d2)
    
    def c(xx):
        return bfield(parameter_map(xx),kh)
    
    pdo = PDO_2d(c11=c11,c22=c22,c1=c1,c2=c2,c12=c12,c=c)
    return pdo, parameter_map, inv_parameter_map

def pdo_param_3d(kh, bfield, z1, z2, z3, y1, y2, y3,
                 y1_d1=lambda xx: 0.*xx[:,0], y1_d2=lambda xx: 0.*xx[:,0], y1_d3=lambda xx: 0.*xx[:,0],
                 y2_d1=lambda xx: 0.*xx[:,0], y2_d2=lambda xx: 0.*xx[:,0], y2_d3=lambda xx: 0.*xx[:,0],
                 y3_d1=lambda xx: 0.*xx[:,0], y3_d2=lambda xx: 0.*xx[:,0], y3_d3=lambda xx: 0.*xx[:,0],
                 y1_d1d1=lambda xx: 0.*xx[:,0], y1_d1d2=lambda xx: 0.*xx[:,0], y1_d1d3=lambda xx: 0.*xx[:,0],
                 y1_d2d1=lambda xx: 0.*xx[:,0], y1_d2d2=lambda xx: 0.*xx[:,0], y1_d2d3=lambda xx: 0.*xx[:,0],
                 y1_d3d1=lambda xx: 0.*xx[:,0], y1_d3d2=lambda xx: 0.*xx[:,0], y1_d3d3=lambda xx: 0.*xx[:,0],
                 y2_d1d1=lambda xx: 0.*xx[:,0], y2_d1d2=lambda xx: 0.*xx[:,0], y2_d1d3=lambda xx: 0.*xx[:,0],
                 y2_d2d1=lambda xx: 0.*xx[:,0], y2_d2d2=lambda xx: 0.*xx[:,0], y2_d2d3=lambda xx: 0.*xx[:,0],
                 y2_d3d1=lambda xx: 0.*xx[:,0], y2_d3d2=lambda xx: 0.*xx[:,0], y2_d3d3=lambda xx: 0.*xx[:,0],
                 y3_d1d1=lambda xx: 0.*xx[:,0], y3_d1d2=lambda xx: 0.*xx[:,0], y3_d1d3=lambda xx: 0.*xx[:,0],
                 y3_d2d1=lambda xx: 0.*xx[:,0], y3_d2d2=lambda xx: 0.*xx[:,0], y3_d2d3=lambda xx: 0.*xx[:,0],
                 y3_d3d1=lambda xx: 0.*xx[:,0], y3_d3d2=lambda xx: 0.*xx[:,0], y3_d3d3=lambda xx: 0.*xx[:,0],
                 z1_d1=lambda xx: 0.*xx[:,0], z1_d2=lambda xx: 0.*xx[:,0], z1_d3=lambda xx: 0.*xx[:,0],
                 z2_d1=lambda xx: 0.*xx[:,0], z2_d2=lambda xx: 0.*xx[:,0], z2_d3=lambda xx: 0.*xx[:,0],
                 z3_d1=lambda xx: 0.*xx[:,0], z3_d2=lambda xx: 0.*xx[:,0], z3_d3=lambda xx: 0.*xx[:,0]):

    """
    Configures a 3D PDO for variable-coefficient PDEs on custom domains by specifying parameter maps
    and their derivatives. This function sets up the PDO based on geometric transformations
    from a reference domain to the target curved domain.

    We solve variable-coefficient PDEs on curved domains Psi by solving
    on a reference rectangle Omega.
    parameter_map (z_func) : given point on Omega, maps to point on Psi
    y_func is maps points on Psi to points on Omega
    partial derivatives are taken with respect to y = (y_1,y_2)
    
    Parameters:
    kh: Wave number or parameter related to the equation's physical properties.
    bfield: Function defining the magnetic field or other spatially varying properties within the domain.
    z1, z2: Functions defining the forward parameter map from the reference domain to the curved domain.
    y1, y2: Functions defining the inverse parameter map from the curved domain to the reference domain.
    y1_d1, y1_d2, y2_d1, y2_d2: First derivatives of the y mapping functions.
    y1_d1d1, y1_d2d2, y2_d1d1, y2_d2d2: Second derivatives of the y mapping functions.
    
    Returns:
    A configured PDO object along with the parameter map and inverse parameter map functions.
    """
    
    
    def parameter_map(xx):
        ZZ = xx.clone()
        ZZ[:,0] = z1(xx)
        ZZ[:,1] = z2(xx)
        ZZ[:,2] = z3(xx)
        return ZZ
    
    def inv_parameter_map(xx):
        YY = xx.clone()
        YY[:,0] = y1(xx)
        YY[:,1] = y2(xx)
        YY[:,2] = y3(xx)
        return YY

    # What we need to change for into divergence-form
    c11 = cii_func(parameter_map,y1_d1,y1_d2,y1_d3)
    c22 = cii_func(parameter_map,y2_d1,y2_d2,y2_d3)
    c33 = cii_func(parameter_map,y3_d1,y3_d2,y3_d3)
    c12 = cij_func(parameter_map,y1_d1,y2_d1,y1_d2,y2_d2,y1_d3,y2_d3)
    c13 = cij_func(parameter_map,y1_d1,y3_d1,y1_d2,y3_d2,y1_d3,y3_d3)
    c23 = cij_func(parameter_map,y2_d1,y3_d1,y2_d2,y3_d2,y2_d3,y3_d3)

    if True: # Keep the old approach as a fallback
        c1  = ci_func (parameter_map,y1_d1d1,y1_d2d2,y1_d3d3)
        c2  = ci_func (parameter_map,y2_d1d1,y2_d2d2,y2_d3d3)
        c3  = ci_func (parameter_map,y3_d1d1,y3_d2d2,y3_d3d3)
    else:
        A = [[y1_d1, y1_d2, y1_d3], [y2_d1, y2_d2, y2_d3], [y3_d1, y3_d2, y3_d3]]

        _, _, J = build_J_from_A(A)

        B = [[[y1_d1d1, y1_d1d2, y1_d1d3], [y2_d1d1, y2_d1d2, y2_d1d3], [y3_d1d1, y3_d1d2, y3_d1d3]],
             [[y1_d2d1, y1_d2d2, y1_d2d3], [y2_d2d1, y2_d2d2, y2_d2d3], [y3_d2d1, y3_d2d2, y3_d2d3]],
             [[y1_d3d1, y1_d3d2, y1_d3d3], [y2_d3d1, y2_d3d2, y2_d3d3], [y3_d3d1, y3_d3d2, y3_d3d3]]]

        trace_JB = get_traceJB(J, B)

        c1 = cj_func(parameter_map, trace_JB, J, A, B, 0)
        c2 = cj_func(parameter_map, trace_JB, J, A, B, 1)
        c3 = cj_func(parameter_map, trace_JB, J, A, B, 2)
    
    def c(xx):
        return bfield(parameter_map(xx),kh)
    
    pdo = PDO_3d(c11=c11,c22=c22,c33=c33,c1=c1,c2=c2,c3=c3,c12=c12,c13=c13,c23=c23,c=c)
    return pdo, parameter_map, inv_parameter_map


def get_param_helper(geom,bfield,kh,d=2):
    """
    Helper function for configuring PDO and parameter mappings based on the specified geometry.
    Supports various predefined geometries like 'sinusoidal', 'annulus', and 'curvy_annulus'.
    
    Parameters:
    geom: String specifying the geometry type.
    bfield: Function defining the magnetic field or spatially varying properties within the domain.
    kh: Wave number or parameter related to the equation's physical properties.
    d: dimension, 2 or 3.
    
    Returns:
    Configured PDO object, parameter map, and inverse parameter map functions for the specified geometry.
    """
    
    if ((geom == 'sinusoidal') or (geom== 'curved')):
        
        mag = 0.25
        psi    = lambda z: 1 - mag * torch.sin(6*z)
        dpsi   = lambda z:   - mag*6 * torch.cos(6*z)
        ddpsi  = lambda z:     mag*36  * torch.sin(6*z)
        
        y1_d1  = lambda xx: torch.ones(xx[:,0].shape,device=xx.device)
        y2_d1  = lambda xx: torch.mul(xx[:,1], dpsi(xx[:,0]))
        y2_d2  = lambda xx: psi(xx[:,0])
        if d==3:
            y3_d3 = lambda xx: torch.ones(xx[:,2].shape,device=xx.device)

        y2_d1d1  = lambda xx: torch.mul(xx[:,1], ddpsi(xx[:,0]))
        
        z1   = lambda xx: xx[:,0]
        z2   = lambda xx: torch.div(xx[:,1],psi(xx[:,0]))
        if d==3:
            z3   = lambda xx: xx[:,2]
        
        y1   = lambda xx: xx[:,0]
        y2   = lambda xx: torch.mul(xx[:,1],psi(xx[:,0]))
        if d==3:
            y3   = lambda xx: xx[:,2]
        
        if d==2:
            return pdo_param_2d(kh, bfield, z1, z2, y1, y2,
                            y1_d1=y1_d1, y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1)
        else:
            return pdo_param_3d(kh, bfield, z1, z2, z3, y1, y2, y3,
                            y1_d1=y1_d1, y2_d1=y2_d1, y2_d2=y2_d2, y3_d3=y3_d3, y2_d1d1=y2_d1d1)
    
    
    elif (geom == 'annulus'):
        
        const_theta = 1 / (2*np.pi) # Whole annulus, use with periodic
        #const_theta = 1/(np.pi/3) # Segment of annulus, not periodic

        r           = lambda zz: (zz[:,0]**2 + zz[:,1]**2)**0.5

        z1 = lambda zz: torch.mul( 1 + 1 * zz[:,1], torch.cos(zz[:,0]/const_theta) )
        z2 = lambda zz: torch.mul( 1 + 1 * zz[:,1], torch.sin(zz[:,0]/const_theta) )
        if d==3:
            z3 = lambda zz: zz[:,2]
        
        y1 = lambda zz: const_theta* torch.atan2(zz[:,1],zz[:,0])
        y2 = lambda zz: r(zz) - 1
        if d==3:
            y3   = lambda zz: zz[:,2]
        
        y1_d1    = lambda zz: -const_theta     * torch.div(zz[:,1], r(zz)**2)
        y1_d2    = lambda zz: +const_theta     * torch.div(zz[:,0], r(zz)**2)
        y1_d1d1  = lambda zz: +2*const_theta   * torch.div(torch.mul(zz[:,0],zz[:,1]), r(zz)**4)
        y1_d2d2  = lambda zz: -2*const_theta   * torch.div(torch.mul(zz[:,0],zz[:,1]), r(zz)**4)
        y1_d1d1 = None; y1_d2d2 = None


        y2_d1    = lambda zz: torch.div(zz[:,0], r(zz))
        y2_d2    = lambda zz: torch.div(zz[:,1], r(zz))
        y2_d1d1  = lambda zz: torch.div(zz[:,1]**2, r(zz)**3)
        y2_d2d2  = lambda zz: torch.div(zz[:,0]**2, r(zz)**3)

        if d==3:
            y3_d3 = lambda zz: torch.ones(zz[:,2].shape,device=zz.device)
        
        if d==2:
            return pdo_param_2d(kh, bfield, z1,z2,y1,y2,
                                y1_d1=y1_d1, y1_d2=y1_d2,
                                y1_d1d1=y1_d1d1, y1_d2d2=y1_d2d2,
                                y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1, y2_d2d2=y2_d2d2)
        else: # d==3
            return pdo_param_3d(kh, bfield, z1,z2,z3,y1,y2,y3,
                                y1_d1=y1_d1, y1_d2=y1_d2,
                                y1_d1d1=y1_d1d1, y1_d2d2=y1_d2d2,
                                y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1, y2_d2d2=y2_d2d2,
                                y3_d3=y3_d3)
    
    elif (geom == 'curvy_annulus'):
        
        const_theta = 1/(np.pi/3)
        const_phase = 5
        const_amp   = 0.2
        
        r           = lambda zz: (zz[:,0]**2 + zz[:,1]**2)**0.5
        
        radius_zmap = lambda zz: 1 + const_amp*torch.cos(const_phase*zz[:,0]/const_theta) + zz[:,1]

        z1 = lambda zz: torch.mul( radius_zmap(zz), torch.cos(zz[:,0] / const_theta) )
        z2 = lambda zz: torch.mul( radius_zmap(zz), torch.sin(zz[:,0] / const_theta) )
        
        theta_ymap = lambda zz: torch.arctan2(zz[:,1], zz[:,0]) 
        
        y1 = lambda zz: const_theta * theta_ymap(zz)
        y2 = lambda zz: r(zz) - const_amp * torch.cos(const_phase*theta_ymap(zz)) - 1
        
        y1_d1    = lambda zz: -const_theta     * torch.div(zz[:,1], r(zz)**2)
        y1_d2    = lambda zz: +const_theta     * torch.div(zz[:,0], r(zz)**2)
        y1_d1d1  = lambda zz: +2*const_theta   * torch.div(torch.mul(zz[:,0],zz[:,1]), r(zz)**4)
        y1_d2d2  = lambda zz: -2*const_theta   * torch.div(torch.mul(zz[:,0],zz[:,1]), r(zz)**4)
        
        def y2_d1(zz):
            result = torch.div(zz[:,0], r(zz))
            
            tmp = torch.div( torch.sin(const_phase * theta_ymap(zz)), r(zz)**2)
            
            result -= const_amp*const_phase * torch.mul(zz[:,1], tmp)
            return result
        
        def y2_d2(zz):
            result = torch.div(zz[:,1], r(zz))
            tmp = torch.div( torch.sin(const_phase * theta_ymap(zz)), r(zz)**2)
            
            result += torch.mul(const_amp*const_phase* zz[:,0], tmp)
            return result
        
        def y2_d1d1(zz):
            result = torch.div(zz[:,1]**2, r(zz)**3)
            
            tmp1 = torch.mul(zz[:,0], torch.sin( const_phase * theta_ymap(zz)))
            tmp2 = torch.mul(zz[:,1], torch.cos( const_phase * theta_ymap(zz)))
                                           
            result += const_amp*const_phase * torch.div( torch.mul(zz[:,1],\
                                                                   2*tmp1+const_phase*tmp2), r(zz)**4) 
            return result
        
        def y2_d2d2(zz):
            result = torch.div(zz[:,0]**2, r(zz)**3)
            
            tmp1 = torch.mul(zz[:,0], torch.cos( const_phase * theta_ymap(zz)))
            tmp2 = torch.mul(zz[:,1], torch.sin( const_phase * theta_ymap(zz)))
                                           
            result += const_amp*const_phase * torch.div( torch.mul(zz[:,0],\
                                                                   const_phase*tmp1-2*tmp2), r(zz)**4)                           
                                           
            return result
        
        return pdo_param_2d(kh, bfield, z1,z2,y1,y2,\
                         y1_d1=y1_d1, y1_d2=y1_d2,\
                         y1_d1d1=y1_d1d1, y1_d2d2=y1_d2d2,\
                         y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1, y2_d2d2=y2_d2d2)

    elif (geom == "twisted_torus"):
        if d==2:
            ValueError("twsited torus is 3D only")

        tau = 1.0
        R = 1.5
        bnds = [[-1.,-1.,-1.],[1.,1.,1.]]
        def z1(zz):
            ZZ = 2*zz - 1
            c=torch.cos(tau*torch.pi*ZZ[:,0])
            s=torch.sin(tau*torch.pi*ZZ[:,0])
            c2 = torch.multiply(c,c)
            cs = torch.multiply(c,s)
            q = torch.multiply(c2,ZZ[:,1])-torch.multiply(cs,ZZ[:,2])+c*(R+1)
            return q

        def z1_d1(zz):
            c = torch.cos(tau*torch.pi*zz[:,0])
            s = torch.sin(tau*torch.pi*zz[:,0])

            return -2*tau*torch.pi*c*s * zz[:,1] - tau*torch.pi*(c*c - s*s) * zz[:,2] - tau*torch.pi*s*(R+1)

        def z1_d2(zz):
            c = torch.cos(tau*torch.pi*zz[:,0])
            return c*c

        def z1_d3(zz):
            c = torch.cos(tau*torch.pi*zz[:,0])
            s = torch.sin(tau*torch.pi*zz[:,0])
            return -c*s

        def z2(zz):
            ZZ = 2*zz - 1
            c=torch.cos(tau*torch.pi*ZZ[:,0])
            s=torch.sin(tau*torch.pi*ZZ[:,0])
            s2 = torch.multiply(s,s)
            cs = torch.multiply(c,s)
            q = torch.multiply(cs,ZZ[:,1])-torch.multiply(s2,ZZ[:,2])+s*(R+1)
            return q

        def z2_d1(zz):
            c = torch.cos(tau*torch.pi*zz[:,0])
            s = torch.sin(tau*torch.pi*zz[:,0])
            return tau*torch.pi*(c*c - s*s) * zz[:,1] - 2*tau*torch.pi*s*c * zz[:,2] + tau*torch.pi*c*(R+1)

        def z2_d2(zz):
            c = torch.cos(tau*torch.pi*zz[:,0])
            s = torch.sin(tau*torch.pi*zz[:,0])
            return c*s

        def z2_d3(zz):
            s = torch.sin(tau*torch.pi*zz[:,0])
            return -(s*s)

        def z3(zz):
            ZZ = 2*zz - 1
            c=torch.cos(tau*torch.pi*ZZ[:,0])
            s=torch.sin(tau*torch.pi*ZZ[:,0])
            q = torch.multiply(s,ZZ[:,1])+torch.multiply(c,ZZ[:,2])
            return q

        def z3_d1(zz):
            c = torch.cos(tau*torch.pi*zz[:,0])
            s = torch.sin(tau*torch.pi*zz[:,0])
            return tau*torch.pi * (c*zz[:,1] - s*zz[:,2])

        def z3_d2(zz):
            s = torch.sin(tau*torch.pi*zz[:,0])
            return s

        def z3_d3(zz):
            c = torch.cos(tau*torch.pi*zz[:,0])
            return c

        def y1(zz):
            th = tau*torch.arctan2(zz[:,1],zz[:,0])
            return th/torch.pi

        def y2(zz):
            # p is a vector of points, Nx3
            th = tau*torch.arctan2(zz[:,1],zz[:,0])
            c=torch.cos(th)
            s=torch.sin(th)
            c2 = torch.multiply(c,c)
            cs = torch.multiply(c,s)
            q = torch.multiply(zz[:,0],c2)+torch.multiply(zz[:,1],cs)-(R+1)*c + torch.multiply(s,zz[:,2])
            return q


        def y3(zz):
            th = tau*torch.arctan2(zz[:,1],zz[:,0])
            c=torch.cos(th)
            s=torch.sin(th)
            s2 = torch.multiply(s,s)
            cs = torch.multiply(c,s)
            q = -torch.multiply(zz[:,0],cs)-torch.multiply(zz[:,1],s2)+(R+1)*s+torch.multiply(c,zz[:,2])
            return q

        #verified
        def y1_d1(zz):
            r2 = zz[:,0] * zz[:,0] + zz[:,1] * zz[:,1]
            return -(zz[:,1] / r2) / torch.pi
        #verified
        def y1_d2(zz):
            r2 = zz[:,0] * zz[:,0] + zz[:,1] * zz[:,1]
            return (zz[:,0] / r2) / torch.pi

        #verified
        def y2_d1(zz):
            th  = tau * torch.arctan2(zz[:,1],zz[:,0])
            c2t = torch.cos(2*th)
            s2t = torch.sin(2*th)
            s   = torch.sin(th)
            c   = torch.cos(th)
            r2  = zz[:,0] * zz[:,0] + zz[:,1] * zz[:,1]
            A   = (s2t*zz[:,0]*zz[:,1] - c2t*zz[:,1]**2 - (R+1)*s*zz[:,1] - zz[:,2]*zz[:,1]*c)
            return (c2t+1) / 2. + A/r2
        #verified
        def y2_d2(zz):
            th = tau*torch.arctan2(zz[:,1],zz[:,0])
            c2t = torch.cos(2*th)
            s2t = torch.sin(2*th)
            s   = torch.sin(th)
            c   = torch.cos(th)
            r2  = zz[:,0]*zz[:,0]+zz[:,1]*zz[:,1]
            A   = ( c2t*zz[:,0]*zz[:,1] - s2t*zz[:,0]*zz[:,0] + (R+1)*s*zz[:,0] + zz[:,2]*zz[:,0]*c)
            return s2t/2. + A/r2

        def y2_d3(zz):
            th = tau*torch.arctan2(zz[:,1],zz[:,0])
            return torch.sin(th)

        def y3_d1(zz):
            th = tau*torch.arctan2(zz[:,1],zz[:,0])
            c2t = torch.cos(2*th)
            s2t = torch.sin(2*th)
            s   = torch.sin(th)
            c   = torch.cos(th)
            r2  = zz[:,0]*zz[:,0]+zz[:,1]*zz[:,1]
            A   = (c2t*zz[:,0]*zz[:,1] + s2t*zz[:,1]**2 - (R+1)*c*zz[:,1] + zz[:,2]*zz[:,1]*s)
            return -s2t/2. + A/r2

        def y3_d2(zz):
            th = tau*torch.arctan2(zz[:,1],zz[:,0])
            c2t = torch.cos(2*th)
            s2t = torch.sin(2*th)
            s   = torch.sin(th)
            c   = torch.cos(th)
            r2  = zz[:,0]*zz[:,0]+zz[:,1]*zz[:,1]
            A   = (s2t*zz[:,0]*zz[:,1] + c2t*zz[:,0]*zz[:,0] - (R+1)*c*zz[:,0] + zz[:,2]*zz[:,0]*s)
            return (c2t-1)/2. - A/r2
        #verified
        def y3_d3(zz):
            th = tau*torch.arctan2(zz[:,1],zz[:,0])
            return torch.cos(th)


        #verified
        def y1_d1d1(zz):
            r2 = zz[:,0]*zz[:,0]+zz[:,1]*zz[:,1]
            return (2*zz[:,1]*zz[:,0]/r2)/(r2*torch.pi)
        def y1_d1d2(zz):
            Y1 = zz[:,0]
            Y2 = zz[:,1]
            r2 = Y1*Y1 + Y2*Y2
            return -(Y1*Y1 - Y2*Y2) / (torch.pi * r2*r2)
        #verified
        def y1_d2d1(zz):
            return y1_d1d2(zz)
        def y1_d2d2(zz):
            r2 = zz[:,0]*zz[:,0]+zz[:,1]*zz[:,1]
            return -2*(zz[:,0]*zz[:,1]/r2)/(r2*torch.pi)

        #verified
        def y2_d1d1(zz):
            th = tau*torch.arctan2(zz[:,1],zz[:,0])
            c2t = torch.cos(2*th)
            s2t = torch.sin(2*th)
            s   = torch.sin(th)
            c   = torch.cos(th)
            r2  = zz[:,0]*zz[:,0]+zz[:,1]*zz[:,1]
            A   = (s2t*zz[:,0]*zz[:,1] - c2t*zz[:,1]*zz[:,1] - (R+1)*s*zz[:,1] - zz[:,2]*zz[:,1]*c)
            dA = zz[:,1]*s2t-(2*c2t*zz[:,0]*(zz[:,1]**2)+2*s2t*zz[:,1]**3-(R+1)*c*zz[:,1]**2 + zz[:,2]*s*(zz[:,1]**2) )/r2
            return (zz[:,1]*s2t + dA - 2*A*zz[:,0]/r2 )/r2

        def y2_d1d2(zz):
            Y1 = zz[:,0]
            Y2 = zz[:,1]
            Y3 = zz[:,2]

            r2  = Y1*Y1 + Y2*Y2
            th  = tau * torch.atan2(Y2, Y1)
            s2t = torch.sin(2*th)
            c2t = torch.cos(2*th)
            s   = torch.sin(th)
            c   = torch.cos(th)

            # dθ/dY2
            t = tau * Y1 / r2

            # A as in y2_d1
            A = s2t*Y1*Y2 - c2t*Y2*Y2 - (R+1)*s*Y2 - Y3*Y2*c

            # partial
            dA_dY2 = 2*c2t*t*Y1*Y2 + s2t*Y1 + 2*s2t*t*Y2*Y2 - 2*c2t*Y2 - (R+1)*(c*t*Y2 + s) - Y3*c + Y3*Y2*s*t

            # partial [(c2t+1)/2] = - s2t * t
            return - s2t*t + (dA_dY2 * r2 - A * (2*Y2)) / (r2*r2)
        
        def y2_d1d3(zz):
            Y1 = zz[:,0]
            Y2 = zz[:,1]

            r2  = Y1*Y1 + Y2*Y2
            th  = tau * torch.atan2(Y2, Y1)
            c   = torch.cos(th)

            # Only the -Y3*Y2*c / r2 piece depends on Y3
            return -(Y2 * c) / r2

        #verified
        def y2_d2d1(zz):
            return y2_d1d2(zz)

        def y2_d2d2(zz):
            th = tau*torch.arctan2(zz[:,1],zz[:,0])
            c2t = torch.cos(2*th)
            s2t = torch.sin(2*th)
            s   = torch.sin(th)
            c   = torch.cos(th)
            r2  = zz[:,0]*zz[:,0]+zz[:,1]*zz[:,1]
            A   = ( c2t*zz[:,0]*zz[:,1] - s2t*zz[:,0]*zz[:,0] + (R+1)*s*zz[:,0] + zz[:,2]*zz[:,0]*c)
            dA  = zz[:,0]*c2t-(2*s2t*zz[:,1]*zz[:,0]**2+2*c2t*(zz[:,0]**3)-(R+1)*c*(zz[:,0]**2)+zz[:,2]*s*(zz[:,0]**2))/r2
            return (c2t*zz[:,0] + dA - 2*A*zz[:,1]/r2 )/r2
        
        def y2_d2d3(zz):
            Y1 = zz[:,0]
            Y2 = zz[:,1]

            r2 = Y1*Y1 + Y2*Y2
            th = tau * torch.atan2(Y2, Y1)

            return torch.cos(th) * tau * (Y1 / r2)

        def y2_d3d1(zz):
            return y2_d1d3(zz)

        def y2_d3d2(zz):
            return y2_d2d3(zz)


        #verified
        def y3_d1d1(zz):
            th = tau*torch.arctan2(zz[:,1],zz[:,0])
            c2t = torch.cos(2*th)
            s2t = torch.sin(2*th)
            s   = torch.sin(th)
            c   = torch.cos(th)
            r2  = zz[:,0]*zz[:,0]+zz[:,1]*zz[:,1]
            A   = (c2t*zz[:,0]*zz[:,1] + s2t*zz[:,1]**2 - (R+1)*c*zz[:,1] + zz[:,2]*zz[:,1]*s)
            dA  = zz[:,1]*c2t+(2*zz[:,0]*(zz[:,1]**2)*s2t-2*(zz[:,1]**3)*c2t-(R+1)*(zz[:,1]**2)*s-zz[:,2]*(zz[:,1]**2)*c)/r2
            return (c2t*zz[:,1] + dA - 2*zz[:,0]*A/r2)/r2

        def y3_d1d2(zz):
            Y1 = zz[:,0]
            Y2 = zz[:,1]
            Y3 = zz[:,2]

            r2  = Y1*Y1 + Y2*Y2
            th  = tau * torch.atan2(Y2, Y1)
            s2t = torch.sin(2*th)
            c2t = torch.cos(2*th)
            s   = torch.sin(th)
            c   = torch.cos(th)

            # dθ/dY2
            t = tau * Y1 / r2

            # A as in y3_d1
            A = c2t*Y1*Y2 + s2t*Y2*Y2 - (R+1)*c*Y2 + Y3*Y2*s

            # dA/dY2
            dA_dY2 = (-2*s2t*t)*Y1*Y2 + c2t*Y1 + (2*c2t*t)*Y2*Y2 + 2*s2t*Y2 + (R+1)*s*t*Y2 - (R+1)*c + Y3*s + Y3*Y2*c*t

            return -c2t*t + (dA_dY2 * r2 - A * (2*Y2)) / (r2 * r2)

        def y3_d1d3(zz):
            Y1 = zz[:,0]
            Y2 = zz[:,1]
            Y3 = zz[:,2]

            r2  = Y1*Y1 + Y2*Y2
            th  = tau * torch.atan2(Y2, Y1)
            s   = torch.sin(th)

            # Only the (Y3*Y2*s)/r2 part of y3_d1 depends on Y3
            return (Y2 * s) / r2

        def y3_d2d1(zz):
            return y3_d1d2(zz)

        #verified
        def y3_d2d2(zz):
            th = tau*torch.arctan2(zz[:,1],zz[:,0])
            c2t = torch.cos(2*th)
            s2t = torch.sin(2*th)
            s   = torch.sin(th)
            c   = torch.cos(th)
            r2  = zz[:,0]*zz[:,0]+zz[:,1]*zz[:,1]
            A   = (s2t*zz[:,0]*zz[:,1] + c2t*zz[:,0]*zz[:,0] - (R+1)*c*zz[:,0] + zz[:,2]*zz[:,0]*s)
            dA  = zz[:,0]*s2t+(2*(zz[:,0]**2)*zz[:,1]*c2t-2*(zz[:,0]**3)*s2t + (R+1)*(zz[:,0]**2)*s + zz[:,2]*(zz[:,0]**2)*c)/r2
            return -(s2t*zz[:,0] + dA - 2*zz[:,1]*A/r2)/r2

        def y3_d2d3(zz):
            Y1 = zz[:,0]
            Y2 = zz[:,1]

            r2 = Y1*Y1 + Y2*Y2
            th = tau * torch.atan2(Y2, Y1)

            return -torch.sin(th) * tau * (Y1 / r2)

        def y3_d3d1(zz):
            return y3_d1d3(zz)

        def y3_d3d2(zz):
            return y3_d2d3(zz)

        return pdo_param_3d(kh, bfield, z1,z2,z3,y1,y2,y3,
                                y1_d1=y1_d1, y1_d2=y1_d2, y2_d1=y2_d1, y2_d2=y2_d2, y3_d3=y3_d3,
                                y1_d1d1=y1_d1d1, y1_d2d2=y1_d2d2,
                                y2_d1d1=y2_d1d1, y2_d2d2=y2_d2d2,
                                y2_d3=y2_d3, y3_d1=y3_d1, y3_d2=y3_d2, y3_d1d1=y3_d1d1, y3_d2d2=y3_d2d2,
                                y1_d1d2=y1_d1d2, y1_d2d1=y1_d2d1,
                                y2_d1d2=y2_d1d2, y2_d1d3=y2_d1d3, y2_d2d1=y2_d2d1, y2_d2d3=y2_d2d3, y2_d3d1=y2_d3d1, y2_d3d2=y2_d3d2,
                                y3_d1d2=y3_d1d2, y3_d1d3=y3_d1d3, y3_d2d1=y3_d2d1, y3_d2d3=y3_d2d3, y3_d3d1=y3_d3d1, y3_d3d2=y3_d3d2) # The new ones
    
    elif (geom == 'corner'):
        
        warnings.warn("Corner geometry does not work")
        
        z1 = lambda zz: zz[:,0]
        def z2(zz):
            bool_map = zz[:,0] >= 0.5
            zz[bool_map,1] *= 0.5
            return zz[:,1]
        
        y1 = lambda zz: zz[:,0]
        
        def y2(zz):
            bool_map = zz[:,0] >= 0.5
            zz[bool_map,1] *= 2
            return zz[:,1]
            
        y1_d1 = lambda zz: torch.ones(zz.shape[0],device=zz.device)
        
        def y2_d2(zz):
            bool_map = zz[:,0] >= 0.5
            result = torch.ones(zz.shape[0], device=zz.device)
            result[bool_map] *= 2
            return result
        
        return pdo_param_2d(kh, bfield,z1,z2,y1,y2,\
                         y1_d1=y1_d1, y2_d2=y2_d2)
        
    else:
        raise ValueError("geom %s not available"%(geom))


#####################################################################################

def get_param_map_and_pdo(geom, bfield, kh, d=2):
    """
    Main function to obtain the PDO and parameter mappings for a given geometry and magnetic field configuration.
    Directly supports simple geometries like 'square' and uses `get_param_helper` for more complex geometries.
    
    Parameters:
    geom: String specifying the geometry type.
    bfield: Function defining the magnetic field or spatially varying properties within the domain.
    kh: Wave number or parameter related to the equation's physical properties.
    d: dimension, 2 or 3.
    
    Returns:
    Configured PDO object along with the parameter map and inverse parameter map functions for the specified geometry.
    """
    
    if (geom == 'square'):
        # identity maps for parameter maps 
        param_map     = lambda xx: xx.clone()
        inv_param_map = lambda xx: xx.clone()
        
        def c_func(xx):
            return bfield(xx,kh)
        op = PDO_2d(const(1),const(1), c=c_func)
        if d==3:
            op = PDO_3d(const(1),const(1),const(1), c=c_func)
        
        return op,param_map,inv_param_map
    else:
        return get_param_helper(geom,bfield,kh,d=d)