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
                result += torch.mul(yi_d1(yy),yj_d1(yy))
            if (bool_expr2):
                result += torch.mul(yi_d2(yy),yj_d2(yy))
            if (bool_expr3):
                result += torch.mul(yi_d3(yy),yj_d3(yy))
            return result
    return cij

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

def pdo_param_3d(kh, bfield, z1, z2, z3, y1, y2, y3, y1_d1=None, y1_d2=None, y1_d3=None, y2_d1=None, y2_d2=None,
       y2_d3=None, y3_d1=None, y3_d2=None, y3_d3=None, y1_d1d1=None, y1_d2d2=None, y1_d3d3=None, y2_d1d1=None,
       y2_d2d2=None, y2_d3d3=None, y3_d1d1=None, y3_d2d2=None, y3_d3d3=None,):

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

    
    c11 = cii_func(parameter_map,y1_d1,y1_d2,y1_d3)
    c22 = cii_func(parameter_map,y2_d1,y2_d2,y2_d3)
    c33 = cii_func(parameter_map,y3_d1,y3_d2,y3_d3)
    c1  = ci_func (parameter_map,y1_d1d1,y1_d2d2,y1_d3d3)
    c2  = ci_func (parameter_map,y2_d1d1,y2_d2d2,y2_d3d3)
    c3  = ci_func (parameter_map,y3_d1d1,y3_d2d2,y3_d3d3)
    c12 = cij_func(parameter_map,y1_d1,y2_d1,y1_d2,y2_d2,y1_d3,y2_d3)
    c13 = cij_func(parameter_map,y1_d1,y3_d1,y1_d2,y3_d2,y1_d3,y3_d3)
    c23 = cij_func(parameter_map,y2_d1,y3_d1,y2_d2,y3_d2,y2_d3,y3_d3)
    
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
        
        mag = 0.025
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
            return pdo_param_2d(kh, bfield, z1, z2, y1, y2,\
                            y1_d1=y1_d1, y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1)
        else:
            return pdo_param_3d(kh, bfield, z1, z2, z3, y1, y2, y3,\
                            y1_d1=y1_d1, y2_d1=y2_d1, y2_d2=y2_d2, y3_d3=y3_d3, y2_d1d1=y2_d1d1)
    
    
    elif (geom == 'annulus'):
        
        const_theta = 1/(np.pi/3)
        r           = lambda zz: (zz[:,0]**2 + zz[:,1]**2)**0.5

        z1 = lambda zz: torch.mul( 1 + 1 * zz[:,1], torch.cos(zz[:,0]/const_theta) )
        z2 = lambda zz: torch.mul( 1 + 1 * zz[:,1], torch.sin(zz[:,0]/const_theta) )
        
        y1 = lambda zz: const_theta* torch.atan2(zz[:,1],zz[:,0])
        y2 = lambda zz: r(zz) - 1
        
        y1_d1    = lambda zz: -const_theta     * torch.div(zz[:,1], r(zz)**2)
        y1_d2    = lambda zz: +const_theta     * torch.div(zz[:,0], r(zz)**2)
        y1_d1d1  = lambda zz: +2*const_theta   * torch.div(torch.mul(zz[:,0],zz[:,1]), r(zz)**4)
        y1_d2d2  = lambda zz: -2*const_theta   * torch.div(torch.mul(zz[:,0],zz[:,1]), r(zz)**4)
        y1_d1d1 = None; y1_d2d2 = None


        y2_d1    = lambda zz: torch.div(zz[:,0], r(zz))
        y2_d2    = lambda zz: torch.div(zz[:,1], r(zz))
        y2_d1d1  = lambda zz: torch.div(zz[:,1]**2, r(zz)**3)
        y2_d2d2  = lambda zz: torch.div(zz[:,0]**2, r(zz)**3)
        
        return pdo_param_2d(kh, bfield, z1,z2,y1,y2,\
                         y1_d1=y1_d1, y1_d2=y1_d2,\
                         y1_d1d1=y1_d1d1, y1_d2d2=y1_d2d2,\
                         y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1, y2_d2d2=y2_d2d2)
    
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