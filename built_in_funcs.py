# Import necessary libraries
import torch  # For tensor computations
import numpy as np  # For numerical operations
torch.set_default_dtype(torch.double)  # Set default tensor data type to double precision for higher numerical accuracy
import pdo  # Import a custom library for dealing with Partial Differential Operators

from scipy.special import hankel1  # Import the Hankel function of the first kind for wave-related computations

# FUNCTIONS FOR DIRICHLET DATA AND BODY LOAD

def parameter_map(xx, psi):
    """
    Applies a parameter mapping to the y-coordinates of a set of points based on a given function `psi`.
    This can be used to transform the computational domain in simulations.
    
    Parameters:
    - xx: A tensor of shape (N, 2) representing N points in a 2D space.
    - psi: A function that takes x-coordinates and returns a scaling factor for the y-coordinates.
    
    Returns:
    - A tensor of shape (N, 2) with transformed y-coordinates.
    """
    xx_tmp = xx.clone()  # Clone the input tensor to avoid modifying it directly
    xx_tmp[:, 1] = torch.div(xx_tmp[:, 1], psi(xx_tmp[:, 0]))  # Apply the transformation to y-coordinates
    return xx_tmp

def inv_parameter_map(xx, psi):
    """
    Applies an inverse parameter mapping to the y-coordinates of a set of points based on a given function `psi`.
    This reverses the effect of `parameter_map`.
    
    Parameters:
    - xx: A tensor of shape (N, 2) representing N points in a 2D space.
    - psi: A function that takes x-coordinates and returns a scaling factor for the y-coordinates.
    
    Returns:
    - A tensor of shape (N, 2) with transformed y-coordinates.
    """
    xx_tmp = xx.clone()  # Clone the input tensor to avoid modifying it directly
    xx_tmp[:, 1] = torch.mul(xx_tmp[:, 1], psi(xx_tmp[:, 0]))  # Reverse the transformation applied to y-coordinates
    return xx_tmp

def parameter_map_pdo(psi, dpsi, ddpsi, bfield, kh):
    """
    Constructs a Partial Differential Operator (PDO) with coefficients that are adjusted based on a parameter mapping.
    This is used to define the PDE in a transformed domain.
    
    Parameters:
    - psi: The mapping function for y-coordinates.
    - dpsi: The first derivative of `psi`.
    - ddpsi: The second derivative of `psi`.
    - bfield: A function defining the magnetic field within the domain.
    - kh: A parameter related to the wave number or magnetic field strength.
    
    Returns:
    - An instance of `PDO_2d` representing the PDE operator with mapped parameters.
    """
    def c22(xx):
        # Coefficient function for the second derivative term with respect to y after transformation
        yy = parameter_map(xx, psi)
        return +(torch.mul(yy[:, 1]**2, dpsi(yy[:, 0])**2) + psi(yy[:, 0])**2)

    def c12(xx):
        # Coefficient function for the mixed derivative term after transformation
        yy = parameter_map(xx, psi)
        return +torch.mul(dpsi(yy[:, 0]), yy[:, 1])

    def c2(xx):
        # Coefficient function for the first derivative term with respect to y after transformation
        yy = parameter_map(xx, psi)
        return -torch.mul(yy[:, 1], ddpsi(yy[:, 0]))

    def c(xx):
        # Coefficient function for the zeroth-order term, representing the magnetic field
        return bfield(xx, kh)

    op = pdo.PDO_2d(c11=pdo.const(+1), c22=c22, c12=c12, c2=c2, c=c)  # Construct the PDO with specified coefficients
    return op

def uu_dir_func_mms(xx, Lx, Ly):
    """
    Defines a Dirichlet boundary condition using the Method of Manufactured Solutions (MMS).
    
    Parameters:
    - xx: A tensor of shape (N, 2) representing N points in a 2D space.
    - Lx, Ly: Parameters defining the wave characteristics in the x and y directions, respectively.
    
    Returns:
    - A tensor representing the boundary condition at each input point.
    """
    uu_exact = torch.sin(Lx * xx[:, 0]) * torch.exp(Ly * xx[:, 1])
    return uu_exact.unsqueeze(-1)  # Add a singleton dimension to match expected shape

def ff_body_func_mms(xx, Lx, Ly):
    """
    Defines a body load (source term) for the PDE using the Method of Manufactured Solutions (MMS).
    
    Parameters:
    - xx: A tensor of shape (N, 2) representing N points in a 2D space.
    - Lx, Ly: Parameters defining the wave characteristics in the x and y directions, respectively.
    
    Returns:
    - A tensor representing the source term at each input point.
    """
    uu_exact = torch.sin(Lx * xx[:, 0]) * torch.exp(Ly * xx[:, 1])
    ff_body = (Lx**2 - Ly**2) * uu_exact
    return ff_body.unsqueeze(-1)  # Add a singleton dimension to match expected shape


def uu_dir_pulse(xx,kh):
    
    uu_dir = torch.zeros(xx.shape[0],1)
    bnd_left   = (xx[:,0] == 0)
    inds_left  = torch.where(bnd_left)[0]
   
    c = 1; width=200
    
    uu_dir[inds_left] = c * torch.exp( - width * (xx[inds_left,1] - 0.5)**2).unsqueeze(-1)
    return uu_dir

def uu_dir_func_greens(xx,kh,center=torch.tensor([-0.1,+0.5])):
    
    dd0 = xx[:,0] - center[0];
    dd1 = xx[:,1] - center[1]; 
    ddsq = np.multiply(dd0,dd0) + np.multiply(dd1,dd1)
    if (kh == 0):
        uu_exact = (1/np.pi) * np.log(ddsq);
    else:
        dist_x = np.sqrt(ddsq)
        uu_exact = (1j/4) * hankel1(0,kh*dist_x);
        uu_exact = np.real(uu_exact)
    return uu_exact.unsqueeze(-1)

def ff_body_pulse(xx,kh):
    
    xx_sq = (xx[:,0] - 0.25)**2 + (xx[:,1] - 0.25)**2
    pulse = 1.5 * torch.exp(-100 * xx_sq)
    
    return pulse.unsqueeze(-1)

def bfield_constant(xx,kh):
    return -(kh**2) * torch.ones(xx.shape[0],device=xx.device)

def bfield_bumpy(xx,kh):
    
    return -(kh**2 * (1 - (torch.sin(4*np.pi*xx[:,0]) \
                           * torch.sin(4*np.pi*xx[:,1]))**2)).unsqueeze(-1)
    
def bfield_crystal(xx,kh,crystal_start=0.2,crystal_end=0.8,dist=0.05):
    
    mag   = 1.0;
    width = 1500;
    
    b = torch.zeros(xx.shape[0],device=xx.device)
    
    xstart=crystal_start; xend = crystal_end
    ystart=crystal_start; yend = crystal_end
    
    for x in np.arange(xstart,xend+dist,dist):
        for y in np.arange(ystart,yend,dist):
            xx_sq_c0 = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
            b += mag * torch.exp(-width * xx_sq_c0)
    max_val = torch.max(b)
    kh_fun = -kh**2 * (1 - b/max_val)
    return kh_fun.unsqueeze(-1)

def bfield_crystal_waveguide(xx,kh):
    
    mag   = 0.930655
    width = 2500; 
    
    b = torch.zeros(xx.shape[0],device=xx.device)
    
    dist = 0.04
    x0=0.1+0.5*dist; x1 = 0.50; x2 = x1+2.5*dist; x3= 0.9
    y0=0.1+0.5*dist; y1 = 0.50; y2 = y1+2.5*dist; y3= 0.9
    
    # box of points [x0,x1] x [y0,y1]
    for x in np.arange(x0,x1,dist):
        for y in np.arange(y0,y1,dist):
            xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
            b += mag * torch.exp(-width * xx_sq_c)

    # box of points [x0,x1] x [y0,y2]
    for x in np.arange(x2,x3,dist):
        for y in np.arange(y0,y2-0.5*dist,dist):
            xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
            b += mag * torch.exp(-width * xx_sq_c)
            
    # box of points [x0,x3] x [y2,y3]
    for x in np.arange(x0,x3,dist):
        for y in np.arange(y2,y3,dist):
            xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
            b += mag * torch.exp(-width * xx_sq_c)    
    
    kh_fun = -kh**2 * (1 - b)
    return kh_fun.unsqueeze(-1)

def bfield_star(xx,kh):
    
    mag   = 1.0;
    width = 2000;
    
    b = torch.zeros(xx.shape[0],device=xx.device)
    theta_step = 2*np.pi / 100
    
    for theta in np.arange(0,np.pi,theta_step):

        radius = 0.4*np.cos(5*theta)
        x = radius * np.cos(theta)+0.5;
        y = radius * np.sin(theta)+0.5;

        xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
        b += mag * torch.exp(-width * xx_sq_c)
        
    kh_fun = -kh**2 * (1 - b)
    return kh_fun.unsqueeze(-1)

def bfield_crystal_rhombus(xx,kh,crystal_start=0.25,crystal_end=0.75,dist=0.05):
    
    mag   = 1.0;
    width = 2000;
    
    b = torch.zeros(xx.shape[0],device=xx.device)
    
    xstart=crystal_start; xend = crystal_end
    ystart=crystal_start; yend = crystal_end
    
    theta_change = np.pi/16
    
    for x in np.arange(xstart,xend+dist,dist):
        for y in np.arange(ystart,yend,dist):
            
            r      = np.sqrt( (x-0.5)**2 + (y-0.5)**2 )
            theta  = np.arctan2(y-0.5,x-0.5)
            
            theta += theta_change 
            
            x_prime = r * np.sin(theta) + 0.5
            y_prime = r * np.cos(theta) + 0.5
            xx_sq_c0 = (xx[:,0] - x_prime)**2 + (xx[:,1] - y_prime)**2
            b += mag * torch.exp(-width * xx_sq_c0)
    
    kh_fun = -kh**2 * (1 - b)
    return kh_fun.unsqueeze(-1)

def bfield_crystal_circle(xx,kh):
    
    mag   = 1.0;
    width = 2000;
    
    b = torch.zeros(xx.shape[0],device=xx.device)
    
    radius = 0.3
    theta_step = 2*np.pi / 40
    
    # box of points [x0,x1] x [y0,y1]
    wedge = 2*np.pi / 10; radian_start = 1.5*np.pi- wedge
    for theta in np.arange(radian_start,radian_start+2*np.pi-wedge,\
                           theta_step):
        x = radius * np.cos(theta)+0.5;
        y = radius * np.sin(theta)+0.5;
        xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
        b += mag * torch.exp(-width * xx_sq_c)    
        
    kh_fun = -kh**2 * (1 - b)
    return kh_fun.unsqueeze(-1)
    

def bfield_gaussian_bumps(xx,kh):
    
    mag = 1.0; width = 2000
    xx_sq_c0 = (xx[:,0] - 0.5)**2 + (xx[:,1] - 0.5)**2
    
    xx_sq_c1 = (xx[:,0] - 0.25)**2 + (xx[:,1] - 0.5)**2
    b = mag * torch.exp(-width * xx_sq_c0) + mag * torch.exp(-width * xx_sq_c1)
    
    kh_fun = -kh**2 * (1 - b)
    return kh_fun.unsqueeze(-1)

def bfield_cavity_scattering(xx,kh):
    
    xx_sq = (xx[:,0] - 0.5)**2 + (xx[:,1] - 0.5)**2
    rho   = torch.sqrt(xx_sq)
    phi   = torch.arctan2(xx[:,1]-0.5,xx[:,0]-0.5)
    
    
    b = (1 - torch.sin(0.5*phi)**200) * torch.exp(-1000*(0.1-rho**2)**2)
    
    kh_fun = -kh**2 * (1 - b)
    return kh_fun.unsqueeze(-1)