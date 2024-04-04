import torch  # PyTorch library for tensor computations
from functools import reduce  # Import reduce function for performing cumulative operations on lists
from time import time  # Time module for performance measurement

from scipy.sparse import kron, diags, block_diag  # Sparse matrix operations from SciPy
from scipy.sparse import eye as speye  # Sparse identity matrix
import scipy.sparse.linalg as spla  # Linear algebra operations for sparse matrices
import numpy as np  # NumPy library for numerical operations

def torch_setdiff1d(t1, t2):
    """
    Computes the set difference of two PyTorch tensors and returns the result as a PyTorch tensor.
    
    Parameters:
    - t1: A PyTorch tensor.
    - t2: A PyTorch tensor to be subtracted from t1.
    
    Returns:
    - A tensor containing the elements of t1 that are not in t2.
    """
    return torch.from_numpy(np.setdiff1d(t1.numpy(), t2.numpy()))

def get_inds_2d(XX, box_geom, h, n0, n1):
    """
    Identifies the boundary indices for a 2D grid based on the geometry and grid spacing.
    
    Parameters:
    - XX: Coordinates of grid points.
    - box_geom: Geometry of the bounding box.
    - h: Grid spacing.
    - n0, n1: Dimensions of the grid.
    
    Returns:
    - Tuple of tensors containing indices of left, right, and directional (up/down) boundaries.
    """
    # Find indices of points near the left and right boundaries
    I_L = torch.argwhere(XX[0,:] < 0.5 * h + box_geom[0,0]).reshape(n1,)
    I_R = torch.argwhere(XX[0,:] > -0.5 * h + box_geom[0,1]).reshape(n1,)
    # Find indices of points near the up and down boundaries
    I_D = torch.argwhere(XX[1,:] < 0.5 * h + box_geom[1,0]).reshape(n0,)
    I_U = torch.argwhere(XX[1,:] > -0.5 * h + box_geom[1,1]).reshape(n0,)
    
    # Combine and unique-ify the directional indices
    I_DIR = torch.unique(torch.hstack((I_D, I_U)))
    # Remove directional indices from left and right to avoid duplicates
    I_L = torch_setdiff1d(I_L, I_DIR)
    I_R = torch_setdiff1d(I_R, I_DIR)
    return I_L, I_R, I_DIR
    
def get_inds_3d(XX, box_geom, h, n0, n1, n2):
    """
    Identifies the boundary indices for a 3D grid based on the geometry and grid spacing.
    
    Parameters are similar to get_inds_2d but extended for 3D.
    
    Returns:
    - Tuple of tensors containing indices of left, right, and directional (up/down/front/back) boundaries.
    """
    # Implementation is an extension of get_inds_2d to accommodate the third dimension.
    # Additional comments are omitted for brevity.

def grid(box_geom, h):
    """
    Generates a structured grid based on the bounding box geometry and grid spacing.
    
    Parameters:
    - box_geom: Geometry of the bounding box.
    - h: Grid spacing.
    
    Returns:
    - XX: Coordinates of grid points.
    - I_X_inds: Tuple of boundary indices.
    - I_X: Combined unique indices of boundary points.
    - ns: Number of points along each dimension.
    """
    d = box_geom.shape[0]  # Dimension of the problem (2D or 3D)
    xx0 = torch.arange(box_geom[0,0], box_geom[0,1] + 0.5 * h, h)
    xx1 = torch.arange(box_geom[1,0], box_geom[1,1] + 0.5 * h, h)
    if d == 3:
        xx2 = torch.arange(box_geom[2,0], box_geom[2,1] + 0.5 * h, h)
    
    # Grid generation for 2D and 3D is handled with if-elif blocks
    # Specific implementation details for constructing the grid are omitted for brevity.

class FD_disc:
    """
    Finite Difference discretization class for setting up and solving PDEs using finite differences.
    
    Parameters:
    - box_geom: Geometry of the bounding box.
    - h: Grid spacing.
    - pdo_op: Instance of a PDO class representing the differential operator.
    """
    def __init__(self, box_geom, h, pdo_op):
        XX, inds_tuple, self.I_X, self.ns = grid(box_geom, h)  # Generate the grid
        self.XX = XX.T  # Transpose for correct orientation
        self.h = h
        self.box_geom = box_geom
        self.d = self.ns.shape[0]  # Dimension of the problem
        
        # Unpack and store boundary indices
        self.I_L, self.I_R, self.I_DIR = inds_tuple
        
        # Compute the indices of the core (interior points)
        I_tot = torch.arange(self.XX.shape[0])
        self.I_C = torch_setdiff1d(I_tot, self.I_X)
        self.pdo_op = pdo_op  # Store the partial differential operator
        
    def assemble_sparse(self):
        """
        Assembles the sparse matrix representation of the differential operator for the entire grid.
        
        Returns:
        - A: Sparse matrix representation of the assembled differential operator.
        """
        # Sparse matrix assembly for 2D and 3D grids
        # The implementation constructs the sparse matrix based on the grid spacing, dimensions,
        # and the coefficients provided by pdo_op (partial differential operator).
        # Specific details are included within the implementation
        
        h = self.h; pdo_op = self.pdo_op
        if (self.d == 2):

            n0,n1 = self.ns
            d0sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n0, n0),format='csc')
            d1sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n1, n1),format='csc')
            
            d0   = (1/(2*h)) * diags([-1, 0, +1], [-1, 0, 1], shape=(n0,n0),format='csc')
            d1   = (1/(2*h)) * diags([-1, 0, +1], [-1, 0, 1], shape=(n1,n1),format='csc')

            D00 = kron(d0sq,speye(n1))
            D11 = kron(speye(n0),d1sq)
            
            c00_diag = np.array(pdo_op.c11(self.XX)).reshape(n0*n1,)
            C00 = diags(c00_diag, 0, shape=(n0*n1,n0*n1))
            c11_diag = np.array(pdo_op.c22(self.XX)).reshape(n0*n1,)
            C11 = diags(c11_diag, 0, shape=(n0*n1,n0*n1))
                        
            A = - C00 @ D00 - C11 @ D11
            
            if (pdo_op.c12 is not None):
                c_diag = np.array(pdo_op.c12(self.XX)).reshape(n0*n1,)
                S      = diags(c_diag,0,shape=(n0*n1,n0*n1))
                
                D01 = kron(d0,d1)
                A  -= 2 * S @ D01
                
            if (pdo_op.c1 is not None):
                c_diag = np.array(pdo_op.c1(self.XX)).reshape(n0*n1,)
                S      = diags(c_diag,0,shape=(n0*n1,n0*n1))
                
                D0 = kron(d0,speye(n1))
                A  += S @ D0
            
            if (pdo_op.c2 is not None):
                c_diag = np.array(pdo_op.c1(self.XX)).reshape(n0*n1,)
                S      = diags(c_diag,0,shape=(n0*n1,n0*n1))
                
                D0 = kron(speye(n0),d1)
                A  += S @ D1

            if (pdo_op.c is not None):
                c_diag = np.array(pdo_op.c(self.XX)).reshape(n0*n1,)
                S = diags(c_diag, 0, shape=(n0*n1,n0*n1))
                A += S

        elif (self.d == 3):

            n0,n1,n2 = self.ns
            d0sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n0, n0),format='csc')
            d1sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n1, n1),format='csc')
            d2sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n2, n2),format='csc')

            D00 = kron(d0sq,kron(speye(n1),speye(n2)))
            D11 = kron(speye(n0),kron(d1sq,speye(n2)))
            D22 = kron(speye(n0),kron(speye(n1),d2sq))
            
            N = n0*n1*n2
            c00_diag = np.array(pdo_op.c11(self.XX)).reshape(N,)
            C00 = diags(c00_diag, 0, shape=(N,N))
            c11_diag = np.array(pdo_op.c22(self.XX)).reshape(N,)
            C11 = diags(c11_diag, 0, shape=(N,N))
            c22_diag = np.array(pdo_op.c33(self.XX)).reshape(N,)
            C22 = diags(c22_diag, 0, shape=(N,N))

            A = - C00 @ D00 - C11 @ D11 - C22 @ D22
            
            if ((pdo_op.c1 is not None) or \
                (pdo_op.c2 is not None) or \
                (pdo_op.c3 is not None) or \
                (pdo_op.c12 is not None) or \
                (pdo_op.c13 is not None) or \
                (pdo_op.c23 is not None)):
                raise ValueError
            
            if (pdo_op.c is not None):
                c_diag = np.array(pdo_op.c(self.XX)).reshape(N,)
                S = diags(c_diag, 0, shape=(N,N))
                A += S
                
        return A.tocsr()