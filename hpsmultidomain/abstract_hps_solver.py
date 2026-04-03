import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty

class AbstractHPSSolver(metaclass=ABCMeta):
    """
    Abstract base class defining the interface for PDE solvers using the HPS framework.
    Subclasses must implement properties for geometry, indexing, and block matrices.
    """

    #################################################
    # Abstract properties defining essential data
    #################################################

    @abstractproperty
    def geom(self):
        """
        Geometry object containing domain bounds and, optionally, a parameterization map.
        Must have attribute `bounds` of shape (2, ndim).
        """
        pass

    @property
    def ndim(self):
        """
        Number of spatial dimensions, inferred from geometry bounds.
        """
        return self.geom.bounds.shape[-1]

    @abstractproperty
    def XX(self):
        """
        Flattened array of coordinates for all boundary (exterior) nodes:
        """
        pass

    @abstractproperty
    def XXfull(self):
        """
        Flattened array of coordinates for all discretization (including interiors) nodes:
        """
        pass

    @abstractproperty
    def p(self):
        """
        Polynomial degree used in each patch (number of Chebyshev nodes per direction).
        """
        pass

    @abstractproperty
    def Ji(self):
        """
        Index array for interior (duplicated interface) points in the global boundary ordering.
        """
        pass

    @abstractproperty
    def Jx(self):
        """
        Index array for unique exterior (non‐duplicated) boundary points.
        """
        pass

    @abstractproperty
    def npoints_dim(self):
        """
        Total number of Chebyshev points per dimension (npan_dim * p for each dimension).
        """
        pass

    #################################################
    # Abstract properties defining Schur complement blocks
    #################################################

    @abstractproperty
    def Aii(self):
        """
        Sparse matrix block coupling interior‐interior (duplicated interface) degrees of freedom.
        """
        pass

    @abstractproperty
    def Aix(self):
        """
        Sparse matrix block coupling interior (duplicated interface) to unique exterior DOFs.
        """
        pass

    @abstractproperty
    def Axx(self):
        """
        Sparse matrix block coupling unique exterior DOFs to themselves.
        """
        pass

    @abstractproperty
    def Axi(self):
        """
        Sparse matrix block coupling unique exterior DOFs to interior (duplicated interface) DOFs.
        """
        pass

    #################################################
    # Utilities to solve the PDE
    #################################################
                  
    @abstractproperty
    def solve_dir_full(self, uu_dir, ff_body=None):
      # Return the solution on XX_full
      pass

    @abstractproperty
    def verify_discretization(self, kh):
      pass
