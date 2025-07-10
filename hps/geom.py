from abc import ABCMeta, abstractmethod, abstractproperty
import jax.numpy as jnp
from hps.pdo import PDO2d, PDO3d

############################################################################################
# Abstract base class for geometries: defines minimal interface (bounds property)
############################################################################################
class AbstractGeometry(metaclass=ABCMeta):
    @abstractproperty
    def bounds(self):
        """
        Returns:
            A 2×ndim array specifying the lower and upper bounds of the domain.
        """
        pass


############################################################################################
# BoxGeometry: simple axis-aligned box defined entirely by its bounds
############################################################################################
class BoxGeometry(AbstractGeometry):
    def __init__(self, box_geom):
        """
        Parameters:
            box_geom: numpy.ndarray of shape (2, ndim), where
                      box_geom[0] = lower-left corner in each dimension,
                      box_geom[1] = upper-right corner in each dimension.
        """
        self.box_geom = box_geom

    @property
    def bounds(self):
        """
        Return the stored box bounds (2×ndim array).
        """
        return self.box_geom


#############################################################################################
# ParametrizedGeometry2D: maps a reference rectangular domain to a curved 2D domain
#############################################################################################
class ParametrizedGeometry2D(AbstractGeometry):
    """
    Implements a 2D geometry defined by mappings between a reference rectangle and a curved domain.

    The reference (parameter) domain is the rectangle given by `box_geom` (2×2 array).
    - `zz` contains the forward mapping functions (z1, z2) that map (x_ref, y_ref) → (x_curved, y_curved).
    - `yy` contains the inverse mapping functions (y1, y2) from the curved domain back to reference coordinates.
    - `yy_deriv` holds all needed first and second partial derivatives of y1 and y2,
      used to construct transformed PDE coefficients for a Helmholtz or Laplace operator.
    """

    def __init__(
        self,
        box_geom,
        z1,
        z2,
        y1,
        y2,
        y1_d1=None,
        y1_d2=None,
        y2_d1=None,
        y2_d2=None,
        y1_d1d1=None,
        y1_d2d2=None,
        y2_d1d1=None,
        y2_d2d2=None,
    ):
        """
        Parameters:
            box_geom: 2×2 array defining reference rectangle bounds.
            z1, z2:   Functions taking a JAX array of shape (...,2) in reference coords → x_curved, y_curved.
            y1, y2:   Inverse mapping functions on the curved domain, returning reference coords.
            y1_d1, y1_d2, y2_d1, y2_d2:   First partial derivatives of y1,y2 wrt x_curved and y_curved.
            y1_d1d1, y1_d2d2, y2_d1d1, y2_d2d2: Second partial derivatives needed for transformed PDE.
        """
        self.box_geom = box_geom
        # Store forward mapping functions (z1, z2)
        self.zz = (z1, z2)
        # Store inverse mapping functions (y1, y2)
        self.yy = (y1, y2)
        # Store derivatives of inverse maps:
        #   y1_d1: ∂y1/∂x_curved,  y1_d2: ∂y1/∂y_curved, etc.
        #   y1_d1d1: ∂²y1/∂x_curved², y1_d2d2: ∂²y1/∂y_curved², etc.
        self.yy_deriv = (
            y1_d1,
            y1_d2,
            y2_d1,
            y2_d2,
            y1_d1d1,
            y1_d2d2,
            y2_d1d1,
            y2_d2d2,
        )

    @property
    def bounds(self):
        """
        Return the reference rectangle bounds (2×2 array).
        """
        return self.box_geom

    @property
    def parameter_map(self):
        """
        Return a function that maps reference coordinates xx (shape (...,2)) to curved coordinates.

        Usage:
            param_map = geometry.parameter_map
            curved_coords = param_map(xx)  # shape (...,2)
        """
        def param_map(xx):
            (z1, z2) = self.zz
            # Evaluate forward map at all points xx
            ZZ = jnp.stack([z1(xx), z2(xx)], axis=-1)
            return ZZ

        return param_map

    @property
    def inv_parameter_map(self):
        """
        Return a function that maps curved coordinates xx (shape (...,2)) to reference coordinates.

        Usage:
            inv_map = geometry.inv_parameter_map
            ref_coords = inv_map(xx_curved)
        """
        def inv_param_map(xx):
            (y1, y2) = self.yy
            YY = jnp.stack([y1(xx), y2(xx)], axis=-1)
            return YY

        return inv_param_map

    def transform_helmholtz_pdo(self, bfield, kh):
        """
        Construct and return a transformed 2D Helmholtz/Laplace PDO (PDO2d) on the curved domain.

        Given:
          - bfield(yy, kh): a function that returns the zeroth-order coefficient c(yy, kh) in PDE.
          - kh: wavenumber*scale (for Helmholtz; if kh=0, yields Laplace).

        We compute the transformed PDE coefficients in physical (curved) coordinates:
          c11(x_curved) = (∂y1/∂x_curved)² + (∂y1/∂y_curved)²
          c22(x_curved) = (∂y2/∂x_curved)² + (∂y2/∂y_curved)²
          c12(x_curved) = (∂y1/∂x_curved)(∂y2/∂x_curved) + (∂y1/∂y_curved)(∂y2/∂y_curved)
          c1(x_curved)  = - (∂²y1/∂x_curved² + ∂²y1/∂y_curved²)
          c2(x_curved)  = - (∂²y2/∂x_curved² + ∂²y2/∂y_curved²)
          c(x_curved)   = bfield(y(x_curved), kh)

        Returns:
            PDO2d object with fields c11, c22, c12, c1, c2, c.
        """
        # Unpack derivative functions of inverse map y1, y2
        (
            y1_d1,
            y1_d2,
            y2_d1,
            y2_d2,
            y1_d1d1,
            y1_d2d2,
            y2_d1d1,
            y2_d2d2,
        ) = self.yy_deriv

        def helper_double_deriv(derivs):
            """
            Build a function that, at point xx, computes sum_{d in derivs} [d(yy)]^2,
            where yy = inv_parameter_map(xx). If all derivs are None, returns None.
            """
            if all(d is None for d in derivs):
                return None
            else:
                def func(xx):
                    yy = self.parameter_map(xx)  # map to curved coords
                    # sum of squares of available first-derivative functions
                    return sum(d(yy) ** 2 for d in derivs if d is not None)
                return func

        def helper_single_deriv(derivs):
            """
            Build a function that, at point xx, computes - sum_{d in derivs} d(yy),
            where d are second derivatives of inverse map. If all derivs None, returns None.
            """
            if all(d is None for d in derivs):
                return None
            else:
                def func(xx):
                    yy = self.parameter_map(xx)
                    return -sum(d(yy) for d in derivs if d is not None)
                return func

        # Build c11, c22 from first derivatives of y1, y2
        c11 = helper_double_deriv([y1_d1, y1_d2])
        c22 = helper_double_deriv([y2_d1, y2_d2])

        # Build c1, c2 from second partial derivatives
        c1 = helper_single_deriv([y1_d1d1, y1_d2d2])
        c2 = helper_single_deriv([y2_d1d1, y2_d2d2])

        # Build c12 cross-term from products of first derivatives, if available
        pairs = [(y1_d1, y2_d1), (y1_d2, y2_d2)]
        if not any(a is not None and b is not None for a, b in pairs):
            c12 = None
        else:
            def c12(xx):
                yy = self.parameter_map(xx)
                result = 0
                for a, b in pairs:
                    if a is not None and b is not None:
                        result += jnp.multiply(a(yy), b(yy))
                return result

        # Zeroth-order coefficient c(x) = bfield(yy, kh)
        def c(xx):
            return bfield(self.parameter_map(xx), kh)

        # Return a PDO2d with all computed coefficient functions
        return PDO2d(c11=c11, c22=c22, c1=c1, c2=c2, c12=c12, c=c)


#############################################################################################
# ParametrizedGeometry3D: maps reference cube to a curved 3D domain
#############################################################################################
class ParametrizedGeometry3D(AbstractGeometry):
    """
    Implements a 3D geometry defined by mappings between a reference box and a curved domain.

    The reference domain is defined by `box_geom` (2×3 array). We store:
      - `zz`: forward mapping functions (z1, z2, z3) mapping reference coords → curved coords.
      - `yy`: inverse mapping functions (y1, y2, y3) mapping curved coords → reference coords.
      - `yy_deriv`: all needed first and second partial derivatives of y1, y2, y3 to build PDE coefficients.
    """

    def __init__(
        self,
        box_geom,
        z1,
        z2,
        z3,
        y1,
        y2,
        y3,
        y1_d1=None,
        y1_d2=None,
        y1_d3=None,
        y2_d1=None,
        y2_d2=None,
        y2_d3=None,
        y3_d1=None,
        y3_d2=None,
        y3_d3=None,
        y1_d1d1=None,
        y1_d2d2=None,
        y1_d3d3=None,
        y2_d1d1=None,
        y2_d2d2=None,
        y2_d3d3=None,
        y3_d1d1=None,
        y3_d2d2=None,
        y3_d3d3=None,
    ):
        """
        Parameters:
            box_geom: 2×3 array defining reference box bounds.
            z1, z2, z3: Functions mapping (x_ref, y_ref, z_ref) → (x_curved, y_curved, z_curved).
            y1, y2, y3: Inverse mapping functions from curved to reference coordinates.
            The remaining parameters are first- and second-order partial derivatives of y1, y2, y3
            with respect to each curved coordinate (x_curved, y_curved, z_curved).
        """
        self.box_geom = box_geom
        # Forward map functions
        self.zz = (z1, z2, z3)
        # Inverse map functions
        self.yy = (y1, y2, y3)
        # Store all derivative functions in a flat tuple
        self.yy_deriv = (
            y1_d1,
            y1_d2,
            y1_d3,
            y2_d1,
            y2_d2,
            y2_d3,
            y3_d1,
            y3_d2,
            y3_d3,
            y1_d1d1,
            y1_d2d2,
            y1_d3d3,
            y2_d1d1,
            y2_d2d2,
            y2_d3d3,
            y3_d1d1,
            y3_d2d2,
            y3_d3d3,
        )

    @property
    def bounds(self):
        """
        Return the reference box bounds (2×3 array).
        """
        return self.box_geom

    @property
    def parameter_map(self):
        """
        Return a function that maps reference coords xx (shape (...,3)) → curved coords.
        """
        def param_map(xx):
            (z1, z2, z3) = self.zz
            ZZ = jnp.stack([z1(xx), z2(xx), z3(xx)], axis=-1)
            return ZZ

        return param_map

    @property
    def inv_parameter_map(self):
        """
        Return a function that maps curved coords xx (shape (...,3)) → reference coords.
        """
        def inv_param_map(xx):
            (y1, y2, y3) = self.yy
            YY = jnp.stack([y1(xx), y2(xx), y3(xx)], axis=-1)
            return YY

        return inv_param_map

    def transform_helmholtz_pdo(self, bfield, kh):
        """
        Construct and return a transformed 3D Helmholtz/Laplace PDO (PDO3d) on the curved domain.

        Similar to 2D, but with six second-derivative terms and three first-derivative terms.
        The transformed PDE coefficients in curved coords are:
          c11 = sum( (∂y1/∂x_curved)^2, (∂y1/∂y_curved)^2, (∂y1/∂z_curved)^2 )
          c22 = sum( (∂y2/∂x_curved)^2, (∂y2/∂y_curved)^2, (∂y2/∂z_curved)^2 )
          c33 = sum( (∂y3/∂x_curved)^2, (∂y3/∂y_curved)^2, (∂y3/∂z_curved)^2 )
          c12 = sum(∂y1/∂? * ∂y2/∂? over corresponding directions)
          c13 = sum(∂y1/∂? * ∂y3/∂? )
          c23 = sum(∂y2/∂? * ∂y3/∂? )
          c1  = - sum(second derivatives of y1)
          c2  = - sum(second derivatives of y2)
          c3  = - sum(second derivatives of y3)
          c   = bfield(yy, kh)

        Returns:
            PDO3d object with fields c11, c22, c33, c12, c13, c23, c1, c2, c3, c.
        """
        # Unpack all first and second derivative functions of inverse map y1, y2, y3
        (
            y1_d1,
            y1_d2,
            y1_d3,
            y2_d1,
            y2_d2,
            y2_d3,
            y3_d1,
            y3_d2,
            y3_d3,
            y1_d1d1,
            y1_d2d2,
            y1_d3d3,
            y2_d1d1,
            y2_d2d2,
            y2_d3d3,
            y3_d1d1,
            y3_d2d2,
            y3_d3d3,
        ) = self.yy_deriv

        def helper_double_deriv(derivs):
            """
            Create a function that computes sum_{d in derivs} [d(yy)]^2 at point xx.
            If all derivs None, returns None.
            """
            if all(d is None for d in derivs):
                return None
            else:
                def func(xx):
                    yy = self.parameter_map(xx)
                    return sum(d(yy) ** 2 for d in derivs if d is not None)
                return func

        def helper_single_deriv(derivs):
            """
            Create a function that computes - sum_{d in derivs} d(yy) at point xx.
            (Used for first-order terms in transformed PDE.)
            If all derivs None, returns None.
            """
            if all(d is None for d in derivs):
                return None
            else:
                def func(xx):
                    yy = self.parameter_map(xx)
                    return -sum(d(yy) for d in derivs if d is not None)
                return func

        def helper_mixed_deriv(pairs):
            """
            Create a function that computes sum_{(f,g) in pairs} f(yy) * g(yy) at point xx.
            If no valid pair, returns None.
            """
            if not any(f is not None and g is not None for f, g in pairs):
                return None
            else:
                def func(xx):
                    yy = self.parameter_map(xx)
                    result = 0
                    for f, g in pairs:
                        if f is not None and g is not None:
                            result += jnp.multiply(f(yy), g(yy))
                    return result
                return func

        # Build diagonal second-derivative coefficients
        c11 = helper_double_deriv([y1_d1, y1_d2, y1_d3])
        c22 = helper_double_deriv([y2_d1, y2_d2, y2_d3])
        c33 = helper_double_deriv([y3_d1, y3_d2, y3_d3])

        # Build first-order coefficients from second derivatives of inverse map
        c1 = helper_single_deriv([y1_d1d1, y1_d2d2, y1_d3d3])
        c2 = helper_single_deriv([y2_d1d1, y2_d2d2, y2_d3d3])
        c3 = helper_single_deriv([y3_d1d1, y3_d2d2, y3_d3d3])

        # Build mixed second-derivative (cross) coefficients
        c12 = helper_mixed_deriv([(y1_d1, y2_d1), (y1_d2, y2_d2), (y1_d3, y2_d3)])
        c13 = helper_mixed_deriv([(y1_d1, y3_d1), (y1_d2, y3_d2), (y1_d3, y3_d3)])
        c23 = helper_mixed_deriv([(y2_d1, y3_d1), (y2_d2, y3_d2), (y2_d3, y3_d3)])

        # Zeroth-order coefficient c(x) = bfield(yy, kh)
        def c(xx):
            return bfield(self.parameter_map(xx), kh)

        # Return a PDO3d with all transformed coefficients
        return PDO3d(
            c11=c11,
            c22=c22,
            c33=c33,
            c1=c1,
            c2=c2,
            c3=c3,
            c12=c12,
            c13=c13,
            c23=c23,
            c=c,
        )
