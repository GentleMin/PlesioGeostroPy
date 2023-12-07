# -*- coding: utf-8 -*-
"""
Completemenery functions for common vector calculus in sympy
"""


from typing import Any
import sympy
from sympy import diff


def dot(vec_a, vec_b):
    """Compute the dot product between two vectors
    
    :param array-like vec_a: left operand vector
    :param array-like vec_b: right operand vector
    
    .. note:: The method implicitly assumes `vec_a` and `vec_b` are of 
        the same length (computed via :meth:`len`), and their components
        can be multiplied and then summed.
        
    :returns: scalar product
    """
    assert len(vec_a) == len(vec_b)
    product = sum([vec_a[idx]*vec_b[idx] for idx in range(len(vec_a))])
    return product

def cross(vec_a, vec_b):
    """Compute the cross product between two vectors
    
    :param array-like vec_a: left operand vector
    :param array-like vec_b: right operand vector
    
    .. note:: The method implicitly assumes `vec_a` and `vec_b` are 
        arrays with three elements (3-D vector), and these components
        can be multiplied and then summed.
        
    :returns: vector product
    """
    assert len(vec_a) == 3
    assert len(vec_b) == 3
    product = (
        vec_a[1]*vec_b[2] - vec_a[2]*vec_b[1],
        vec_a[2]*vec_b[0] - vec_a[0]*vec_b[2],
        vec_a[0]*vec_b[1] - vec_a[1]*vec_b[0]
    )
    return product


class OrthogonalCoordinates3D:
    """Base class for orthogonal curvilinear coordinates in 3D
    
    Example: define a Cartesian system::
    
        x, y, z = sympy.symbols("x y z")
        cart3d = OrthogonalCoordinates3D(x, y, z, x1_name='x', x2_name='y', x3_name='z')
    
    `OrthogonalCoordinates3D` implements indexing. The coordinates
    can be retrieved in two different ways (following last example)::

        >>> cart3d.y    # invoking coordinate by name
        y
        >>> cart3d[1]   # invoking coordinate by index
        y
    
    Computational utilities of `OrthogonalCoordinates3D`, such as `grad`, `curl` etc.,
    are all abstract methods; therefore, the class is intended to be inherited 
    and used, instead of being used directly.
    """
    
    def __init__(self, x1, x2, x3, x1_name=None, x2_name=None, x3_name=None) -> None:
        """Constructor
        
        :param sympy.Symbol x1: first coordinate
        :param sympy.Symbol x2: second coordinate
        :param sympy.Symbol x3: third coordinate
        :param str x1_name: optional name for x1; 
            if given, the attribute will be named by `x1_name`;
            else, the attribute will be named "x1".
        :param str x2_name: see `x1_name`
        :param str x3_name: see `x1_name`        
        """
        super().__setattr__("_is_locked", False)
        self._coords = (
            "x1" if x1_name is None else x1_name,
            "x2" if x2_name is None else x2_name,
            "x3" if x3_name is None else x3_name
        )
        self.__setattr__(self._coords[0], x1)
        self.__setattr__(self._coords[1], x2)
        self.__setattr__(self._coords[2], x3)
    
    @property
    def coords(self):
        """Setter/getter interface for the coordinates, indexable
        """
        return self[0], self[1], self[2]
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        """Overriden setattr for attribute locks
        """
        if not self._is_locked:
            super().__setattr__(__name, __value)
        else:
            raise AttributeError
    
    def _lock_coordinates(self) -> None:
        """This locks the attributes and prevents from assignments
        """
        self._is_locked = True
    
    def __getitem__(self, __key: int):
        coordinate = self._coords[__key]
        return getattr(self, coordinate)
    
    def grad(self, scalar_in, **kwargs):
        """Compute gradient of a scalar
        
        abstract method, to be overriden
        """
        raise NotImplementedError
    
    def div(self, vector_in, **kwargs):
        """Compute divergence of a vector
        
        abstract method, to be overriden
        """
        raise NotImplementedError
    
    def curl(self, vector_in, **kwargs):
        """Compute curl/rot of a vector
        
        abstract method, to be overriden
        """
        raise NotImplementedError
    
    def laplacian(self, tensor_in, rank=0, **kwargs):
        """Compute Laplacian of a tensor
        
        abstract method, to be overriden
        """
        raise NotImplementedError
    
    def transform_to(self, tensor_in, 
        new_sys: "OrthogonalCoordinates3D", coeffs_new=False):
        """Transform to other coordinate systems
        
        abstract method, to be overriden
        """
        raise NotImplementedError


class CartesianCoordinates3D(OrthogonalCoordinates3D):
    """Cartesian coordinate system in 3D
    
    Example: define a Cartesian system. The default names for the coordinates
    are x, y and z::
    
        >>> x1, x2, x3 = sympy.symbols("x_1 x_2 x_3")
        >>> cart3d = CartesianCoordinates3D(x1, x2, x3)
        >>> cart3d.x   # Invoking the x coordinate
        x_1
        >>> cart3d.y   # Invoking the y coordinate
        x_2
    
    You can also pass in additional names to name the coordinates::
    
        >>> cart3d = CartesianCoordinates3D(x1, x2, x3, x_name="x1", y_name="x2", zname="x3")
        >>> cart3d.x1  # Invoking the x coordinate
        x_1
        >>> cart3d.x2  # Invoking the y coordinate
        x_2
    
    It also supports indexing::
    
        >>> cart3d[0]
        x_1
    
    """
    
    def __init__(self, x, y, z, x_name='x', y_name='y', z_name='z') -> None:
        """Constructor
        
        :param sympy.Symbol x: x coordinate
        :param sympy.Symbol y: y coordinate
        :param sympy.Symbol z: z coordinate
        :param str x_name: optional name for x; 
            if given, the attribute will be named by `x_name`;
            else, the attribute will be named "x".
        :param str y_name: see `x_name`
        :param str z_name: see `x_name`
        """
        super().__init__(x, y, z, x1_name=x_name, x2_name=y_name, x3_name=z_name)
    
    def grad(self, scalar_in, **kwargs):
        """Compute the grad of a scalar in 3-D Cartesian coords
        
        :param sympy.Expr scalar_in: input scalar
        :param \**kwargs: additional arguments passed to ``sympy.diff``
        
        :returns: gradient, 3-tuple of `sympy.Expr`, 
            with elements corresponding to (x,y,z) components
        """
        vector_out = (
            diff(scalar_in, self[0], **kwargs), 
            diff(scalar_in, self[1], **kwargs),
            diff(scalar_in, self[2], **kwargs)
        )
        return vector_out
    
    def div(self, vector_in, **kwargs):
        """Vector divergence, to be implemented"""
        return super().div(vector_in, **kwargs)
        
    def curl(self, vector_in, **kwargs):
        """Compute the curl of a vector in 3-D Cartesian coords
        
        :param array-like vector_in: input vector, assumed to be 
            3-component (x,y,z) array, with `sympy.Expr` as elements
        :param \**kwargs: additional arguments passed to ``sympy.diff``
        
        :returns: curl, 3-tuple of `sympy.Expr`, 
            with elements corresponding to (x,y,z) components
        """
        vector_out = (
            diff(vector_in[2], self[1], **kwargs) \
                - diff(vector_in[1], self[2], **kwargs),
            diff(vector_in[0], self[2], **kwargs) \
                - diff(vector_in[2], self[0], **kwargs),
            diff(vector_in[1], self[0], **kwargs) \
                - diff(vector_in[0], self[1], **kwargs)
        )
        return vector_out
    
    def laplacian(self, tensor_in, rank=0, **kwargs):
        """Compute the Laplacian of a tensor
        
        :param tensor_in: input tensor
        :param int rank: rank of the input tensor, default to 0.
            Currently, only `rank` = 0 (scalar Laplacian) is implemented.
        :param \**kwargs: additional arguments passed to ``sympy.diff``
        
        :returns: Laplacian, same shape as the input tensor
        """
        if rank == 0:
            tensor_out = diff(tensor_in, (self[0], 2), **kwargs) \
                + diff(tensor_in, (self[1], 2), **kwargs) \
                + diff(tensor_in, (self[2], 2), **kwargs)
            return tensor_out
        else:
            raise NotImplementedError

    def transform_to(self, tensor_in, 
        new_sys: OrthogonalCoordinates3D, coeffs_new=False):
        return super().transform_to(tensor_in, new_sys, coeffs_new)


class CylindricalCoordinates(OrthogonalCoordinates3D):
    """Cylindrical coordinate system in 3D
    
    Example: define a cylindrical system. The default names for the coordinates
    are s, p and z::
    
        >>> x1, x2, x3 = sympy.symbols("x_1 x_2 x_3")
        >>> cyl3d = CylindricalCoordinates3D(x1, x2, x3)
        >>> cyl3d.s   # Invoking the radial coordinate
        x_1
        >>> cyl3d.p   # Invoking the azimuthal coordinate
        x_2
    
    You can also pass in additional names to name the coordinates::
    
        >>> cyl3d = CylindricalCoordinates3D(x1, x2, x3, s_name="r")
        >>> cyl3d.r  # Invoking the radial coordinate
        x_1
        >>> cyl3d.p  # Invoking the azimuthal coordinate
        x_2
    
    It also supports indexing::
    
        >>> cyl3d[2]
        x_3
    
    """
     
    def __init__(self, s, p, z, s_name='s', p_name='p', z_name='z') -> None:
        """Constructor
        
        :param sympy.Symbol s: radial coordinate
        :param sympy.Symbol p: azimuthal coordinate
        :param sympy.Symbol z: vertical coordinate
        :param str s_name: optional name for s; 
            if given, the attribute will be named by `s_name`;
            else, the attribute will be named "s".
        :param str p_name: see `s_name`
        :param str z_name: see `s_name`
        """
        super().__init__(s, p, z, x1_name=s_name, x2_name=p_name, x3_name=z_name)
    
    def grad(self, scalar_in, **kwargs):
        """Compute the grad of a scalar in cylindrical coordinates
        
        :param sympy.Expr scalar_in: input scalar
        :param \**kwargs: additional arguments passed to ``sympy.diff``
        
        :returns: gradient, 3-tuple of `sympy.Expr`, 
            with elements corresponding to (s,p,z) components
        """
        vector_out = (
            diff(scalar_in, self[0], **kwargs),
            diff(scalar_in, self[1], **kwargs)/self[0],
            diff(scalar_in, self[2], **kwargs)
        )
        return vector_out
    
    def div(self, vector_in, **kwargs):
        """Compute the divergence of a vector in cylindrical coordinates
        
        :param array-like vector_in: input vector, assumed to be
            3-component array, with `sympy.Expr` as components
        :param \**kwargs: additional arguments passed to ``sympy.diff``
        
        :returns: divergence scalar, `sympy.Expr`
        """
        scalar_out = 1/self[0]*diff(self[0]*vector_in[0], self[0], **kwargs) \
            + 1/self[0]*diff(vector_in[1], self[1], **kwargs)\
            + diff(vector_in[2], self[2], **kwargs)
        return scalar_out
    
    def curl(self, vector_in, **kwargs):
        """Compute the curl of a vector in cylindrical coordinates
        
        :param array-like vector_in: input vector, assumed to be 
            3-component array, with `sympy.Expr` as components
        :param \**kwargs: additional arguments passed to ``sympy.diff``
        
        :returns: curl, 3-tuple of `sympy.Expr`, 
            with elements corresponding to (s,p,z) components
        """
        vector_out = (
            diff(vector_in[2], self[1], **kwargs)/self[0] \
                - diff(vector_in[1], self[2], **kwargs),
            diff(vector_in[0], self[2], **kwargs) \
                - diff(vector_in[2], self[0], **kwargs),
            diff(self[0]*vector_in[1], self[0], **kwargs)/self[0] \
                - diff(vector_in[0], self[1], **kwargs)/self[0]
        )
        return vector_out
    
    def laplacian(self, tensor_in, rank=0, **kwargs):
        """Compute the Laplacian of a tensor in cylindrical coordinates
        
        :param tensor_in: input tensor
        :param int rank: rank of the input tensor, default to 0.
            Currently, only `rank` = 0 (scalar Laplacian) is implemented.
        :param \**kwargs: additional arguments passed to ``sympy.diff``
        
        :returns: tensor, `sympy.Expr` or List of `sympy.Expr`
            depending on the input tensor rank
        """
        if rank == 0:
            tensor_out = diff(self[0]*diff(tensor_in, self[0], **kwargs), self[0], **kwargs)/self[0] \
                + diff(tensor_in, (self[1], 2), **kwargs)/self[0]**2 \
                + diff(tensor_in, (self[2], 2), **kwargs)
            return tensor_out
        else:
            raise NotImplementedError
    
    def transform_to(self, v_in, new_sys: OrthogonalCoordinates3D, 
        coeffs_new=False) -> OrthogonalCoordinates3D:
        """Transform vector under cylindrical coordinates to new coordinate system
        """
        if isinstance(new_sys, SphericalCoordinates):
            if coeffs_new:
                v1 = sympy.sin(new_sys[1])*v_in[0] + sympy.cos(new_sys[1])*v_in[2]
                v2 = sympy.cos(new_sys[1])*v_in[0] - sympy.sin(new_sys[1])*v_in[2]
                v3 = v_in[1]
            else:
                v1 = self[0]/sympy.sqrt(self[0]**2 + self[2]**2)*v_in[0] \
                    + self[2]/sympy.sqrt(self[0]**2 + self[2]**2)*v_in[2]
                v2 = self[2]/sympy.sqrt(self[0]**2 + self[2]**2)*v_in[0] \
                    - self[0]/sympy.sqrt(self[0]**2 + self[2]**2)*v_in[2]
                v3 = v_in[1]
            return Vector3D([v1, v2, v3], coord_sys=new_sys)
        return super().transform_to(v_in, new_sys, coeffs_new)


class SphericalCoordinates(OrthogonalCoordinates3D):
    """Spherical coordinate system in 3D
    
    Example: define a spherical coordinate system. The default names for the 
    coordinates are r (radius), t (colatitude) and p (azimuth)::
    
        >>> x1, x2, x3 = sympy.symbols("x_1 x_2 x_3")
        >>> sph3d = SphericalCoordinates3D(x1, x2, x3)
        >>> sph3d.r   # Invoking the radial coordinate
        x_1
        >>> sph3d.p   # Invoking the azimuthal coordinate
        x_3
    
    You can also pass in additional names to name the coordinates::
    
        >>> sph3d = SphericalCoordinates3D(x1, x2, x3, r_name="x1")
        >>> sph3d.x1  # Invoking the radial coordinate
        x_1
        >>> sph3d.p  # Invoking the azimuthal coordinate
        x_3
    
    It also supports indexing::
    
        >>> sph3d[1]
        x_2
    
    """
            
    def __init__(self, r, t, p, r_name='r', t_name='t', p_name='p') -> None:
        super().__init__(r, t, p, x1_name=r_name, x2_name=t_name, x3_name=p_name)
    
    def grad(self, scalar_in, **kwargs):
        """Scalar gradient, to be implemented
        """
        return super().grad(scalar_in, **kwargs)
    
    def div(self, vector_in, **kwargs):
        """Compute the divergence of a vector in spherical coordinates
        
        :param array-like vector_in: input vector, assumed to be
            3-component array, with `sympy.Expr` as components
        :param \**kwargs: additional arguments passed to ``sympy.diff``
        
        :returns: divergence scalar, `sympy.Expr`
        """
        scalar_out = 1/self[0]**2*diff(self[0]**2*vector_in[0], self[0], **kwargs) \
            + 1/(self[0]*sympy.sin(self[1]))*diff(sympy.sin(self[1])*vector_in[1], self[1], **kwargs) \
            + 1/(self[0]*sympy.sin(self[1]))*diff(vector_in[2], self[2], **kwargs)
        return scalar_out
    
    def curl(self, vector_in, **kwargs):
        """Vector curl, to be implemented"""
        return super().curl(vector_in, **kwargs)
    
    def laplacian(self, tensor_in, rank=0, **kwargs):
        """Tensor Laplacian, to be implemented"""
        return super().laplacian(tensor_in, rank, **kwargs)
    
    def surface_grad(self, scalar_in, **kwargs):
        """Surface gradient, to be implemented
        """
        raise NotImplementedError
    
    def surface_div(self, vector_in, **kwargs):
        """Compute the surface divergence of a 2-D vector in spherical coordinates
        
        :param array-like vector_in: input vector, assumed to be 
            2-component array, with `sympy.Expr` as components.
            These are assumed to be colatitudal (theta) and azimuthal (phi)
            components of a vector field.
        :param \**kwargs: additional arguments passed to ``sympy.diff``
        
        :returns: surface divergence, `sympy.Expr`
        """
        assert len(vector_in) == 2
        scalar_out = 1/(self[0]*sympy.sin(self[1]))*diff(sympy.sin(self[1])*vector_in[0], self[1], **kwargs) \
            + 1/(self[0]*sympy.sin(self[1]))*diff(vector_in[1], self[2], **kwargs)
        return scalar_out
    
    def transform_to(self, v_in, new_sys: OrthogonalCoordinates3D, 
            coeffs_new=False) -> OrthogonalCoordinates3D:
        if isinstance(new_sys, CylindricalCoordinates):
            if coeffs_new:
                v1 = new_sys[0]/sympy.sqrt(new_sys[0]**2 + new_sys[2]**2)*v_in[0] \
                    + new_sys[2]/sympy.sqrt(new_sys[0]**2 + new_sys[2]**2)*v_in[1]
                v2 = v_in[2]
                v3 = new_sys[2]/sympy.sqrt(new_sys[0]**2 + new_sys[2]**2)*v_in[0] \
                    - new_sys[0]/sympy.sqrt(new_sys[0]**2 + new_sys[2]**2)*v_in[1]
            else:
                v1 = sympy.sin(self[1])*v_in[0] + sympy.cos(self[1])*v_in[1]
                v2 = v_in[2]
                v3 = sympy.cos(self[1])*v_in[0] - sympy.sin(self[1])*v_in[1]
            return Vector3D([v1, v2, v3], coord_sys=new_sys)
        return super().transform_to(v_in, new_sys, coeffs_new)


class Tensor3D:
    """Base class for tensors in 3-D
    
    `Tensor3D` and the child classes provide an interface to store the components
    together with the coordinate system, thus providing the full information of 
    a tensor object.
    These classes are also designed to have methods to compute the differential
    forms of the tensor object, which really just call the corresponding methods
    in the `OrthogonalCoordinates3D` classes to compute.
    
    :param tensor: tensor elements
    :param OrthogonalCoordinates3D coord_sys: coordinate system
    :param int ndim: dimension
    :param int rank: rank of the tensor 
    """
    
    def __init__(self, tensor, coord_sys: OrthogonalCoordinates3D) -> None:
        """Constructor
        
        :param array-like tensor: tensor
        :param OrthogonalCoordinates3D coordinates: coordinate system
        """
        self.tensor = tensor
        self.coord_sys = coord_sys
        self.ndim = 3
        # Check if every rank is 3-D, and get the rank
        self.rank = self._get_rank(tensor)
    
    @staticmethod
    def _get_rank(tensor) -> int:
        """Computing the rank of a tensor
        """
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            if len(tensor) != 3:
                raise AssertionError
            return Tensor3D._get_rank(tensor[0]) + 1
        return 0

    def transform_to(self, new_sys: OrthogonalCoordinates3D, **kwargs) -> "Tensor3D":
        """Transform to another coordinates system
        """
        self.coord_sys.transform_to(self, new_sys=new_sys, **kwargs)


class Scalar3D(Tensor3D):
    """Scalar in 3D
    """
    
    def __init__(self, scalar, coord_sys: OrthogonalCoordinates3D) -> None:
        super().__init__(scalar, coord_sys)
        assert self.rank == 0
    
    def grad(self, **kwargs):
        """Compute gradient, shortcut for calling the method in `coord_sys`"""
        return self.coord_sys.grad(self.tensor, **kwargs)
    
    def laplacian(self, **kwargs):
        """Compute Laplacian, shortcut for calling the method in `coord_sys`"""
        return self.coord_sys.laplacian(self.tensor, rank=0, **kwargs)

    def transform_to(self, new_sys: OrthogonalCoordinates3D, **kwargs) -> Tensor3D:
        """Scalar remains unchanged.
        """
        return self


class Vector3D(Tensor3D):
    """Vector in 3D
    """
    
    def __init__(self, vector, coord_sys: OrthogonalCoordinates3D) -> None:
        super().__init__(vector, coord_sys)
        assert self.rank == 1
        for i_dim in range(self.ndim):
            self.__setattr__(self.coord_sys._coords[i_dim], self.tensor[i_dim])
    
    def __getitem__(self, __key):
        return self.tensor[__key]
    
    def __setitem__(self, __key, __val):
        self.tensor[__key] = __val
    
    def div(self, **kwargs):
        """Compute divergence, shortcut for calling the method in `coord_sys`"""
        return self.coord_sys.div(self.tensor, **kwargs)
    
    def curl(self, **kwargs):
        """Compute curl, shortcut for calling the method in `coord_sys`"""
        return self.coord_sys.curl(self.tensor, **kwargs)
    
    def laplacian(self, **kwargs):
        """Compute Laplacian, shortcut for calling the method in `coord_sys`"""
        return self.coord_sys.laplacian(self.tensor, rank=1, **kwargs)
