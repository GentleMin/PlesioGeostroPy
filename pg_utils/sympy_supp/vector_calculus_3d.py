# -*- coding: utf-8 -*-
"""
Completemenery functions for common vector calculus in sympy
"""


from typing import Any
import sympy
from sympy import diff


def dot(vec_a, vec_b):
    assert len(vec_a) == len(vec_b)
    product = sum([vec_a[idx]*vec_b[idx] for idx in range(len(vec_a))])
    return product

def cross(vec_a, vec_b):
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
    """
    
    def __init__(self, x1, x2, x3, x1_name=None, x2_name=None, x3_name=None) -> None:
        """Constructor
        
        :param x1: sympy.Symbol, first coordinate
        :param x2: sympy.Symbol, second coordinate
        :param x3: sympy.Symbol, third coordinate
        :param x1_name: str, optional name for x1; 
            if given, the attribute will be named by `x1_name`;
            else, the attribute will be named "x1".
        :param x2_name: see `x1_name`
        :param x3_name: see `x1_name`
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
        raise NotImplementedError
    
    def div(self, vector_in, **kwargs):
        raise NotImplementedError
    
    def curl(self, vector_in, **kwargs):
        raise NotImplementedError
    
    def laplacian(self, tensor_in, rank=0, **kwargs):
        raise NotImplementedError
    
    def transform_to(self, tensor_in, 
                     new_sys: "OrthogonalCoordinates3D", coeffs_new=False):
        raise NotImplementedError


class CartesianCoordinates3D(OrthogonalCoordinates3D):
    """Cartesian coordinates in 3D
    """
    
    def __init__(self, x, y, z, x_name='x', y_name='y', z_name='z') -> None:
        super().__init__(x, y, z, x1_name=x_name, x2_name=y_name, x3_name=z_name)
    
    def grad(self, scalar_in, **kwargs):
        vector_out = (
            diff(scalar_in, self[0], **kwargs), 
            diff(scalar_in, self[1], **kwargs),
            diff(scalar_in, self[2], **kwargs)
        )
        return vector_out
    
    def div(self, vector_in, **kwargs):
        return super().div(vector_in, **kwargs)
        
    def curl(self, vector_in, **kwargs):
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
    """Cylindrical coordinates in 3D
    """
    
    def __init__(self, s, p, z, s_name='s', p_name='p', z_name='z') -> None:
        super().__init__(s, p, z, x1_name=s_name, x2_name=p_name, x3_name=z_name)
    
    def grad(self, scalar_in, **kwargs):
        vector_out = (
            diff(scalar_in, self[0], **kwargs),
            diff(scalar_in, self[1], **kwargs)/self[0],
            diff(scalar_in, self[2], **kwargs)
        )
        return vector_out
    
    def div(self, vector_in, **kwargs):
        scalar_out = 1/self[0]*diff(self[0]*vector_in[0], self[0], **kwargs) \
            + 1/self[0]*diff(vector_in[1], self[1], **kwargs)\
            + diff(vector_in[2], self[2], **kwargs)
        return scalar_out
    
    def curl(self, vector_in, **kwargs):
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
        if rank == 0:
            tensor_out = diff(self[0]*diff(tensor_in, self[0], **kwargs), self[0], **kwargs)/self[0] \
                + diff(tensor_in, (self[1], 2), **kwargs)/self[0]**2 \
                + diff(tensor_in, (self[2], 2), **kwargs)
            return tensor_out
        else:
            raise NotImplementedError
    
    def transform_to(self, v_in, new_sys: OrthogonalCoordinates3D, 
        coeffs_new=False) -> OrthogonalCoordinates3D:
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
    """Spherical coordinates in 3D
    """
    
    def __init__(self, r, t, p, r_name='r', t_name='t', p_name='p') -> None:
        super().__init__(r, t, p, x1_name=r_name, x2_name=t_name, x3_name=p_name)
    
    def grad(self, scalar_in, **kwargs):
        return super().grad(scalar_in, **kwargs)
    
    def div(self, vector_in, **kwargs):
        scalar_out = 1/self[0]**2*diff(self[0]**2*vector_in[0], self[0], **kwargs) \
            + 1/(self[0]*sympy.sin(self[1]))*diff(sympy.sin(self[1])*vector_in[1], self[1], **kwargs) \
            + 1/(self[0]*sympy.sin(self[1]))*diff(vector_in[2], self[2], **kwargs)
        return scalar_out
    
    def curl(self, vector_in, **kwargs):
        return super().curl(vector_in, **kwargs)
    
    def laplacian(self, tensor_in, rank=0, **kwargs):
        return super().laplacian(tensor_in, rank, **kwargs)
    
    def surface_grad(self, scalar_in, **kwargs):
        raise NotImplementedError
    
    def surface_div(self, vector_in, **kwargs):
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
    """
    
    def __init__(self, tensor, coord_sys: OrthogonalCoordinates3D) -> None:
        """Constructor
        
        :param tensor: array-like, tensor elements
        :param coordinates: OrthogonalCoordinates3D, coordinate system
        """
        self.tensor = tensor
        self.coord_sys = coord_sys
        self.ndim = 3
        # Check if every rank is 3-D, and get the rank
        self.rank = self._get_rank(tensor)
    
    def _get_rank(self, tensor) -> int:
        """Computing the rank of a tensor
        """
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            if len(tensor) != self.ndim:
                raise AssertionError
            return self._get_rank(tensor[0]) + 1
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
        """Shortcut for calling the coordinate operation"""
        return self.coord_sys.grad(self.tensor, **kwargs)
    
    def laplacian(self, **kwargs):
        """Shortcut for calling the coordinate operation"""
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
        """Shortcut for calling the coordinate operation"""
        return self.coord_sys.div(self.tensor, **kwargs)
    
    def curl(self, **kwargs):
        """Shortcut for calling the coordinate operation"""
        return self.coord_sys.curl(self.tensor, **kwargs)
    
    def laplacian(self, **kwargs):
        """Shortcut for calling the coordinate operation"""
        return self.coord_sys.laplacian(self.tensor, rank=1, **kwargs)
