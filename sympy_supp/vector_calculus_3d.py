# -*- coding: utf-8 -*-
"""Completemenery functions for common vector calculus in sympy
"""


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


class OrthgonalCoordinates3D:
    
    def __init__(self, basis) -> None:
        assert len(basis) == 3
        self.x1, self.x2, self.x3 = basis
    
    def grad(self, scalar_in, **kwargs):
        raise NotImplementedError
    
    def div(self, vector_in, **kwargs):
        raise NotImplementedError
    
    def curl(self, vector_in, **kwargs):
        raise NotImplementedError
    
    def laplacian(self, tensor_in, rank=0, **kwargs):
        raise NotImplementedError


class CartesianCoordinates3D(OrthgonalCoordinates3D):
    
    def __init__(self, basis) -> None:
        super().__init__(basis)
    
    def grad(self, scalar_in, **kwargs):
        vector_out = (
            diff(scalar_in, self.x1, **kwargs), 
            diff(scalar_in, self.x2, **kwargs),
            diff(scalar_in, self.x3, **kwargs)
        )
        return vector_out
    
    def div(self, vector_in, **kwargs):
        return super().div(vector_in, **kwargs)
        
    def curl(self, vector_in, **kwargs):
        vector_out = (
            diff(vector_in[2], self.x2, **kwargs) - diff(vector_in[1], self.x3, **kwargs),
            diff(vector_in[0], self.x3, **kwargs) - diff(vector_in[2], self.x1, **kwargs),
            diff(vector_in[1], self.x1, **kwargs) - diff(vector_in[0], self.x2, **kwargs)
        )
        return vector_out
    
    def laplacian(self, tensor_in, rank=0, **kwargs):
        if rank == 0:
            tensor_out = diff(tensor_in, (self.x1, 2), **kwargs) + \
                diff(tensor_in, (self.x2, 2), **kwargs) + \
                diff(tensor_in, (self.x3, 2), **kwargs)
            return tensor_out
        else:
            raise NotImplementedError


class CylindricalCoordinates(OrthgonalCoordinates3D):
    
    def __init__(self, basis) -> None:
        super().__init__(basis)
    
    def grad(self, scalar_in, **kwargs):
        vector_out = (
            diff(scalar_in, self.x1, **kwargs),
            diff(scalar_in, self.x2, **kwargs)/self.x1,
            diff(scalar_in, self.x3, **kwargs)
        )
        return vector_out
    
    def div(self, vector_in, **kwargs):
        scalar_out = 1/self.x1*diff(self.x1*vector_in[0], self.x1, **kwargs) + \
            1/self.x1*diff(vector_in[1], self.x2, **kwargs) + \
            diff(vector_in[2], self.x3, **kwargs)
        return scalar_out
    
    def curl(self, vector_in, **kwargs):
        vector_out = (
            diff(vector_in[2], self.x2, **kwargs)/self.x1 - diff(vector_in[1], self.x3, **kwargs),
            diff(vector_in[0], self.x3, **kwargs) - diff(vector_in[2], self.x1, **kwargs),
            diff(self.x1*vector_in[1], self.x1, **kwargs)/self.x1 - diff(vector_in[0], self.x2, **kwargs)/self.x1
        )
        return vector_out
    
    def laplacian(self, tensor_in, rank=0, **kwargs):
        if rank == 0:
            tensor_out = diff(self.x1*diff(tensor_in, self.x1, **kwargs), self.x1, **kwargs)/self.x1 + \
                diff(tensor_in, (self.x2, 2), **kwargs)/self.x1**2 + \
                diff(tensor_in, (self.x3, 2), **kwargs)
            return tensor_out
        else:
            raise NotImplementedError


class SphericalCoordinates(OrthgonalCoordinates3D):
    
    def __init__(self, basis) -> None:
        super().__init__(basis)
    
    def grad(self, scalar_in, **kwargs):
        return super().grad(scalar_in, **kwargs)
    
    def div(self, vector_in, **kwargs):
        scalar_out = 1/self.x1**2*diff(self.x1**2*vector_in[0], self.x1, **kwargs) + \
            1/(self.x1*sympy.sin(self.x2))*diff(sympy.sin(self.x2)*vector_in[1], self.x2, **kwargs) + \
            1/(self.x1*sympy.sin(self.x2))*diff(vector_in[2], self.x3, **kwargs)
        return scalar_out
    
    def curl(self, vector_in, **kwargs):
        return super().curl(vector_in, **kwargs)
    
    def laplacian(self, tensor_in, rank=0, **kwargs):
        return super().laplacian(tensor_in, rank, **kwargs)
    
    def surface_grad(self, scalar_in, **kwargs):
        raise NotImplementedError
    
    def surface_div(self, vector_in, **kwargs):
        assert len(vector_in) == 2
        scalar_out = 1/(self.x1*sympy.sin(self.x2))*diff(sympy.sin(self.x2)*vector_in[0], self.x2, **kwargs) + \
            1/(self.x1*sympy.sin(self.x2))*diff(vector_in[1], self.x3, **kwargs)
        return scalar_out
        
