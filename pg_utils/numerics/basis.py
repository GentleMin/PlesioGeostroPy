# -*- coding: utf-8 -*-

"""
Numerical implementation of basis functions, transforms.

2024 note: the current implementation of the eigenvalue problem does not explicitly 
rely on the spectral-to-physical space transform or its inverse transforms.

Jingtao Min @ ETH Zurich 2025
"""

import numpy as np
from scipy import special as specfun
from typing import Optional
from . import special, utils


class Grid1D:
    
    def __init__(self, interval: np.ndarray, points: np.ndarray, weights: Optional[np.ndarray] = None):
        self.range = interval
        self.pts = points
        self.wts = weights


class FourierGrid1D(Grid1D):
    
    def __init__(self, Kmax: int, interval: Optional[np.ndarray] = None, prec: Optional[int] = None
    ):
        if prec is None:
            L = interval[-1] - interval[0]
            N = 2*Kmax + 1
            points = (L/N)*np.arange(N)
            weights = 1./N
        super().__init__(interval, points, weights)


class SpectralBasisSpace1D:
    
    def __init__(self, N: int, interval: Optional[np.ndarray] = None, grid: Optional[Grid1D] = None) -> None:
        self.N = N
        self.range = interval
        self.grid = grid
    
    def matrix_fwd(self, grid: Optional[Grid1D] = None, prec: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError
    
    def matrix_bwd(self, grid: Optional[Grid1D] = None, prec: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError
    
    def transform_fwd(self, 
        values: np.ndarray, grid: Optional[Grid1D] = None, 
        prec: Optional[int] = None
    ) -> np.ndarray:
        return self.matrix_fwd(grid=grid, prec=prec) @ values
    
    def transform_bwd(self, 
        coeffs: np.ndarray, grid: Optional[Grid1D] = None, 
        prec: Optional[int] = None
    ) -> np.ndarray:
        return self.matrix_bwd(grid=grid, prec=prec) @ coeffs
    
    def mat_phys_to_spec(self, *args, **kwargs):
        return self.matrix_fwd(*args, **kwargs)
    
    def mat_spec_to_phys(self, *args, **kwargs):
        return self.matrix_bwd(*args, **kwargs)
    
    def mat_int(self, *args, **kwargs):
        return self.matrix_fwd(*args, **kwargs)
    
    def mat_proj(self, *args, **kwargs):
        return self.matrix_bwd(*args, **kwargs)

    def phys_to_spec(self, *args, **kwargs):
        return self.transform_fwd(*args, **kwargs)
    
    def spec_to_phys(self, *args, **kwargs):
        return self.transform_bwd(*args, **kwargs)
        
    def integrate(self, *args, **kwargs):
        return self.transform_fwd(*args, **kwargs)
    
    def project(self, *args, **kwargs):
        return self.transform_bwd(*args, **kwargs)
    

class ChebyshevT(SpectralBasisSpace1D):
    
    @classmethod
    def quadrature_grid(cls, N_quad, prec=None):
        if prec is not None:
            return NotImplementedError
        x_quad, wt_quad = specfun.roots_chebyt(N_quad)
        return x_quad, wt_quad
    
    def __init__(self, N, interval = None, grid = None):
        super().__init__(N, interval, grid)


class JacobiPolar_2side(SpectralBasisSpace1D):
    
    @classmethod
    def quadrature_grid(cls, N_quad, a, b, mode='jacobi', prec=None):
        """ Generate quadrature grid for the two-sided polar Jacobi polynomial
        """
        
        if mode == 'jacobi':
            a_quad, b_quad = a, b
            # x_quad, wt_quad = specfun.roots_jacobi(N_quad, a_quad, b_quad)
            roots = special.roots_jacobi_mp(N_quad, *utils.to_mpmath_f(np.array([a_quad, b_quad]), prec=prec).tolist())
            x_quad, wt_quad = roots.xi, roots.wt
        elif mode == 'chebyt':
            a_quad, b_quad = -1/2, -1/2
            x_quad, wt_quad = specfun.roots_chebyt(N_quad)
        elif mode == 'lowest':
            a_quad = 0 if np.isclose(a, np.round(a)) else -1/2
            b_quad = 0 if np.isclose(b, np.round(b)) else -1/2
            # x_quad, wt_quad = specfun.roots_jacobi(N_quad, a_quad, b_quad)
            roots = special.roots_jacobi_mp(N_quad, *utils.to_mpmath_f(np.array([a_quad, b_quad]), prec=prec).tolist())
            x_quad, wt_quad = roots.xi, roots.wt
        return x_quad, wt_quad, a_quad, b_quad
        
    def __init__(self, N, k1, k2, a, b, qmode='jacobi', dealias=1, prec=None):
        
        int_prec = 85 if prec is None else prec

        N_quad = int(np.round(dealias*N))
        x_quad, wt_quad, a_quad, b_quad = self.quadrature_grid(N_quad, a, b, mode=qmode, prec=int_prec)
        if prec is None:
            x_quad = utils.to_numpy_f(x_quad)
            wt_quad = utils.to_numpy_f(wt_quad)*(
                np.power(2, k1 + k2 - a - b - 2)*
                np.power(1 - x_quad, a - k1 - a_quad)*
                np.power(1 + x_quad, b - k2 - b_quad)
            )
        else:
            raise NotImplementedError
        
        super().__init__(N, np.array([-1., +1.]), x_quad)
        self.k1 = k1
        self.k2 = k2
        self.a = a
        self.b = b
        self.a_quad = a_quad
        self.b_quad = b_quad
        self.N_quad = N_quad
        self.wt_quad = wt_quad
        self.prec = prec
        self.int_prec = int_prec
        self.grd_Phi = None
        self.Phi = None
        
    def jacobi_matrix(self, grid = None, prec = None):
        
        grid = self.grid if grid is None else grid
        if prec is None:
            Phi = special.eval_jacobi_recur_Nmax(self.N-1, self.a, self.b, grid)
        else:
            Phi = special.eval_jacobi_recur_gmpy2(self.N-1, 
                *utils.to_gmpy2_f(np.array([self.a, self.b]), prec=prec).tolist(),
                utils.to_gmpy2_f(grid, prec=prec)
            )
        return Phi
    
    def cache_Phi(self, grid = None, prec = None):
        
        grid = self.grid if grid is None else grid
        Phi = self.jacobi_matrix(grid=grid, prec=prec)
        self.grd_Phi = grid
        self.Phi = utils.to_numpy_f(Phi)
        
    def matrix_gram(self, quad = False, prec = None, diag = True):
        
        if quad:
            Phi = special.eval_jacobi_recur_Nmax(self.N-1, self.a, self.b, self.grid)
            pref = np.power(np.sqrt((1 - self.grid)/2), self.k1)*np.power(np.sqrt((1 + self.grid)/2), self.k2)
            D = (Phi*(pref**2*self.wt_quad)) @ Phi.T
            if diag:
                D = np.diag(D)
        else:
            D = np.zeros(self.N)
            D[0] = specfun.gamma(self.a + 1)*specfun.gamma(self.b + 1)/(2*specfun.gamma(self.a + self.b + 2))
            n_arr = np.arange(1, self.N)
            D[1:] = specfun.poch(n_arr + 1, self.a)/(2*(2*n_arr + (self.a + self.b + 1))*specfun.poch(n_arr + self.b + 1, self.a))
            if not diag:
                D = np.diag(D)
        return D
    
    def matrix_fwd(self, grid = None, prec = None):
        
        Phi = special.eval_jacobi_recur_Nmax(self.N-1, self.a, self.b, self.grid)
        pref = np.power(np.sqrt((1 - self.grid)/2), self.k1)*np.power(np.sqrt((1 + self.grid)/2), self.k2)
        d = self.matrix_gram(quad=False, prec=prec, diag=True)
        Phi = ((Phi*(pref*self.wt_quad)).T/d).T
        return Phi
    
    def transform_fwd(self, values, grid = None, prec = None):
        
        if self.grd_Phi is not None and np.allclose(self.grid, self.grd_Phi):
            Phi = self.Phi
        else:
            Phi = utils.to_numpy_f(self.jacobi_matrix(grid=self.grid))
        
        pref = np.power(np.sqrt((1 - self.grid)/2), self.k1)*np.power(np.sqrt((1 + self.grid)/2), self.k2)
        d = self.matrix_gram(quad=False, prec=prec, diag=True)
        coeffs = (Phi @ (pref*self.wt_quad*values))/d
        return coeffs
    
    def matrix_bwd(self, grid = None, prec = None):
        
        grid = self.grid if grid is None else grid
        Phi = special.eval_jacobi_recur_Nmax(self.N-1, self.a, self.b, grid)
        Phi = Phi*np.power(np.sqrt((1 - grid)/2), self.k1)*np.power(np.sqrt((1 + grid)/2), self.k2)
        return Phi.T
    
    def transform_bwd(self, coeffs, grid = None, prec = None):
        
        grid = self.grid if grid is None else grid
        pref = np.power(np.sqrt((1 - grid)/2), self.k1)*np.power(np.sqrt((1 + grid)/2), self.k2)
        
        if self.grd_Phi is not None and np.allclose(grid, self.grd_Phi):
            Phi = self.Phi
        else:
            Phi = utils.to_numpy_f(self.jacobi_matrix(grid=grid))
        
        values = pref*(Phi.T @ coeffs)
        return values


class Worland(JacobiPolar_2side):
    
    def __init__(self, N, l, qmode='jacobi', dealias=1):
        super().__init__(N, 0, l, -1/2, l-1/2, qmode=qmode, dealias=dealias)
