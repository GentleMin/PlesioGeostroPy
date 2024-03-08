# -*- coding: utf-8 -*-
"""
Computation of special functions, mostly the Jacobi polynomial

This module is for efficient evaluation of the special functions,
as this will be a computation-/memory-intensive part in quadratures
"""

import numpy as np
import mpmath as mp
import gmpy2 as gp
from scipy.special import eval_jacobi, roots_jacobi
from typing import Optional, Union, Literal

from .utils import transform_dps_prec

QUADPREC_DPS = 33
QUADPREC_PREC = 113

class RootsJacobiResult:
    """Result object of root-finding routine for Jacobi polynomials
    
    :ivar np.ndarray xi: final computational result of the roots
    :ivar np.ndarray wt: weights, dependent on the final roots
    :ivar bool flag: whether the computation has converged
    :ivar str msg: additional message
    """
    
    def __init__(self, xi: np.ndarray, wt: np.ndarray, flag: bool, msg: str) -> None:
        """
        :param np.ndarray xi: final computational result of the roots
        :param np.ndarray wt: weights, dependent on the final roots
        :param bool flag: whether the computation has converged
        :param str msg: additional message
        """
        self.xi = xi
        self.wt = wt
        self.flag = flag
        self.msg = msg
        
    def __repr__(self) -> str:
        o_str = "Jacobi polynomial root-finding result:\n"
        o_str += "Root-finding %s.\n%s" % (
            "successful" if self.flag else "failed", 
            self.msg)
        return o_str
    
    def __str__(self) -> str:
        o_str = "Jacobi polynomial root-finding result:\n"
        o_str += "flag:  %s.\nmsg: %s\nxi:\n%s\nwt:\n%s" % (
            "successful" if self.flag else "failed", 
            self.msg, 
            np.array2string(self.xi, max_line_width=50, threshold=10, edgeitems=2),
            np.array2string(self.wt, max_line_width=50, threshold=10, edgeitems=2))
        return o_str


def roots_jacobi_mp(n: int, alpha: mp.mpf, beta: mp.mpf, 
    n_dps: int = QUADPREC_DPS, extra_dps: int = 8, 
    max_iter: int = 10) -> RootsJacobiResult:
    """Multi-precision Jacobi root calculation.
    
    Calculates the nodes (=roots of Jacobi polynomial) and weights 
    for Gauss-Jacobi quadrature to arbitrary precision.
    
    :param int n: number of nodes / degree of Jacobi polynomial
    :param mpmath.mpf alpha: alpha value, with precision that is 
        consistent with the desired precision of output
    :param mpmath.mpf beta: beta value, with precision that is 
        consistent with the desired precision of output
    :param int n_dps: number of decimal digits, default=33, i.e.
        approx. quadruple precision
    :param int extra_dps: additional decimal digits during calculation
    :param int max_iter: maximum iteration, default=10
    
    :returns: `RootsJacobiResult` object, containing the calculated 
        quadrature nodes and weights, as `mpmath.mpf` wrapped in `numpy.ndarray`
    """
    
    # Initial nodes
    xi_dp, _ = roots_jacobi(n, float(alpha), float(beta))
    threshold = 1/10**n_dps
    
    # Switch working precision
    with mp.workdps(n_dps + extra_dps):
        
        np2mp = np.vectorize(lambda x: mp.mpf(x), otypes=(object,))
        wt_cf = (
            -((2*n + alpha + beta + 2)/(n + alpha + beta + 1))
            *(mp.gamma(n + alpha + 1)/mp.gamma(n + alpha + beta + 1))
            *(mp.gamma(n + beta + 1)/mp.factorial(n + 1))
            *mp.power(2, alpha + beta)
        )
        
        def f(n, xi):
            return np.array(
                [mp.jacobi(n, alpha, beta, xi_tmp) for xi_tmp in xi], dtype=object)
        def df(n, xi):
            return (n + alpha + beta + 1)/2*np.array(
                [mp.jacobi(n-1, alpha+1, beta+1, xi_tmp) for xi_tmp in xi], dtype=object)
        
        xi_mp = np2mp(xi_dp)
        xi_prev = xi_mp
        xi_mp = xi_mp - f(n, xi_mp)/df(n, xi_mp)
        
        for i_iter in range(1, max_iter):
            if max(abs(xi_prev - xi_mp)) <= threshold:
                wt_mp = wt_cf/df(n, xi_mp)/f(n+1, xi_mp)
                return RootsJacobiResult(xi_mp, wt_mp, True, 
                    "Convergence to {:d} digits after {:d} iters".format(n_dps, i_iter))
            xi_prev = xi_mp
            xi_mp = xi_mp - f(n, xi_mp)/df(n, xi_mp)
        
        wt_mp = wt_cf/df(n, xi_mp)/f(n+1, xi_mp)
        
    return RootsJacobiResult(xi_mp, wt_mp, False, 
        "Maximum iters {:d} reached without converging to {:d} digits".format(max_iter, n_dps))


def eval_jacobi_nrange(n_min: int, n_max: int, alpha: float, beta: float, 
    z: np.ndarray) -> np.ndarray:
    """Evaluate Jacobi polynomials for a range of degrees
    
    :param int n_min: minimum degree
    :param int n_max: maximum degree. It is required that `n_max` >= `n_min`
        and `n_max` >= 0 (non-negative)
    :param float alpha: alpha index for Jacobi polynomials
    :param float beta: beta index for Jacobi polynomials
    :param np.ndarray z: 1-D array of grid points where the Jacobi polynomials
        are to be evaluated; assumed to be within interval [-1, +1]
    :returns: Array with shape (n_max - n_min + 1, z.size), 
        values of the Jacobi polynomials at grid points
    """
    # Computing up to degree n_max
    assert n_min <= n_max and n_max >= 0
    if n_min == n_max or n_max == 0:
        return eval_jacobi(n_max, alpha, beta, z)
    Jacobi_vals = eval_jacobi_recur_Nmax(n_max, alpha, beta, z)
    # Retrieve desired degrees
    if n_min >= 0:
        return Jacobi_vals[n_min:, :]
    else:
        return np.r_[np.zeros((-n_min, z.size), dtype=z.dtype), Jacobi_vals]


def eval_jacobi_recur(Nmesh: np.ndarray, alpha: float, beta: float, 
    zmesh: np.ndarray) -> np.ndarray:
    """Evaluate Jacobi polynomials using recurrence relations
    
    This function is intended to maintain the same signature 
    as `scipy.special.eval_jacobi` and `sympy.jacobi`, 
    albeit several restrictions regarding the input params (see note)
    
    :param np.ndarray Nmesh: mesh for degrees *(N,Nz)*
    :param float alpha: alpha index
    :param float beta: beta index
    :param np.ndarray zmesh: mesh for evaluation grid *(N,Nz)*
    
    .. note::
    
        This function is designed in such a way that the Jacobi polynomial 
        is evaluated on a grid that remains the same for all degrees.
        Denoting the total number of degrees with *N* and total number of 
        grid points *Nz*, the standard for input parameter is as follows
        * Param `Nmesh` is of shape *(N,Nz)*; first index changes deg
        * Param `zmesh` is of shape *(N,Nz)*; second index changes z
    
    """
    assert Nmesh.shape == zmesh.shape
    
    n_array = Nmesh[:, 0]
    z_array = zmesh[0, :]
    Nmax = n_array.max()
    idx_pos = n_array > 0
    
    Jacobi_vals = np.zeros_like(zmesh)
    Jacobi_vals[n_array == 0, :] = 1.
    if Nmax >= 1:
        Jacobi_Nmax = eval_jacobi_recur_Nmax(Nmax, alpha, beta, z_array)
        Jacobi_vals[idx_pos, :] = Jacobi_Nmax[n_array[idx_pos], :]
        
    return Jacobi_vals


def eval_jacobi_recur_Nmax(Nmax: int, alpha: float, beta: float, 
    z: np.ndarray) -> np.ndarray:
    """Evaluate Jacobi polynomials with recurrence relation up to a degree
    
    This functions generates values for Jacobi polynomials from degree 0
    up to a specified degree, using recurrence relations.
    
    :param int Nmax: maximum degree, required to be >= 1
    :param float alpha: alpha index for Jacobi polynomials
    :param float beta: beta index for Jacobi polynomials
    :param np.ndarray z: 1-D array of grid points where the Jacobi polynomials
        are to be evaluated; assumed to be within interval [-1, +1]
    :returns: Array with shape (Nmax + 1, z.size), values for Jacobi
        polynomials at grid points specified in `z`.
    """
    assert Nmax >= 1
    # Set computing degrees
    n_array = np.arange(Nmax + 1)                                   # O(N)
    # Initializing the matrix: N * M
    Jacobi_vals = np.zeros((n_array.size, z.size), dtype=z.dtype)   # O(MN)
    # Start from non-negative degrees
    Jacobi_vals[0, :] = 1.
    Jacobi_vals[1, :] = (alpha - beta)/2 + (alpha + beta + 2)/2*z
    if Nmax == 1:
        return Jacobi_vals
    
    # # Use recurrence relations
    # # Computation O(kN) + Memory O(k)
    # # This is so far the fastest implementation with O(k) extra memory cost
    # alpha_beta_diffsqr = (alpha + beta)*(alpha - beta)
    # for i_row in range(idx+2, n_array.size):
    #     n = float(n_array[i_row])
    #     a = n + alpha
    #     b = n + beta
    #     c = a + b
    #     cf_0 = 2.*n*(c - n)*(c - 2.)
    #     cf_1 = (c - 1)*(c*(c - 2)*z + alpha_beta_diffsqr)
    #     # cfs_1 = (c - 1)*c*(c - 2)*z + (c - 1)*alpha_beta_diffsqr
    #     # cfs_1 = (c*(c*(c - 3) + 2)*z + (c - 1)*alpha_beta_diffsqr)
    #     cf_2 = -2.*(a - 1.)*(b - 1.)*c
    #     Jacobi_vals[i_row, :] = (cf_1*Jacobi_vals[i_row-1] + cf_2*Jacobi_vals[i_row-2])/cf_0
    
    # # Use recurrence relations
    # # Computation O(kN) + extra Memory O(kN)
    # # Buying computational efficiency with extra memory cost
    alpha_beta_diffsqr = alpha**2 - beta**2
    n_float = n_array[2:].astype(np.float64)
    a_array = n_float + alpha
    b_array = n_float + beta
    c_array = a_array + b_array
    cf_0_array = 2.*n_float*(c_array - n_float)*(c_array - 2.)
    cf_1_array = np.transpose(((c_array - 1)/cf_0_array)*(np.outer(z, c_array*(c_array - 2)) + alpha_beta_diffsqr))
    # cf_1_array = np.transpose(((c_array - 1)*alpha_beta_diffsqr + np.outer(z, c_array*(c_array - 1)*(c_array - 2)))/cf_0_array)
    cf_2_array = -2.*(a_array - 1.)*(b_array - 1.)*c_array/cf_0_array
    for i_n in range(n_array.size - 2):
        Jacobi_vals[i_n+2, :] = cf_1_array[i_n]*Jacobi_vals[i_n+1] + cf_2_array[i_n]*Jacobi_vals[i_n]
    
    return Jacobi_vals


def eval_jacobi_recur_mp(Nmesh: np.ndarray, 
    alpha: Union[mp.mpf, gp.mpfr], beta: Union[mp.mpf, gp.mpfr], zmesh: np.ndarray, 
    dps: int = QUADPREC_DPS, backend: Literal["mpmath", "gmpy2"] = "gmpy2") -> np.ndarray:
    """Evaluate Jacobi polynomials using recurrence relations to arb prec
    
    This function is intended to maintain the same signature 
    as `scipy.special.eval_jacobi` and `sympy.jacobi`, 
    albeit several restrictions regarding the input params (see note)
    
    :param np.ndarray Nmesh: mesh for degrees *(N,Nz)*
    :param mpmath.mpf alpha: alpha index
    :param mpmath.mpf beta: beta index
    :param np.ndarray zmesh: mesh for evaluation grid *(N,Nz)*
    :param int dps: number of decimal places for calculation, default to 113.
    :param Literal["mpmath", "gmpy2"] backend: backend for calculation.
        Default to "gmpy2".
    
    .. note::
    
        This function is designed in such a way that the Jacobi polynomial 
        is evaluated on a grid that remains the same for all degrees.
        Denoting the total number of degrees with *N* and total number of 
        grid points *Nz*, the standard for input parameter is as follows
        * Param `Nmesh` is of shape *(N,Nz)*; first index changes deg
        * Param `zmesh` is of shape *(N,Nz)*; second index changes z
        
    .. note::
    
        The input parameters `alpha` and `beta` as well as `zmesh` 
        need to at least match the precision of the desired output, 
        otherwise the multi-precision evaluation is meaningless.
    """
    assert Nmesh.shape == zmesh.shape
    Nmesh = Nmesh.astype(np.int32)
    
    n_array = Nmesh[:, 0]
    z_array = zmesh[0, :]
    Nmax = n_array.max()
    idx_pos = n_array > 0
    
    if backend == "mpmath":
    
        with mp.workdps(dps):
            Jacobi_vals = np.full(zmesh.shape, mp.mpf("0."))
            Jacobi_vals[n_array == 0, :] = mp.mpf("1.")
        
        if Nmax >= 1:
            Jacobi_Nmax = eval_jacobi_recur_mpmath(Nmax, alpha, beta, z_array, dps=dps)
            Jacobi_vals[idx_pos, :] = Jacobi_Nmax[n_array[idx_pos], :]
    
    elif backend == "gmpy2":
        
        _, prec = transform_dps_prec(dps=dps)
        with gp.local_context(gp.context(), precision=prec):
            Jacobi_vals = np.full(zmesh.shape, gp.mpfr("0.", prec))
            Jacobi_vals[n_array == 0, :] = gp.mpfr("1.", prec)
        
        if Nmax >= 1:
            Jacobi_Nmax = eval_jacobi_recur_gmpy2(Nmax, alpha, beta, z_array, prec=prec)
            Jacobi_vals[idx_pos, :] = Jacobi_Nmax[n_array[idx_pos], :]
    
    else:
        raise TypeError
        
    return Jacobi_vals


def eval_jacobi_recur_mpmath(Nmax: int, alpha: mp.mpf, beta: mp.mpf, 
    z: np.ndarray, dps: int = QUADPREC_DPS) -> np.ndarray:
    """Evaluate Jacobi polynomials with recurrence relation up to a degree, 
    to (arbitrary) multi-precision.
    
    This functions generates values for Jacobi polynomials from degree 0
    up to a specified degree, using recurrence relations.
    
    :param int Nmax: maximum degree, required to be >= 1
    :param mpmath.mpf alpha: alpha index for Jacobi polynomials
    :param mpmath.mpf beta: beta index for Jacobi polynomials
    :param np.ndarray z: 1-D array of grid points where the Jacobi polynomials
        are to be evaluated; assumed to be within interval [-1, +1]
    :param int dps: number of decimal places for calculation
    :returns: Array with shape (Nmax + 1, z.size), values for Jacobi
        polynomials at grid points specified in `z`.
    
    .. note::
    
        The input parameters `alpha` and `beta` as well as `z` 
        need to at least match the precision of the desired output, 
        otherwise the multi-precision evaluation is meaningless.
    """
    assert Nmax >= 1
    # Set computing degrees
    n_array = np.arange(Nmax + 1)
    
    Jacobi_vals = np.zeros((n_array.size, z.size), dtype=object)
    
    # Arbitrary-precision calculations are best to be wrapped 
    # in local environments whose precision is fixed.
    with mp.workdps(dps):
        
        Jacobi_vals[0, :] = mp.mpf("1.")
        # Jacobi_vals[1, :] = [mp.jacobi(1, alpha, beta, z_tmp) for z_tmp in z]
        Jacobi_vals[1, :] = (alpha - beta)/2 + (alpha + beta + 2)/2*z
        if Nmax == 1:
            return Jacobi_vals
        
        # # Use recurrence relations
        # # Computation O(kN) + extra Memory O(kN)
        # # Buying computational efficiency with extra memory cost
        n_trunc = n_array[2:]
        a_array = n_trunc + alpha
        b_array = n_trunc + beta
        c_array = a_array + b_array
        cf_0_array = 2*n_trunc*(c_array - n_trunc)*(c_array - 2)
        cf_1_array = np.transpose(((c_array - 1)/cf_0_array)
            *(np.outer(z, c_array*(c_array - 2)) + (alpha**2 - beta**2)))
        cf_2_array = -2*(a_array - 1)*(b_array - 1)*c_array/cf_0_array
        for i_n in range(n_array.size - 2):
            Jacobi_vals[i_n+2, :] = (
                cf_1_array[i_n]*Jacobi_vals[i_n+1] 
                + cf_2_array[i_n]*Jacobi_vals[i_n]
            )
    
    return Jacobi_vals


def eval_jacobi_recur_gmpy2(Nmax: int, alpha: gp.mpfr, beta: gp.mpfr, 
    z: np.ndarray, prec: int = QUADPREC_PREC) -> np.ndarray:
    """Evaluate Jacobi polynomials with recurrence relation up to a degree, 
    to (arbitrary) multi-precision, array operations using gmpy2.
    
    This functions generates values for Jacobi polynomials from degree 0
    up to a specified degree, using recurrence relations.
    
    :param int Nmax: maximum degree, required to be >= 1
    :param gmpy2.mpfr alpha: alpha index for Jacobi polynomials
    :param gmpy2.mpfr beta: beta index for Jacobi polynomials
    :param np.ndarray z: 1-D array of grid points where the Jacobi polynomials
        are to be evaluated; assumed to be within interval [-1, +1]
    :param int prec: precision (no. of binary digits) for calculation
    :returns: Array with shape (Nmax + 1, z.size), values for Jacobi
        polynomials at grid points specified in `z`.
    
    .. note::
    
        The input parameters `alpha` and `beta` as well as `z` 
        need to at least match the precision of the desired output, 
        otherwise the multi-precision evaluation is meaningless.
    """
    assert Nmax >= 1
    # Set computing degrees
    n_array = np.arange(Nmax + 1)
        
    Jacobi_vals = np.zeros((n_array.size, z.size), dtype=object)
    
    # Arbitrary-precision calculations are best to be wrapped 
    # in local environments whose precision is fixed.
    with gp.local_context(gp.context(), precision=prec):
        
        Jacobi_vals[0, :] = gp.mpfr("1.", prec)
        Jacobi_vals[1, :] = (alpha - beta)/2 + (alpha + beta + 2)/2*z
        if Nmax == 1:
            return Jacobi_vals
        
        # # Use recurrence relations
        # # Computation O(kN) + extra Memory O(kN)
        # # Buying computational efficiency with extra memory cost
        n_trunc = n_array[2:]
        a_array = n_trunc + alpha
        b_array = n_trunc + beta
        c_array = a_array + b_array
        cf_0_array = 2*n_trunc*(c_array - n_trunc)*(c_array - 2)
        cf_1_array = np.transpose(((c_array - 1)/cf_0_array)
            *(np.outer(z, c_array*(c_array - 2)) + (alpha**2 - beta**2)))
        cf_2_array = -2*(a_array - 1)*(b_array - 1)*c_array/cf_0_array
        for i_n in range(n_array.size - 2):
            Jacobi_vals[i_n+2, :] = (
                cf_1_array[i_n]*Jacobi_vals[i_n+1] 
                + cf_2_array[i_n]*Jacobi_vals[i_n]
            )
    
    return Jacobi_vals
