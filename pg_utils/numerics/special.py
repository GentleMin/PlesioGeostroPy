# -*- coding: utf-8 -*-
"""
Computation of special functions, mostly the Jacobi polynomial

This module is for efficient evaluation of the special functions,
as this will be a computation-/memory-intensive part in quadratures
"""

import numpy as np
from scipy.special import eval_jacobi


def eval_jacobi_nrange(n_min: int, n_max: int, alpha: float, beta: float, z: np.ndarray):
    """Evaluate Jacobi polynomial in an array
    """
    # Computing up to degree n_max
    assert n_min <= n_max and n_max >= 0
    if n_min == n_max or n_max == 0:
        return eval_jacobi(n_max, alpha, beta, z)
    Jacobi_vals = eval_jacobi_to_Nmax(n_max, alpha, beta, z)
    # Retrieve desired degrees
    if n_min >= 0:
        return Jacobi_vals[n_min:, :]
    else:
        return np.r_[np.zeros((-n_min, z.size), dtype=z.dtype), Jacobi_vals]


def eval_jacobi_to_Nmax(Nmax: int, alpha: float, beta: float, z: np.ndarray):
    """Evaluate Jacobi polynomials up to a degree
    """
    assert Nmax >= 1
    # Set computing degrees
    n_array = np.arange(Nmax + 1)                                   # O(N)
    # Initializing the matrix: N * M
    Jacobi_vals = np.zeros((n_array.size, z.size), dtype=z.dtype)   # O(MN)
    # Start from non-negative degrees
    Jacobi_vals[0, :] = eval_jacobi(0, alpha, beta, z)      # O(N) Jacobi eval
    Jacobi_vals[1, :] = eval_jacobi(1, alpha, beta, z)      # O(N) Jacobi eval
    if Nmax == 1:
        return Jacobi_vals
    
    # # Use recurrence relations for the rest
    # # Computation O(kN) + Memory O(k)
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
    
    # # Use recurrence relations for the rest
    # # Computation O(kN) + Memory O(kN)
    # # Buying computational efficiency with extra memory cost
    alpha_beta_diffsqr = (alpha + beta)*(alpha - beta)
    n_float = n_array[2:].astype(np.float64)
    a_array = n_float + alpha
    b_array = n_float + beta
    c_array = a_array + b_array
    cf_0_array = 2.*n_float*(c_array - n_float)*(c_array - 2.)
    cf_1_array = np.transpose(((c_array - 1)/cf_0_array)*(np.outer(z, c_array*(c_array - 2)) + alpha_beta_diffsqr))
    # cf_1_array = np.transpose((c_array - 1)*(np.outer(z, c_array*(c_array - 2)) + alpha_beta_diffsqr)/cf_0_array)
    cf_2_array = -2.*(a_array - 1.)*(b_array - 1.)*c_array/cf_0_array
    for i_n in range(n_array.size - 2):
        i_row = i_n + 2
        Jacobi_vals[i_row, :] = cf_1_array[i_n]*Jacobi_vals[i_row-1] + cf_2_array[i_n]*Jacobi_vals[i_row-2]
        # Jacobi_vals.append(cf_1_array[i_n]*Jacobi_vals[-1] + cf_2_array[i_n]*Jacobi_vals[-2])
    
    return Jacobi_vals


