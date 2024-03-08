# -*- coding: utf-8 -*-
"""Miscellaneous utilities for numerical calculation
"""


import numpy as np
import gmpy2 as gp
import mpmath as mp
from typing import Union, List, Optional, Callable, Any, Literal
from scipy.sparse import coo_array


"""
--------------------------------------
Analytical references
--------------------------------------

* Analytical eigenfrequencies for the PG inertial modes
* Analytical eigenfrequencies for the 3-D inertial modes
* Analytical eigenfrequencies for the Malkus bg field in PG
* Analytical eigenfrequencies for the Malkus bg field in 3D

"""


def eigenfreq_psi_op(m: Union[int, np.ndarray], n: Union[int, np.ndarray], 
    prec: Optional[int]=None) -> Union[float, np.ndarray]:
    """Analytic eigenfrequency for the self-adjoint operator
    for stream function Psi in the vorticity equation
    
    .. math:: \\omega = - \\frac{m}{n(2n + 2m + 1) + \\frac{m}{2} + \\frac{m^2}{4}}
    
    .. note:: If `prec` is specified other than None, then the input must be
        consistently in gmpy2 form in order to properly compute in multi-precision.
    """
    if prec is None:
        return -m/(n*(2*n + 2*m + 1) + m/2 + m**2/4)
    else:
        with gp.local_context(gp.context(), precision=prec):
            return -m/(n*(2*n + 2*m + 1) + m/2 + m**2/4)


def eigenfreq_inertial3d(m: Union[int, np.ndarray], n: Union[int, np.ndarray]):
    """Analytic eigenfrequency for the 3D inertial modes
    
    .. math:: 
    
        \\omega = -\\frac{2}{m+2} 
        \\left(\\sqrt{1 + \\frac{m(m+2)}{n(2n+2m+1)}} - 1\\right)
    """
    return -2/(m+2)*(np.sqrt(1 + m*(m+2)/n/(2*n+2*m+1)) - 1)


def eigenfreq_Malkus_pg(m: Union[int, np.ndarray], n: Union[int, np.ndarray], 
    Le: Union[float, gp.mpfr], mode: str="all", timescale: str="spin", prec: Optional[int] = None):
    """Analytic eigenfrequency for the PG model with Malkus bg field
    
    :param Union[int, np.ndarray] m: azimuthal wavenumber
    :param Union[int, np.ndarray] n: order of the eigenmode
    :param float Le: Lehnert number (see also :data:`~pg_utils.pg_model.params.Le` )
    :param str mode: fast or slow, default to "all"
    :param str timescale: characteristic timescale, default to "spin", 
        alternative: "alfven". See note below for more details.
    :param Optional[int] prec: precision to be computed to. 
        Default to None, calculate to double precision using numpy.
    
    :returns: eigenfrequency array(s)
    
    .. note::
    
        When using spin rate for characteristic time scale, i.e. :math:`\\tau=\\Omega^{-1}`
        
        .. math::

            \\omega = \\frac{\\omega_0}{2} 
            \\left(1 \\pm \\sqrt{\\mathrm{Le}^2 \\frac{4m(m - \\omega_0)}{\\omega_0^2}}\\right)
        
        When using Alfven time scale, i.e. :math:`\\tau=\\frac{\\sqrt{\\rho\mu_0}L}{B}`
        
        .. math::
        
            \\omega = \\frac{\\omega_0}{2\\mathrm{Le}} 
            \\left(1 \\pm \\sqrt{\\mathrm{Le}^2 \\frac{4m(m - \\omega_0)}{\\omega_0^2}}\\right)
        
        where :math:`\\omega_0` is the inertial mode eigenfrequency for the PG model,
        see :func:`eigenfreq_psi_op` for details. The plus sign gives the fast mode,
        and the minus sign gives the slow mode.
    """
    omega0 = eigenfreq_psi_op(m, n, prec=prec)
    if prec is None:
        if timescale.lower() == "spin":
            prefactor = omega0/2
        elif timescale.lower() == "alfven":
            prefactor = omega0/2/Le
        else:
            raise AttributeError
        bg_field_mod = np.sqrt(1 + Le**2*(4*m*(m - omega0))/(omega0**2))
        return prefactor*(1 + bg_field_mod), prefactor*(1 - bg_field_mod)
    else:
        with gp.local_context(gp.context(), precision=prec):
            if timescale.lower() == "spin":
                prefactor = omega0/2
            elif timescale.lower() == "alfven":
                prefactor = omega0/2/Le
            else:
                raise AttributeError
            bg_field_mod = 1 + Le**2*(4*m*(m - omega0))/(omega0**2)
            bg_field_mod = np.vectorize(gp.sqrt, otypes=(object,))(bg_field_mod)
            return prefactor*(1 + bg_field_mod), prefactor*(1 - bg_field_mod)


def eigenfreq_Malkus_3d(m: Union[int, np.ndarray], n: Union[int, np.ndarray], 
    Le: float, mode: str="all", timescale: str="spin"):
    """Analytic eigenfrequency for 3D eigemodes with Malkus bg field
    
    :param Union[int, np.ndarray] m: azimuthal wavenumber
    :param Union[int, np.ndarray] n: order of the eigenmode
    :param float Le: Lehnert number (see also :data:`~pg_utils.pg_model.params.Le` )
    :param str mode: fast or slow, default to "all"
    :param str timescale: characteristic timescale, default to "spin", 
        alternative: "alfven". See note below for more details.
    
    :returns: eigenfrequency array(s)
    
    .. note::
    
        When using spin rate for characteristic time scale, i.e. :math:`\\tau=\\Omega^{-1}`
        
        .. math::

            \\omega = \\frac{\\omega_0}{2} 
            \\left(1 \\pm \\sqrt{\\mathrm{Le}^2 \\frac{4m(m - \\omega_0)}{\\omega_0^2}}\\right)
        
        When using Alfven time scale, i.e. :math:`\\tau=\\frac{\\sqrt{\\rho\mu_0}L}{B}`
        
        .. math::
        
            \\omega = \\frac{\\omega_0}{2\\mathrm{Le}} 
            \\left(1 \\pm \\sqrt{\\mathrm{Le}^2 \\frac{4m(m - \\omega_0)}{\\omega_0^2}}\\right)
        
        where :math:`\\omega_0` is the inertial mode eigenfrequency in 3-D,
        see :func:`eigenfreq_inertial3d` for details. The plus sign gives the fast mode,
        and the minus sign gives the slow mode.
    """
    omega0 = eigenfreq_inertial3d(m, n)
    bg_field_mod = Le**2*(4*m*(m - omega0))/(omega0**2)
    if timescale.lower() == "spin":
        prefactor = omega0/2
    elif timescale.lower() == "alfven":
        prefactor = omega0/2/Le
    else:
        raise AttributeError
    if mode == "fast":
        return prefactor*(1 + np.sqrt(1 + bg_field_mod))
    elif mode == "slow":
        return prefactor*(1 - np.sqrt(1 + bg_field_mod))
    else:
        return prefactor*(1 + np.sqrt(1 + bg_field_mod)), \
            prefactor*(1 - np.sqrt(1 + bg_field_mod))


"""
------------------------------
Conversion utilities
------------------------------

* Decimal places - binary precision conversion
* Conversion between numpy, gmpy2 and mpmath dtypes (float/complex)
* Conversion between dense and sparse object arrays

"""


def transform_dps_prec(dps: Optional[int] = None, prec: Optional[int] = None, 
    dps_default: int = 16, prec_default: int = 53) -> np.ndarray:
    """Conversion between decimal points and precision
    
    The current implementation seems to be consistent with the dps-precision
    conversion in `sympy`, `mpmath` and `gmpy2`.
    Currently, 3.322 is used as a proxy for :math:`\log_2(10)`, this value seems
    accurate enough.
    """
    if dps is not None:
        return dps, int(np.round(3.322*(dps + 1)))
    elif prec is not None:
        return int(np.round(prec/3.322)) - 1, prec
    else:
        return dps_default, prec_default


def to_gpmy2_f(x: np.ndarray, dps: Optional[int] = None, 
    prec: Optional[int] = None) -> np.ndarray:
    """Convert float array to gmpy2 float array
    """
    if dps is None and prec is None:
        return np.vectorize(lambda x: gp.mpfr(str(x)), otypes=(object,))(x)
    _, prec_target = transform_dps_prec(dps=dps, prec=prec)
    with mp.workprec(prec_target):
        return np.vectorize(lambda x: gp.mpfr(str(x), prec_target), otypes=(object,))(x)


def to_mpmath_f(x: np.ndarray, dps: Optional[int] = None, 
    prec: Optional[int] = None) -> np.ndarray:
    """Convert float array to mpmath float array
    """
    if dps is None and prec is None:
        return np.vectorize(lambda x: mp.mpf(str(x)), otypes=(object,))(x)
    dps_target, _ = transform_dps_prec(dps=dps, prec=prec)
    with mp.workdps(dps_target):
        return np.vectorize(lambda x: mp.mpf(str(x)), otypes=(object,))(x)


def to_numpy_f(x: np.ndarray) -> np.ndarray:
    """Convert float array to numpy float64 array
    """
    return x.astype(np.float64)


def to_gpmy2_c(x: np.ndarray, dps: Optional[int] = None, 
    prec: Optional[int] = None) -> np.ndarray:
    """Convert float array to gmpy2 float array
    """
    if dps is None and prec is None:
        return np.vectorize(
            lambda x: gp.mpc(gp.mpfr(str(x.real)), imag=gp.mpfr(str(x.imag))), 
            otypes=(object,))(x)
    _, prec_target = transform_dps_prec(dps=dps, prec=prec)
    with mp.workprec(prec_target):
        return np.vectorize(
            lambda x: gp.mpc(gp.mpfr(str(x.real), prec_target), imag=gp.mpfr(str(x.imag), prec_target), 
                            precision=prec_target), 
            otypes=(object,))(x)


def to_mpmath_c(x: np.ndarray, dps: Optional[int] = None, 
    prec: Optional[int] = None) -> np.ndarray:
    """Convert float array to mpmath float array
    """
    if dps is None and prec is None:
        return np.vectorize(
            lambda x: mp.mpc(real=str(x.real), imag=str(x.imag)), 
            otypes=(object,))(x)
    dps_target, _ = transform_dps_prec(dps=dps, prec=prec)
    with mp.workdps(dps_target):
        return np.vectorize(
            lambda x: mp.mpc(real=str(x.real), imag=str(x.imag)), 
            otypes=(object,))(x)


def to_numpy_c(x: np.ndarray) -> np.ndarray:
    """Convert complex array to numpy complex128 array
    """
    return x.astype(np.complex128)


def array_to_str(x: np.ndarray, str_fun: Callable[[Any], str] = str) -> np.ndarray:
    """Convert array to List of strings
    
    :param Callable[[Any], str] str_fun: an optional stringify function,
        default to the string function `str`.
    
    .. warning::

        This function should really be used only for converting 1-D
        arrays to strings. Otherwise the returned object will not be 
        a full list of strings, but merely a list of arrays.
    """
    # return [str_fun(item) for item in x]
    return list(np.vectorize(str_fun, otypes=(object,))(x))


def is_eq_sparse(array_1, array_2):
    """Compare if two sparse matrices are identical"""
    return len((array_1 != array_2).data) == 0


def is_eq_coo(array_1: coo_array, array_2: coo_array):
    """Compare if two COORdinate format sparse arrays are identical
    
    This method is reserved for comparing COO arrays that is not of
    a built-in dtype (i.e. of an object type). These sparse arrays
    do not support many of the sparse operations, such as conversion to
    csr
    """
    return (
        array_1.shape == array_2.shape and 
        np.min(array_1.row == array_2.row) and 
        np.min(array_1.col == array_2.col) and
        np.min(array_1.data == array_2.data)
    )


def allclose_sparse(array_1, array_2, rtol=1e-5, atol=1e-8):
    """Compare if two sparse matrices are close enough"""
    c = np.abs(array_1 - array_2) - rtol * np.abs(array_2)
    return c.max() <= atol


def to_dense_obj(obj_array: coo_array, fill_zero: Any):
    """Convert a sparse object array to dense form
    """
    dense_array = np.full(obj_array.shape, fill_zero, dtype=object)
    for i_data, data in enumerate(obj_array.data):
        dense_array[obj_array.row[i_data], obj_array.col[i_data]] = data
    return dense_array


def to_dense_gmpy2(gmpy2_array: coo_array, prec: int, 
    mode: Literal['c', 'f'] = 'c') -> np.ndarray:
    """Convert a sparse gmpy2 array to dense form
    """
    gmpy2_zero = gp.mpfr('0.0', prec) if mode == 'f' else gp.mpc('0.0+0.0j', prec)
    return to_dense_obj(gmpy2_array, gmpy2_zero)


"""
-----------------------------
Eigenvalue processing
-----------------------------
"""

def cluster_modes(eig_vals: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8):
    """Clustering of eigenvalues.
    This function takes in an array of eigenvalues, and decide whether they
    are degenerate or distinct, and then outputs the clustered result.
    
    :param np.ndarray eig_vals: array of eigenvalues;
    :param float rtol: relative tolerance between eigenvalues considered degenerate
    :param float atol: absolute tolerance between eigenvalues considered degenerate
    :returns: index of distinct modes, degenerate ones share the same index
    
    .. note:: The input eigenvalue array should be already "sorted" in some ways,
        so that clustering only occurs for adjacent eigenvalues.
    
    Example:
    
    .. code-block:: python

        >>> a = np.array([1., 2., 2.0000001, 2.99999999, 3.0, 3.00000001, 4.5])
        >>> clusters = cluster_modes(a, rtol=1e-5, atol=1e-5)
        >>> clusters
        np.array([0, 1, 1, 1, 2, 2, 3])
    """
    counter = 0
    modes = []
    eig_ref = eig_vals[0]
    for eig_tmp in eig_vals:
        if np.abs(eig_tmp - eig_ref) > rtol*np.abs(eig_ref) + atol:
            eig_ref = eig_tmp
            counter += 1
        modes.append(counter)
    return np.array(modes)


def intermodal_separation(eig_vals: np.ndarray, **opt_cluster) -> np.ndarray:
    """Calculate intermodal separation ([Boyd]_)
    
    :param np.ndarray eig_vals: array of eigenvalues
    :param \**opt_cluster: keywords for clustering options, 
        see :py:func:`cluster_modes`
    
    .. note:: The input eigenvalue array should be already "sorted" in some ways,
        so that clustering only occurs for adjacent eigenvalues.
        
    .. [Boyd] Boyd, *Chebyshev and Fourier Spectral Methods*.
    """
    mode_idx = cluster_modes(eig_vals, **opt_cluster)
    mode_eigens = np.zeros(mode_idx.max() + 1, np.complex128)
    assert mode_eigens.size >= 1
    for i_eig, eig_tmp in enumerate(eig_vals):
        mode_eigens[mode_idx[i_eig]] = eig_tmp
    mode_dist = np.zeros(mode_eigens.size)
    mode_dist[0] = np.abs(mode_eigens[1] - mode_eigens[0])
    mode_dist[-1] = np.abs(mode_eigens[-2] - mode_eigens[-1])
    mode_dist[1:-1] = (
        + np.abs(mode_eigens[2:] - mode_eigens[1:-1])
        + np.abs(mode_eigens[:-2] - mode_eigens[1:-1])
    )/2
    return mode_dist[mode_idx]

    
def eigen_drift(eig_base: np.ndarray, eig_comp: np.ndarray, waterlevel: float = 0., 
    **opt_cluster):
    """Calculate eigenvalue drift ratio using Boyd's method ([Boyd]_)
    
    :param np.ndarray eig_base: eigenvalue array used as a base
    :param np.ndarray eig_comp: eigenvalue array used for comparison
    :param float waterlevel: waterlevel for near-trivial eigenvalue to avoid
        division by zero; default to zero (assuming nontrivial eigenvalues).
    :param \**opt_cluster: optional keyword arguments for clustering,
        see :py:func:`cluster_modes`
    
    .. [Boyd] Boyd, *Chebyshev and Fourier Spectral Methods*.
    
    .. note:: Be sure to pass in pre-sorted eigenvalues.
    """
    eig_dist = intermodal_separation(eig_base, **opt_cluster)
    eig_diff, _ = np.meshgrid(eig_comp, eig_base, indexing='ij')
    eig_diff = np.abs(eig_diff - eig_base)
    eig_nearest_idx = np.argmin(eig_diff, axis=0)
    eig_diff = np.min(eig_diff, axis=0)
    return eig_diff/(eig_dist + waterlevel), eig_nearest_idx


def spec_tail_exp_rate(spectrum: np.ndarray):
    """Calculate maximum *exponential rate of convergence* 
    from the trailing part of a spectrum.
    """
    lg_coeff = np.log10(np.abs(spectrum) / np.max(np.abs(spectrum), axis=0))
    max_idx = np.argmax(lg_coeff, axis=0)
    max_tail = np.zeros_like(lg_coeff)
    max_tail[-1, ...] = lg_coeff[-1, ...]
    
    for idx in reversed(range(max_tail.shape[0] - 1)):
        max_tail[idx, ...] = max_tail[idx + 1, ...]
        update_idx = lg_coeff[idx, ...] > max_tail[idx, ...]
        max_tail[idx][update_idx] = lg_coeff[idx][update_idx]
        
    n_idx = np.tensordot(
        np.arange(lg_coeff.shape[0]), 
        np.ones_like(lg_coeff[0, ...]), 
        axes=0
    )
    tail_exp_rate = max_tail / (np.abs(max_idx - n_idx) + 1)
    return np.min(tail_exp_rate, axis=0)


