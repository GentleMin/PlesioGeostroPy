# -*- coding: utf-8 -*-
"""Miscellaneous utilities for numerical calculation
"""


import numpy as np
import gmpy2 as gp
import mpmath as mp
from typing import Union, List, Optional, Callable, Any, Literal, Tuple
from scipy.sparse import coo_array
from dataclasses import dataclass


"""
--------------------------------------
Analytical references
--------------------------------------

* Analytical eigenfrequencies for the PG inertial modes
* Analytical eigenfrequencies for the 3-D inertial modes
* Analytical eigenfrequencies for the Malkus bg field in PG
* Analytical eigenfrequencies for the Malkus bg field in 3D

"""


import sympy as sym
from ..pg_model.core import s, r, theta, p, z

m, N, i, j = sym.symbols('m,N,i,j', integer=True)
sigma = sym.Symbol(r'\sigma')
omega = sym.Symbol(r'\omega')
elamb = sym.Symbol(r'\lambda')


eigenfreq_poly_terms = {
    's': (
        (-1)**j*sym.factorial(2*(2*N + m - j))
        /(sym.factorial(j)*sym.factorial(2*N + m - j)*sym.factorial(2*(N-j)))
        *((2*N + m - 2*j)*sigma - 2*(N - j))*sigma**(2*(N - j)-1)
    ),
    'a': (
        (-1)**j*sym.factorial(2*(2*N + m - j + 1))
        /(sym.factorial(j)*sym.factorial(2*N + m - j + 1)*sym.factorial(2*(N-j) + 1))
        *((2*N + m - 2*j + 1)*sigma - (2*(N - j) + 1))*sigma**(2*(N - j))
    )
}
eigenfreq_polys = {
    's': sym.Sum(eigenfreq_poly_terms['s'], (j, 0, N)),
    'a': sym.Sum(eigenfreq_poly_terms['a'], (j, 0, N)),
}


mode_poly_cfs = {
    's': (
        (-1)**(i + j)*sym.factorial2(2*(N + m + i + j) - 1)
        /(2**(j + 1)*sym.factorial2(2*i - 1))
        /(sym.factorial(N - i - j)*sym.factorial(i)*sym.factorial(j)*sym.factorial(m + j))
    ),
    'a': (
        (-1)**(i + j)*sym.factorial2(2*(N + m + i + j) + 1)
        /(2**(j + 1)*sym.factorial2(2*i + 1))
        /(sym.factorial(N - i - j)*sym.factorial(i)*sym.factorial(j)*sym.factorial(m + j))
    )
}
mode_poly_terms_cyl = {
    's': [
        (m + (2*j + m)*sigma)*sigma**(2*i) * (1 - sigma**2)**(j - 1) * s**(m + 2*j - 1) * z**(2*i),
        (m + 2*j + m*sigma)*sigma**(2*i) * (1 - sigma**2)**(j - 1) * s**(m + 2*j - 1) * z**(2*i),
        2*i*sigma**(2*i - 1) * (1 - sigma**2)**j * s**(m + 2*j) * z**(2*i - 1),
    ]
}


def eigenfreq_inviscid(N_val, m_val, parity='s', sort=True, **solve_kwargs):
    """Calculate eigenfrequencies of the inviscid inertial modes in unit sphere
    """
    poly = eigenfreq_polys[parity]
    poly = poly.subs({N: N_val, m: m_val}).doit()
    roots = sym.nroots(poly, **solve_kwargs)
    eigenfreqs = 2*np.array(roots)
    eigenfreqs = eigenfreqs[np.argsort(np.abs(eigenfreqs))]
    return eigenfreqs


def which_eigenfreq(freq_0, Ns, ms):
    """
    """
    if isinstance(Ns, int):
        Ns = np.arange(1, Ns)
    Ns = np.atleast_1d(np.asarray(Ns))
    ms = np.atleast_1d(np.asarray(ms))
    m, n, k = 0, 0, 0
    parity = 's'
    freq = 0.
    r_diff = 100.
    
    for m_tmp in ms:
        for n_tmp in Ns:
            for par in ['s', 'a']:
                
                freqs = eigenfreq_inviscid(n_tmp, m_tmp, parity=par)
                r_diff_tmp = np.abs(freqs - freq_0)/np.abs(freq_0)
                k_tmp = np.argmin(r_diff_tmp)
                r_diff_tmp = r_diff_tmp[k_tmp]
                
                if r_diff_tmp >= r_diff:
                    continue
                
                r_diff = r_diff_tmp
                m, n, k = m_tmp, n_tmp, k_tmp
                parity = par
                freq = freqs[k_tmp]
    
    return m, n, k, parity, freq


def eigenmode_poly_inviscid(N_val, parity='s'):
    u_s = -sym.I*sym.Add(*[
        (mode_poly_cfs[parity]*mode_poly_terms_cyl[parity][0]).subs({N: N_val, i: i_val, j: j_val}) 
        for i_val in range(N_val + 1) for j_val in range(N_val - i_val + 1)
    ])
    u_p = sym.Add(*[
        (mode_poly_cfs[parity]*mode_poly_terms_cyl[parity][1]).subs({N: N_val, i: i_val, j: j_val}) 
        for i_val in range(N_val + 1) for j_val in range(N_val - i_val + 1)
    ])
    u_z = +sym.I*sym.Add(*[
        (mode_poly_cfs[parity]*mode_poly_terms_cyl[parity][2]).subs({N: N_val, i: i_val, j: j_val}) 
        for i_val in range(N_val + 1) for j_val in range(N_val - i_val + 1)
    ])
    return u_s, u_p, u_z


def eigenfreq_psi_op_pg1(m: Union[int, np.ndarray], n: Union[int, np.ndarray], 
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
            m_gp = to_gmpy2_f(m)
            n_gp = to_gmpy2_f(n)
            return -m_gp/(n_gp*(2*n_gp + 2*m_gp + 1) + m_gp/2 + m_gp**2/4)


def eigenfreq_psi_op(m: Union[int, np.ndarray], n: Union[int, np.ndarray], 
    prec: Optional[int]=None) -> Union[float, np.ndarray]:
    """Analytic eigenfrequency for the self-adjoint operator
    for stream function Psi in the vorticity equation
    
    .. math:: \\omega = - \\frac{m}{n(2n + 2m + 1) + \\frac{m}{2} + \\frac{m^2}{4}}
    
    .. note:: If `prec` is specified other than None, then the input must be
        consistently in gmpy2 form in order to properly compute in multi-precision.
    """
    if prec is None:
        return -m/(n*(2*n + 2*m + 1) + m/2 + m**2/6)
    else:
        with gp.local_context(gp.context(), precision=prec):
            m_gp = to_gmpy2_f(m)
            n_gp = to_gmpy2_f(n)
            return -m_gp/(n_gp*(2*n_gp + 2*m_gp + 1) + m_gp/2 + m_gp**2/6)


def eigenfreq_inertial3d_columnar_approx(m: Union[int, np.ndarray], n: Union[int, np.ndarray]):
    """Analytic eigenfrequency for the 3D inertial modes, approximated for columnar modes
    
    .. math:: 
    
        \\omega = -\\frac{2}{m+2} 
        \\left(\\sqrt{1 + \\frac{m(m+2)}{n(2n+2m+1)}} - 1\\right)
    """
    return -2/(m+2)*(np.sqrt(1 + m*(m+2)/n/(2*n+2*m+1)) - 1)


def eigenfreq_inertial3d_columnar(m: Union[int, np.ndarray], n: Union[int, np.ndarray],
    prec: Optional[int] = None) -> Union[float, np.ndarray]:
    """Analytic eigenfrequency for the 3D columnar inertial modes
    """
    dps = 18 if prec is None else transform_dps_prec(prec=prec)[0] + 3
    
    if isinstance(m, int) and isinstance(n, int):
        freqs = eigenfreq_inviscid(n, m, parity='s', n=dps)[0]
    else:
        n_mesh = n*np.ones_like(m)
        shape = n_mesh.shape
        n_arr = np.ravel(n_mesh)
        m_arr = np.ravel(m*np.ones_like(n))
        freqs = np.array([
            eigenfreq_inviscid(n_arr[i], m_arr[i], parity='s', n=dps)[0]
            for i in range(n_arr.size)
        ]).reshape(shape)
    
    if prec is None:
        return to_numpy_f(freqs)
    else:
        return to_gmpy2_f(freqs, prec=prec)


def eigenfreq_Rossby_to_Malkus(
    m, omega_0, Le: Union[float, gp.mpfr], 
    timescale: str="spin", prec: Optional[int] = None
):
    """Analytic eigenfrequency for the PG model with Malkus bg field
    
    :param Union[int, np.ndarray] m: azimuthal wavenumber
    :param omega_0: inertial mode eigenfrequencies
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
        
        where :math:`\\omega_0` is the inertial mode eigenfrequency. 
        The plus sign gives the fast mode, and the minus sign gives the slow mode.
    """
    if prec is None:
        if timescale.lower() == "spin":
            prefactor = omega_0/2
        elif timescale.lower() == "alfven":
            prefactor = omega_0/2/Le
        else:
            raise AttributeError
        bg_field_mod = np.sqrt(1 + Le**2*(4*m*(m - omega_0))/(omega_0**2))
        return prefactor*(1 + bg_field_mod), prefactor*(1 - bg_field_mod)
    else:
        with gp.local_context(gp.context(), precision=prec):
            if timescale.lower() == "spin":
                prefactor = omega_0/2
            elif timescale.lower() == "alfven":
                prefactor = omega_0/2/Le
            else:
                raise AttributeError
            bg_field_mod = 1 + Le**2*(4*m*(m - omega_0))/(omega_0**2)
            bg_field_mod = np.vectorize(gp.sqrt, otypes=(object,))(bg_field_mod)
            return prefactor*(1 + bg_field_mod), prefactor*(1 - bg_field_mod)


def eigenfreq_Malkus_pg1(m: Union[int, np.ndarray], n: Union[int, np.ndarray], 
    Le: Union[float, gp.mpfr], timescale: str="spin", prec: Optional[int] = None):
    """Analytic eigenfrequency for the old PG model with Malkus bg field
    """
    omega0 = eigenfreq_psi_op_pg1(m, n, prec=prec)
    return eigenfreq_Rossby_to_Malkus(m, omega0, Le, timescale=timescale, prec=prec)


def eigenfreq_Malkus_pg(m: Union[int, np.ndarray], n: Union[int, np.ndarray], 
    Le: Union[float, gp.mpfr, np.ndarray], timescale: str="spin", prec: Optional[int] = None):
    """Analytic eigenfrequency for the PG model with Malkus bg field
    """
    omega0 = eigenfreq_psi_op(m, n, prec=prec)
    return eigenfreq_Rossby_to_Malkus(m, omega0, Le, timescale=timescale, prec=prec)


def eigenfreq_Malkus_3d(m: Union[int, np.ndarray], n: Union[int, np.ndarray], 
    Le: Union[float, gp.mpfr], timescale: str="spin", prec: Optional[int] = None):
    """Analytic eigenfrequency for 3D eigemodes with Malkus bg field
    """
    omega0 = eigenfreq_inertial3d_columnar(m, n, prec=prec)
    return eigenfreq_Rossby_to_Malkus(m, omega0, Le, timescale=timescale, prec=None)


def eigenfreq_Malkus_3d_approx(m: Union[int, np.ndarray], n: Union[int, np.ndarray], 
    Le: float, timescale: str="spin"):
    """Analytic eigenfrequency for 3D eigemodes with Malkus bg field
    """
    omega0 = eigenfreq_inertial3d_columnar_approx(m, n)
    return eigenfreq_Rossby_to_Malkus(m, omega0, Le, timescale=timescale, prec=None)

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


def to_gmpy2_f(x: np.ndarray, dps: Optional[int] = None, 
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


def to_gmpy2_c(x: np.ndarray, dps: Optional[int] = None, 
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


def isclose_gp(a: np.ndarray, b: np.ndarray, rtol=1e-5, atol=1e-8, prec=None):
    """isclose for multi-precision arrays with sanitized input values
    Does not test for infinity and NaNs; only use for sanitized inputs (i.e. without NaNs, infs)
    """
    if prec is None:
        is_close = np.abs(a - b) <= atol + rtol*np.abs(b)
    else:
        with gp.local_context(gp.context(), precision=prec):
            is_close = np.abs(a - b) <= atol + rtol*np.abs(b)
    return is_close


def allclose_gp(a: np.ndarray, b: np.ndarray, rtol=1e-5, atol=1e-8, prec=None):
    """Allclose for multi-precision arrays with sanitized input values
    Does not test for infinity and NaNs; only use for sanitized inputs (i.e. without NaNs, infs)
    """
    return np.all(isclose_gp(a, b, rtol=rtol, atol=atol, prec=prec))


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


def to_mpmath_matrix(dense_array: np.ndarray, prec: int):
    """Convert a numpy array to mpmath matrix
    """
    with mp.workprec(prec):
        mp_mat = mp.matrix(dense_array.tolist())
    return mp_mat


"""
-----------------------------
Coordinate transforms
-----------------------------
"""

def is_shape_broadcastable(shape_1: tuple, shape_2: tuple) -> bool:
    """Check if two shapes are compatible in broadcast
    """
    for dim1, dim2 in zip(shape_1[::-1], shape_2[::-1]):
        if dim1 == 1 or dim2 == 1 or dim1 == dim2:
            continue
        else:
            return False
    return True


def is_broadcastable(array_1: np.ndarray, array_2: np.ndarray) -> bool:
    """Check if two arrays are compatible in broadcast
    """
    return is_shape_broadcastable(array_1.shape, array_2.shape)


def coord_cart2cyl(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinate transform: Cartesian to cylindrical
    """
    assert is_broadcastable(x, y) and is_broadcastable(x, z), \
        "Shapes {}, {}, {} incompatible".format(x.shape, y.shape, z.shape)
    s = np.sqrt(x**2 + y**2)
    p = np.arctan2(y, x)
    return s, p, z


def coord_cyl2cart(
    s: np.ndarray, p: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinate transform: cylindrical to Cartesian
    """
    assert is_broadcastable(s, p) and is_broadcastable(s, z), \
        "Shapes {}, {}, {} incompatible".format(s.shape, p.shape, z.shape)
    x = s*np.cos(p)
    y = s*np.sin(p)
    return x, y, z


def coord_cart2sph(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinate transform: Cartesian to spherical
    """
    assert is_broadcastable(x, y) and is_broadcastable(x, z), \
        "Shapes {}, {}, {} incompatible".format(x.shape, y.shape, z.shape)
    r = np.sqrt(x**2 + y**2 + z**2)
    t = np.arccos(z/r)
    p = np.arctan2(y, x)
    return r, t, p


def coord_sph2cart(
    r: np.ndarray, t: np.ndarray, p: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinate transform: Spherical to Cartesian
    """
    assert is_broadcastable(r, t) and is_broadcastable(r, p), \
        "Shapes {}, {}, {} incompatible".format(r.shape, t.shape, p.shape)
    z = r*np.cos(t)
    s = r*np.sin(t)
    x = s*np.cos(p)
    y = s*np.sin(p)
    return x, y, z


def vector_cart2cyl(
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vector transform: Cartesian to cylindrical
    """
    assert is_broadcastable(vx, vy) and is_broadcastable(vx, vz), \
        "Shapes {}, {}, {} incompatible".format(vx.shape, vy.shape, vz.shape)
    s, p, _ = coord_cart2cyl(x, y, z)
    c_p, s_p = np.cos(p), np.sin(p)
    vs = vx*c_p + vy*s_p
    vp = -vx*s_p + vy*c_p
    return vs, vp, vz, s, p, z


def vector_cyl2cart(
    vs: np.ndarray, vp: np.ndarray, vz: np.ndarray,
    s: np.ndarray, p: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vector transform: Cartesian to cylindrical
    """
    assert is_broadcastable(vs, vp) and is_broadcastable(vs, vz), \
        "Shapes {}, {}, {} incompatible".format(vs.shape, vp.shape, vz.shape)
    x, y, _ = coord_cyl2cart(s, p, z)
    c_p, s_p = np.cos(p), np.sin(p)
    vx = vs*c_p - vp*s_p
    vy = vs*s_p + vp*c_p
    return vx, vy, vz, x, y, z


def vector_cart2sph(
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vector transform: Cartesian to spherical
    """
    assert is_broadcastable(vx, vy) and is_broadcastable(vx, vz), \
        "Shapes {}, {}, {} incompatible".format(vx.shape, vy.shape, vz.shape)
    vs, vp, _, s, p, _ = vector_cart2cyl(vx, vy, vz, x, y, z)
    r = np.sqrt(s**2 + z**2)
    t = np.arccos(z/r)
    c_t, s_t = np.cos(t), np.sin(t)
    vr = vz*c_t + vs*s_t
    vt = -vz*s_t + vs*c_t
    return vr, vt, vp, r, t, p


def vector_sph2cart(
    vr: np.ndarray, vt: np.ndarray, vp: np.ndarray,
    r: np.ndarray, t: np.ndarray, p: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vector transform: spherical to Cartesian
    """
    assert is_broadcastable(vr, vt) and is_broadcastable(vr, vp), \
        "Shapes {}, {}, {} incompatible".format(vr.shape, vt.shape, vp.shape)
    c_t, s_t = np.cos(t), np.sin(t)
    s, z = r*s_t, r*c_t
    vz = vr*c_t - vt*s_t
    vs = vr*s_t + vt*c_t
    vx, vy, _, x, y, z = vector_cyl2cart(vs, vp, vz, s, p, z)
    return vx, vy, vz, x, y, z


def vector_sph2cyl(
    vr: np.ndarray, vt: np.ndarray, vp: np.ndarray,
    r: np.ndarray, t: np.ndarray, p: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vector transform: spherical to cylindrical
    """
    assert is_broadcastable(vr, vt) and is_broadcastable(vr, vp), \
        "Shapes {}, {}, {} incompatible".format(vr.shape, vt.shape, vp.shape)
    c_t, s_t = np.cos(t), np.sin(t)
    s, z = r*s_t, r*c_t
    vs = vr*s_t + vt*c_t
    vz = vr*c_t - vt*s_t
    return vs, vp, vz, s, p, z


"""
----------------------------------
Non-dimensionalisation transforms
----------------------------------
"""

def tscale_convert(
    in_tscale: Literal['rotation', 'Alfven', 'diffusion_mag', 'diffusion_mag_LJ2022'],
    out_tscale: Literal['rotation', 'Alfven', 'diffusion_mag', 'diffusion_mag_LJ2022'],
    **dimless_params
):
    """Conversion between dimensionless parameters with different time scales
    """
    if in_tscale == 'rotation':
        Le = dimless_params.get('Lehnert')
        Em = dimless_params.get('Ekman_mag')
        Ek = dimless_params.get('Ekman', 0.)
        if (Le is None) or (Em is None) or (Ek is None):
            raise TypeError("Dimensionless params need to be specified.")
        t_factor = 1.
    
    if in_tscale == 'Alfven':
        Le = dimless_params.get('Lehnert')
        Lu = dimless_params.get('Lundquist')
        Pm = dimless_params.get('Prandtl_mag', 0.)
        if (Le is None) or (Lu is None) or (Pm is None):
            raise TypeError("Dimensionless params need to be specified.")
        Em = Le/Lu
        Ek = Pm*Em
        t_factor = 1./Le
    
    if in_tscale == 'diffusion_mag':
        Lambda = dimless_params.get('Elsasser')
        Em = dimless_params.get('Ekman_mag')
        Ek = dimless_params.get('Ekman', 0.)
        if (Lambda is None) or (Em is None) or (Ek is None):
            raise TypeError("Dimensionless params need to be specified.")
        Le = np.sqrt(Em*Lambda)
        t_factor = 1./Em
    
    if in_tscale == 'diffusion_mag_LJ2022':
        Lambda = dimless_params.get('Elsasser')
        Em = dimless_params.get('Ekman_mag')
        Ek = dimless_params.get('Ekman', 0.)
        if (Lambda is None) or (Em is None) or (Ek is None):
            raise TypeError("Dimensionless params need to be specified.")
        Le = 2*np.sqrt(Em*Lambda)
        Em = 2*Em
        Ek = 2*Ek
        t_factor = 1./Em
    
    if out_tscale == 'rotation':
        t_factor *= 1.
        return t_factor, {'Lehnert': Le, 'Ekman_mag': Em, 'Ekman': Ek}
    
    if out_tscale == 'Alfven':
        t_factor *= Le
        Lu = Le/Em
        Pm = Ek/Em
        return t_factor, {'Lehnert': Le, 'Lundquist': Lu, 'Prandtl_mag': Pm}
    
    if out_tscale == 'diffusion_mag':
        t_factor *= Em
        Lambda = Em*Le**2
        return t_factor, {'Elsasser': Lambda, 'Ekman_mag': Em, 'Ekman': Ek}
    
    if out_tscale == 'diffusion_mag_LJ2022':
        t_factor *= Em
        Lambda = Em*Le**2/2
        Em /= 2
        Ek /= 2
        return t_factor, {'Elsasser': Lambda, 'Ekman_mag': Em, 'Ekman': Ek}


"""
-----------------------------
Eigenvalue processing
-----------------------------
"""

# def cluster_modes(eig_vals: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8):
#     """Clustering of eigenvalues.
#     This function takes in an array of eigenvalues, and decide whether they
#     are degenerate or distinct, and then outputs the clustered result.
    
#     :param np.ndarray eig_vals: array of eigenvalues;
#     :param float rtol: relative tolerance between eigenvalues considered degenerate
#     :param float atol: absolute tolerance between eigenvalues considered degenerate
#     :returns: index of distinct modes, degenerate ones share the same index
    
#     .. note:: The input eigenvalue array should be already "sorted" in some ways,
#         so that clustering only occurs for adjacent eigenvalues.
    
#     Example:
    
#     .. code-block:: python

#         >>> a = np.array([1., 2., 2.0000001, 2.99999999, 3.0, 3.00000001, 4.5])
#         >>> clusters = cluster_modes(a, rtol=1e-5, atol=1e-5)
#         >>> clusters
#         np.array([0, 1, 1, 1, 2, 2, 3])
#     """
#     counter = 0
#     modes = []
#     eig_ref = eig_vals[0]
#     for eig_tmp in eig_vals:
#         if np.abs(eig_tmp - eig_ref) > rtol*np.abs(eig_ref) + atol:
#             eig_ref = eig_tmp
#             counter += 1
#         modes.append(counter)
#     return np.array(modes)


# def intermodal_separation(eig_vals: np.ndarray, **opt_cluster) -> np.ndarray:
#     """Calculate intermodal separation ([Boyd]_)
    
#     :param np.ndarray eig_vals: array of eigenvalues
#     :param \**opt_cluster: keywords for clustering options, 
#         see :py:func:`cluster_modes`
    
#     .. note:: The input eigenvalue array should be already "sorted" in some ways,
#         so that clustering only occurs for adjacent eigenvalues.
        
#     .. [Boyd] Boyd, *Chebyshev and Fourier Spectral Methods*.
#     """
#     mode_idx = cluster_modes(eig_vals, **opt_cluster)
#     mode_eigens = np.zeros(mode_idx.max() + 1, np.complex128)
#     assert mode_eigens.size >= 1
#     for i_eig, eig_tmp in enumerate(eig_vals):
#         mode_eigens[mode_idx[i_eig]] = eig_tmp
#     mode_dist = np.zeros(mode_eigens.size)
#     mode_dist[0] = np.abs(mode_eigens[1] - mode_eigens[0])
#     mode_dist[-1] = np.abs(mode_eigens[-2] - mode_eigens[-1])
#     mode_dist[1:-1] = (
#         + np.abs(mode_eigens[2:] - mode_eigens[1:-1])
#         + np.abs(mode_eigens[:-2] - mode_eigens[1:-1])
#     )/2
#     return mode_dist[mode_idx]

def cluster_modes(
    eig_vals: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8, 
    mode: Literal["global", "sorted"] = "global") -> np.ndarray:
    """Clustering of eigenvalues.
    
    This function takes in an array of eigenvalues, and decide whether they
    are degenerate or distinct, and then outputs the clustered result.
    
    :param np.ndarray eig_vals: array of eigenvalues;
    :param float rtol: relative tolerance between eigenvalues considered degenerate
    :param float atol: absolute tolerance between eigenvalues considered degenerate
    :param Literal["global", "sorted"] mode: which mode to use;
    
        * "sorted" mode assumes that the array is sorted in a way such that 
          the degenerate eigenvalues / eigenvalues of the same mode are adjacent,
          and has the complexity of O(N)
        * "global" mode drops this assumption, but has complexity O(N^2)
        
    :returns: index of distinct modes, degenerate ones share the same index
    
    """
    if mode == "global":
        return _cluster_modes_global(eig_vals, rtol, atol)
    elif mode == "sorted":
        return _cluster_modes_sorted(eig_vals, rtol, atol)
    else:
        raise ValueError(
            f"Clustering mode can only be global or sorted but got {mode}."
        )


def _cluster_modes_global(eig_vals: np.ndarray, rtol: float, 
    atol: float) -> np.ndarray:
    """Clustering of eigenvalues in arbitrary order
    
    See :py:func:`cluster_modes` for parameter details
    """
    mode_count = 0
    mode_idx = np.full(eig_vals.shape, fill_value=-1, dtype=np.int32)
    for i_eig, eig_ref in enumerate(eig_vals):
        if mode_idx[i_eig] >= 0:
            continue
        i_vicinity = np.abs(eig_vals - eig_ref) < rtol*np.abs(eig_ref) + atol
        mode_idx[i_vicinity] = mode_count
        mode_count += 1
    return mode_idx


def _cluster_modes_sorted(eig_vals: np.ndarray, rtol: float, 
    atol: float) -> np.ndarray:
    """Clustering of sorted eigenvalues.
    
    See :py:func:`cluster_modes` for parameter details
    
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


def intermodal_separation(
    eig_vals: np.ndarray, 
    mode: Literal["global", "sorted"] = "global", 
    **opt_cluster
) -> np.ndarray:
    """Calculate intermodal separation ([Boyd]_)
    
    :param np.ndarray eig_vals: array of eigenvalues
    :param Literal["global", "sorted"] mode: clustering and separation mode
    
        * "sorted" mode assumes that the array is sorted in a way such that 
          the degenerate eigenvalues / eigenvalues of the same mode are adjacent,
          and has the complexity of O(N) in both clustering and calculating sep
        * "global" mode drops this assumption; algorithm has complexity O(N^2)
    
    :param \**opt_cluster: keywords for clustering options, 
        see :py:func:`cluster_modes`
            
    .. [Boyd] Boyd, *Chebyshev and Fourier Spectral Methods*.
    """
    if mode == "global":
        return _intermodal_separation_global(eig_vals, **opt_cluster)
    elif mode == "sorted":
        return _intermodal_separation_sorted(eig_vals, **opt_cluster)
    else:
        raise ValueError(f"Mode can only be global or sorted but got {mode}.")


def _intermodal_separation_global(eig_vals: np.ndarray, **opt_cluster) -> np.ndarray:
    """Calculate intermodal separation for eigenvalues in arb order
    
    See :py:func:`intermodal_separation` for parameter details
    """
    mode_idx = cluster_modes(eig_vals, **opt_cluster, mode="global")
    mode_eigens = np.array([eig_vals[np.argmax(mode_idx == mode_tmp)] 
        for mode_tmp in range(mode_idx.max() + 1)])
    mode_dist = np.meshgrid(mode_eigens, mode_eigens)
    mode_dist = np.abs(mode_dist[0] - mode_dist[1])
    np.fill_diagonal(mode_dist, np.max(mode_dist))
    mode_dist = np.min(mode_dist, axis=1)
    return mode_dist[mode_idx]


def _intermodal_separation_sorted(eig_vals: np.ndarray, **opt_cluster) -> np.ndarray:
    """Calculate intermodal separation for sorted eigenvalues
    
    See :py:func:`intermodal_separation` for parameter details
    
    .. note:: The input eigenvalue array should be already "sorted" in some ways,
        so that clustering only occurs for adjacent eigenvalues.
    """
    mode_idx = cluster_modes(eig_vals, **opt_cluster, mode="sorted")
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
    mode: Literal["global", "sorted"] = "global", **opt_cluster):
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
    eig_dist = intermodal_separation(eig_base, mode=mode, **opt_cluster)
    eig_diff, _ = np.meshgrid(eig_comp, eig_base, indexing='ij')
    eig_diff = np.abs(eig_diff - eig_base)
    eig_nearest_idx = np.argmin(eig_diff, axis=0)
    eig_diff = np.min(eig_diff, axis=0)
    return eig_diff/(eig_dist + waterlevel), eig_nearest_idx


def eigenvalue_tracing(*eigenvalues: np.ndarray, init_threshold: float = 1e+4, 
    init_filter_mode: Literal["global", "sorted"] = "global", 
    **opt_init_cluster):
    """
    """
    assert len(eigenvalues) >= 2
    eigen_drift, nearest_idx = eigen_drift(
        eigenvalues[0], eigenvalues[1], mode=init_filter_mode, **opt_init_cluster)
    idx_bool_tmp = 1./(eigen_drift + 1e-16) > init_threshold
    traced_indices = np.full((np.sum(idx_bool_tmp), len(eigenvalues)))
    traced_indices[:, 0] = np.arange(len(eigenvalues[0]))[idx_bool_tmp]
    traced_indices[:, 1] = nearest_idx


def eigenvec_similarity(evec_base: np.ndarray, evec_comp: np.ndarray):
    """Compute max similarity between eigenvectors
    """
    dim = min([evec_base.shape[0], evec_comp.shape[0]])
    # evec_base_pad = np.pad(evec_base, ((0,dim-evec_base.shape[0]), (0,0)), mode='constant', constant_values=0.)
    # evec_comp_pad = np.pad(evec_comp, ((0,dim-evec_comp.shape[0]), (0,0)), mode='constant', constant_values=0.)
    kernel = evec_base[:dim].conj().T @ evec_comp[:dim]
    norm_base = np.linalg.norm(evec_base, axis=0)**2
    norm_comp = np.linalg.norm(evec_comp, axis=0)**2
    similarity = np.abs(kernel)**2/np.outer(norm_base, norm_comp)
    most_similar_idx = np.argmax(similarity, axis=1)
    max_similarity = np.max(similarity, axis=1)
    return max_similarity, most_similar_idx


def val_tracing_nearest(seeds: np.ndarray, *values: np.ndarray, 
    rtol: float = 1e-1, atol: float = 1e-7, fill_value = np.nan):
    """Trace the values that are closest to the input seeds
    """
    assert len(values) > 0
    traced_values = np.full((len(seeds), len(values)), fill_value=fill_value, dtype=values[0].dtype)
    for i_set, val_array in enumerate(values):
        nearest_idx = np.argmin(np.abs(np.subtract.outer(seeds, val_array)), axis=1)
        nearest_val = val_array[nearest_idx]
        i_accept = (np.abs(nearest_val - seeds) < rtol + atol*np.abs(seeds))
        traced_values[i_accept, i_set] = nearest_val[i_accept]
    return traced_values, nearest_idx


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


@dataclass
class EigenvalueSpectrum:
    """Class for keeping eigenvalue spectrum
    """
    data: np.ndarray
    N: int
    name: Optional[str] = None


class EigenvalueSpectraSet:
    """A collection of eigenvalue spectra
    """
    def __init__(self, *spectra: EigenvalueSpectrum) -> None:
        self.set = list(spectra)



"""
-----------------------------
Spectrum processing
-----------------------------
"""

def normalize(array: np.ndarray, 
    mode: Literal["max", "norm"] = "max", zero_phase: bool = False):
    """Normalize a spectrum or general vector
    """
    if mode == "norm":
        normalizer = np.linalg.norm(array.flatten())
    elif mode == "max":
        max_idx = np.argmax(np.abs(array))
        normalizer = array.flatten()[max_idx] if zero_phase else np.abs(array.flatten()[max_idx])
    return array/normalizer

