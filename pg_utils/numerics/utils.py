# -*- coding: utf-8 -*-


import numpy as np
from typing import Union, List, Optional


def eigenfreq_psi_op(m: Union[int, np.ndarray], n: Union[int, np.ndarray]):
    """Analytic eigenfrequency for the self-adjoint operator
    for stream function Psi in the vorticity equation
    """
    return -m/(n*(2*n + 2*m + 1) + m/2 + m**2/4)


def eigenfreq_inertial3d(m: Union[int, np.ndarray], n: Union[int, np.ndarray]):
    """Analytic eigenfrequency for the 3D inertial modes
    """
    return -2/(m+2)*(np.sqrt(1 + m*(m+2)/n/(2*n+2*m+1)) - 1)


def eigenfreq_Malkus_pg(m: Union[int, np.ndarray], n: Union[int, np.ndarray], 
    Le: float, mode: str="all", timescale: str="spin"):
    """Analytic eigenfrequency for the PG model with Malkus bg field
    """
    omega0 = eigenfreq_psi_op(m, n)
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


def eigenfreq_Malkus_3d(m: Union[int, np.ndarray], n: Union[int, np.ndarray], 
    Le: float, mode: str="all", timescale: str="spin"):
    """Analytic eigenfrequency for 3D eigemodes with Malkus bg field
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

