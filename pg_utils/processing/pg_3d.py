# -*- coding: utf-8 -*-
"""
The PG-3D interface

Jingtao Min @ ETH-EPM, 10.2024
"""


from . import base
import numpy as np
from typing import Optional


def moments_3d(
    B_3d: dict, 
    z_coord: np.ndarray,
    moments: Optional[base.LabeledCollection] = None,
    cast_real: bool = False, 
):
    """Convert a 3-D field to PG
    """
    if moments is None:
        moments = base.CollectionPG()
    
    if cast_real:
        B_3d = {k: np.real(v) for k, v in B_3d.items()}
    
    moments.Mss = B_3d['s']*B_3d['s']
    moments.Mpp = B_3d['phi']*B_3d['phi']
    moments.Msp = B_3d['s']*B_3d['phi']
    
    moments.Msz = B_3d['s']*B_3d['z']
    moments.Mpz = B_3d['phi']*B_3d['z']
    
    moments.zMss = z_coord*moments.Mss
    moments.zMpp = z_coord*moments.Mpp
    moments.zMsp = z_coord*moments.Msp
    
    return moments


def moments_3d_linearised(
    B_3d: dict, 
    b_3d: dict,
    z_coord: np.ndarray,
    moments: Optional[base.LabeledCollection] = None,
    cast_real: bool = False, 
):
    """Convert a 3-D field to PG
    """
    if moments is None:
        moments = base.CollectionPG()
    
    if cast_real:
        B_3d = {k: np.real(v) for k, v in B_3d.items()}
    
    moments.Mss = 2*B_3d['s']*b_3d['s']
    moments.Mpp = 2*B_3d['phi']*b_3d['phi']
    moments.Msp = B_3d['s']*b_3d['phi'] + b_3d['s']*B_3d['phi']
    
    moments.Msz = B_3d['s']*b_3d['z'] + b_3d['s']*B_3d['z']
    moments.Mpz = B_3d['phi']*b_3d['z'] + b_3d['phi']*B_3d['z']
    
    moments.zMss = z_coord*moments.Mss
    moments.zMpp = z_coord*moments.Mpp
    moments.zMsp = z_coord*moments.Msp
    
    return moments


def moments_int_pg(
    moments: base.LabeledCollection,
    z_coord: np.ndarray,
    wt: np.ndarray,
    axis: int = -1,
    out_field: Optional[base.LabeledCollection] = None,
):
    """Integrate moments to PG quantities
    """
    if out_field is None:
        out_field = base.CollectionPG()
        
    sign_z = np.sign(z_coord)
    
    out_field.Mss = np.nansum(wt*moments.Mss, axis=axis)
    out_field.Mpp = np.nansum(wt*moments.Mpp, axis=axis)
    out_field.Msp = np.nansum(wt*moments.Msp, axis=axis)
    
    out_field.Msz = np.nansum(sign_z*wt*moments.Msz, axis=axis)
    out_field.Mpz = np.nansum(sign_z*wt*moments.Mpz, axis=axis)
    
    out_field.zMss = np.nansum(sign_z*wt*moments.zMss, axis=axis)
    out_field.zMpp = np.nansum(sign_z*wt*moments.zMpp, axis=axis)
    out_field.zMsp = np.nansum(sign_z*wt*moments.zMsp, axis=axis)
    
    return out_field
