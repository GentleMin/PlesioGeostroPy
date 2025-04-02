# -*- coding: utf-8 -*-
"""
The PG-3D interface

Jingtao Min @ ETH-EPM, 10.2024
"""


from ..pg_model import base
from ..pg_model import expansion as xpd
from ..numerics import symparser, basis
from .. import tools
import numpy as np
from typing import Optional, List, Literal
from dataclasses import dataclass


def moments_3d(
    B_3d: dict, 
    z_coord: np.ndarray,
    moments: Optional[base.LabeledCollection] = None,
    cast_real: bool = False, 
):
    """Convert 3-D magnetic field to quadratic magnetic dyads
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
    """Convert 3-D background and perturbed magnetic fields to quadratic magnetic dyads
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


def recipe_basis_2_polar_jacobi(recipe: xpd.ExpansionRecipe) -> base.LabeledCollection:
    """Parse spectral expansion basis to two-sided polar Jacobi
    """
    basis_dict = {
        key: symparser.basis_2_jacobi_polar(recipe.base_expr[recipe.rad_xpd.bases[key]])
        for key in recipe.rad_xpd.bases._field_names
    }
    return base.LabeledCollection(recipe.rad_xpd.bases._field_names, **basis_dict)


"""
================================================================
Projection from 3D to PG state variables
================================================================
"""


import sys
sys.path.append('../Modes_3D/MCmodes/')

from fields import VectorFieldSingleM
from operators.polynomials import SphericalHarmonicMode
from operators.worland_transform import WorlandTransform
from operators.associated_legendre_transform import AssociatedLegendreTransformSingleM

from scipy import special as specfun
from . import postproc as pproc
from ..numerics import utils as nutils


@dataclass
class ResolutionContext3D_cstM:
    N: int
    L: int
    m_val: int


def spectrify_bg_3d(modes_bg: List[SphericalHarmonicMode], res: ResolutionContext3D_cstM) -> VectorFieldSingleM:
    """Return spectral representation of a background field
    This allows using VectorFieldSingleM utilities for field evaluations
    """
    bg_spec = VectorFieldSingleM.from_SH_mode(res.N, res.L, modes_bg[0])
    for mode in modes_bg[1:]:
        bg_spec = bg_spec + VectorFieldSingleM.from_SH_mode(res.N, res.L, mode)
    return bg_spec


def eigmode_3d_from_eigvec(eig_vec: np.ndarray, res: ResolutionContext3D_cstM, normalize: Literal['u', 'b', 'none'] = 'none'):
    """Parse eigenvector in 3D calculation to vector fields
    """
    Nd = eig_vec.shape[0] // 2
    u = VectorFieldSingleM(res.N, res.L, res.m_val, eig_vec[:Nd])
    b = VectorFieldSingleM(res.N, res.L, res.m_val, eig_vec[Nd:])
    if normalize == 'u':
        norm = np.sqrt(u.energy)
    elif normalize == 'b':
        norm = np.sqrt(b.energy)
    else:
        norm = 1.
    u.normalise(norm)
    b.normalise(norm)
    return u, b


def generate_spec_transforms(recipe: xpd.ExpansionRecipe, res_3d: ResolutionContext3D_cstM, res_target: int):
    """Generate spectral transform based on PG spectral recipe
    """
    m_sub = {xpd.m: res_3d.m_val}
    basis_sym = recipe_basis_2_polar_jacobi(recipe)
    basis_fun = base.LabeledCollection(
        basis_sym._field_names,
        **{
            key: symparser.basis_evaluator(basis_sym[key].subs(m_sub), res_target, qmode='lowest', dealias=1.1, prec=None)
            for key in basis_sym._field_names
        }
    )
    return basis_fun


def pg_phys_moments_lin_zint(
    B_field: VectorFieldSingleM, b_field: VectorFieldSingleM, 
    s_grid: np.ndarray, Nz: int, 
    out_field: Optional[base.LabeledCollection] = None
):
    """Calculate z-integrated linearized magnetic moments from 3D to PG in the physical space.
    Returns a collection of integrated moments on the equatorial plane (or rather on the cylindrical radius)
    """
    z_grid, wt_z = specfun.roots_legendre(Nz)
    H_grid = np.sqrt(1 - s_grid**2)
    
    z_quad = np.outer(z_grid, H_grid)
    wt_quad = np.outer(wt_z, H_grid)
    s_quad = np.ones_like(z_quad)*s_grid
    
    r_pts, t_pts, p_pts = nutils.coord_cart2sph(s_quad.flatten(), np.array(0.), z_quad.flatten())
    
    b_val = b_field.evaluate(r_pts, t_pts, 0.)
    b_val = nutils.vector_sph2cyl(b_val['r'], b_val['theta'], b_val['phi'], r_pts, t_pts, p_pts)[:3]
    b_val = {'s': b_val[0].reshape(z_quad.shape), 'phi': b_val[1].reshape(z_quad.shape), 'z': b_val[2].reshape(z_quad.shape)}
    
    B_val = B_field.evaluate(r_pts, t_pts, 0.)
    B_val = nutils.vector_sph2cyl(B_val['r'], B_val['theta'], B_val['phi'], r_pts, t_pts, p_pts)[:3]
    B_val = {'s': B_val[0].reshape(z_quad.shape), 'phi': B_val[1].reshape(z_quad.shape), 'z': B_val[2].reshape(z_quad.shape)}
    
    if out_field is None:
        out_field = base.CollectionPG()
    
    m_3d = moments_3d_linearised(B_val, b_val, z_quad)
    out_field = moments_int_pg(m_3d, z_quad, wt_quad, axis=0, out_field=out_field)
    return out_field


def pg_phys_b_equatorial(
    b_field: VectorFieldSingleM,
    s_grid: np.ndarray, dz_fdm: float = 0.01,
    out_field: Optional[base.LabeledCollection] = None
):
    """Calculate PG magnetic field on the equatorial plane.
    z-derivative approximated using finite difference
    """
    z_grid = np.array([-dz_fdm, 0., dz_fdm])
    H_grid = np.sqrt(1 - s_grid**2)
    z_quad = np.outer(z_grid, H_grid)
    s_quad = np.ones_like(z_quad)*s_grid
    
    r_pts, t_pts, p_pts = nutils.coord_cart2sph(s_quad.flatten(), np.array(0.), z_quad.flatten())
    b_val = b_field.evaluate(r_pts, t_pts, 0.)
    b_val = nutils.vector_sph2cyl(b_val['r'], b_val['theta'], b_val['phi'], r_pts, t_pts, p_pts)[:3]
    b_val = {'s': b_val[0].reshape(z_quad.shape), 'phi': b_val[1].reshape(z_quad.shape), 'z': b_val[2].reshape(z_quad.shape)}
    
    if out_field is None:
        out_field = base.CollectionPG()
    
    out_field.Bs_e = b_val['s'][1, :]
    out_field.Bp_e = b_val['phi'][1, :]
    out_field.Bz_e = b_val['z'][1, :]
    out_field.dBs_dz_e = (b_val['s'][2, :] - b_val['s'][0, :])/(2*dz_fdm)
    out_field.dBp_dz_e = (b_val['phi'][2, :] - b_val['phi'][0, :])/(2*dz_fdm)
    return out_field


def pg_phys_b_ul(
    b_field: VectorFieldSingleM,
    s_grid: np.ndarray, 
    out_field: Optional[base.LabeledCollection] = None
):
    """Calculate PG magnetic field at the top and lower boundary
    """
    z_grid = np.array([-1., +1.])
    H_grid = np.sqrt(1 - s_grid**2)
    z_quad = np.outer(z_grid, H_grid)
    s_quad = np.ones_like(z_quad)*s_grid
    
    r_pts, t_pts, p_pts = nutils.coord_cart2sph(s_quad.flatten(), np.array(0.), z_quad.flatten())
    b_val = b_field.evaluate(r_pts, t_pts, 0.)
    b_val = nutils.vector_sph2cyl(b_val['r'], b_val['theta'], b_val['phi'], r_pts, t_pts, p_pts)[:3]
    b_val = {'s': b_val[0].reshape(z_quad.shape), 'phi': b_val[1].reshape(z_quad.shape), 'z': b_val[2].reshape(z_quad.shape)}
    
    if out_field is None:
        out_field = base.CollectionPG()
    
    out_field.Bs_p = b_val['s'][1, :]
    out_field.Bp_p = b_val['phi'][1, :]
    out_field.Bz_p = b_val['z'][1, :]
    out_field.Bs_m = b_val['s'][0, :]
    out_field.Bp_m = b_val['phi'][0, :]
    out_field.Bz_m = b_val['z'][0, :]
    return out_field


def project_3d_to_conj(
    u_field: VectorFieldSingleM,
    b_field: VectorFieldSingleM,
    B_field: VectorFieldSingleM,
    res_3d: ResolutionContext3D_cstM,
    tgt_recipe: xpd.ExpansionRecipe,
    tgt_res: dict,
    timer: Optional[tools.ProcTimer] = None,
    verbose: bool = False,
):
    """Project 3-D calculations to PG-conjugate/canonical variables
    """
    if timer is None:
        timer = tools.ProcTimer(start=True)
    
    Nmax = max([N for N in tgt_res.values() if N is not None])
    func_basis = generate_spec_transforms(tgt_recipe, res_3d, Nmax)
    timer.flag(loginfo='Grids for spectral transforms generated.', print_str=verbose, mode='0+')
    
    a_mag, b_mag = func_basis.M_1.a_quad, func_basis.M_1.b_quad
    for fname in func_basis._field_names:
        if ('M' in fname or 'B' in fname) and (func_basis[fname].a_quad != a_mag or func_basis[fname].b_quad != b_mag):
            raise ValueError('Magnetic quantities must be evaluated on a common set of radial grids!')
    
    xi_grid_mag = func_basis.M_1.grid
    s_grid_mag = np.sqrt((1 + xi_grid_mag)/2)
    Nz = np.round(1.2*max([res_3d.N, res_3d.L // 2]))
    eqpg_3d = base.CollectionPG()
    
    eqpg_3d = pg_phys_moments_lin_zint(B_field, b_field, s_grid_mag, Nz, out_field=eqpg_3d)
    timer.flag(loginfo='Evaluation of quadratic moments complete.', print_str=verbose, mode='0+')
    
    eqpg_3d = pg_phys_b_equatorial(b_field, s_grid_mag, out_field=eqpg_3d)
    timer.flag(loginfo='Evaluation of equatorial field complete.', print_str=verbose, mode='0+')
    
    eqpg_3d = pg_phys_b_ul(b_field, s_grid_mag, out_field=eqpg_3d)
    timer.flag(loginfo='Evaluation of boundary fields complete.', print_str=verbose, mode='0+')
    
    eqcg_3d = pproc.arr_pg_2_conj(eqpg_3d)
    evec_cg = np.concatenate([
        func_basis[key].integrate(eqcg_3d[key])[:tgt_res[key]]
        if eqcg_3d[key] is not None else np.zeros(tgt_res[key])
        for key in eqcg_3d._field_names if key != 'Br_b'
    ])
    timer.flag(loginfo='Spectral transform complete.', print_str=verbose, mode='0+')
    
    return evec_cg
    
