# -*- coding: utf-8 -*-
"""
Utilities for post-processing and visualization of the results

Jingtao Min @ ETH Zurich, 2023
"""

import numpy as np
import os, h5py

from typing import Optional, List, Callable, Tuple
from sympy import S, diff, lambdify
from scipy import special as specfun

from ..pg_model import *
from ..pg_model import base, core, expansion
from ..pg_model import expansion as xpd

from ..numerics import matrices as nmatrix


def get_eigen_field_function(xpd_recipe: xpd.ExpansionRecipe, 
    phys_par: dict) -> base.LabeledCollection:
    """Lambdify functions of field variables
    
    This method converts symbolic expansions of the unknown variables
    to numerical functions of the variables for efficient evaluation
    
    :param expansion.ExpansionRecipe xpd_recipe: expansion recipe
    :param dict phys_par: physical parameter to be substituted into expressions
    
    :returns: collection of functions for numerical evaluation, 
        the first N functions correspond to the fields indicated in `xpd_recipe`,
        and their names also inherits from the recipe;
        the trailing 4 functions are `U_s` (radial velocity), `U_p` (azimuthal 
        velocity), `U_z` (axial velocity) and `Zeta` (axial vorticity).
    :rtype: base.LabeledCollection
    
    .. note::

        This method assumes implicitly that stream function ``Psi`` is part
        of the variables of the `xpd_recipe` (which should be the case, as long
        as we are solving the PG model). This is required to calculate the
        velocity field and the axial vorticity.
    
    """
    # Build fields
    field_f = base.LabeledCollection(
        xpd_recipe.rad_xpd.fields._field_names + ["U_s", "U_p", "U_z", "Zeta"],
        **{fname: xpd_recipe.rad_xpd[fname]*xpd_recipe.fourier_xpd.bases
           for fname in xpd_recipe.rad_xpd.fields._field_names})
    field_f.U_s = core.U_pg[0].subs({core.pgvar.Psi: field_f.Psi})
    field_f.U_p = core.U_pg[1].subs({core.pgvar.Psi: field_f.Psi})
    field_f.U_z = core.U_pg[2].subs({core.pgvar.Psi: field_f.Psi})
    field_f.Zeta = (diff(core.s*field_f.U_p, core.s) - diff(field_f.U_s, core.p))/core.s
    # Substitution and symbolic evaluation
    field_f.apply(
        lambda expr: expr.subs(xpd_recipe.base_expr).subs(phys_par).subs({H: H_s}).doit().simplify(), 
        inplace=True, metadata=False
    )
    # Lambdify
    field_f.apply(
        lambda expr: lambdify(
            (s, p, z, xpd.n_trial, *[cf for cf in xpd_recipe.rad_xpd.coeffs]), 
            expr, modules=[{'jacobi_u': specfun.eval_jacobi}, "scipy", "numpy"]
        ), 
        inplace=True, metadata=False
    )
    return field_f


def eigen_func_from_conjugate(xpd_recipe: xpd.ExpansionRecipe, 
    phys_par: dict) -> base.LabeledCollection:
    """Lambdify functions of field variables from conjugate expansions
    
    This method converts symbolic expansions of the conjugate variables
    to numerical functions of PG variables for efficient evaluation
    
    :param expansion.ExpansionRecipe xpd_recipe: expansion recipe (for conjugate variables)
    :param dict phys_par: physical parameter to be substituted into expressions
    
    :returns: collection of functions for numerical evaluation, 
        the leading functions are PG variable functions;
        the trailing 4 functions are `U_s` (radial velocity), `U_p` (azimuthal 
        velocity), `U_z` (axial velocity) and `Zeta` (axial vorticity).
    :rtype: base.LabeledCollection
    
    .. note::

        This method assumes implicitly that stream function ``Psi`` is part
        of the variables of the `xpd_recipe` (which should be the case, as long
        as we are solving the PG model). This is required to calculate the
        velocity field and the axial vorticity.
    """
    # Build conjugate fields
    field_in = base.CollectionConjugate(
        **{fname: xpd_recipe.rad_xpd[fname]*xpd_recipe.fourier_xpd.bases
           for fname in xpd_recipe.rad_xpd.fields._field_names})
    for fname in field_in._field_names:
        if field_in[fname] is None:
            field_in[fname] = S.Zero
    # Convert to PG fields
    field_in = core.conjugate_to_PG(field_in)
    field_f = base.LabeledCollection(
        base.CollectionPG.pg_field_names + ["U_s", "U_p", "U_z", "Zeta"], 
        **{fname: field_in[fname] for fname in field_in._field_names})
    field_f.U_s = core.U_pg[0].subs({core.pgvar.Psi: field_f.Psi})
    field_f.U_p = core.U_pg[1].subs({core.pgvar.Psi: field_f.Psi})
    field_f.U_z = core.U_pg[2].subs({core.pgvar.Psi: field_f.Psi})
    # field_f.Zeta = (diff(core.s*field_f.U_p, core.s) - diff(field_f.U_s, core.p))/core.s
    field_f.Zeta = -(diff(s/H*diff(field_f.Psi, s), s) + diff(field_f.Psi, (p, 2))/s/H)/s
    # Substitution and symbolic evaluation
    field_f.apply(
        lambda expr: expr.subs(xpd_recipe.base_expr).subs(phys_par).subs({H: H_s}).doit().simplify(), 
        inplace=True, metadata=False
    )
    # Lambdify
    field_f.apply(
        lambda expr: lambdify(
            (s, p, z, xpd.n_trial, *[cf for cf in xpd_recipe.rad_xpd.coeffs]), 
            expr, modules=[{'jacobi_u': specfun.eval_jacobi}, "scipy", "numpy"]
        ), 
        inplace=True, metadata=False
    )
    return field_f


def eigen_func_from_reduced(xpd_recipe: xpd.ExpansionRecipe, 
    phys_par: dict) -> base.LabeledCollection:
    """Lambdify functions of field variables from reduced form
    
    This method converts symbolic expansions of the reduced variables
    to numerical functions of PG variables for efficient evaluation
    
    :param expansion.ExpansionRecipe xpd_recipe: expansion recipe (for conjugate variables)
    :param dict phys_par: physical parameter to be substituted into expressions
    
    :returns: collection of functions for numerical evaluation, 
        the leading functions are PG variable functions;
        the trailing 4 functions are `U_s` (radial velocity), `U_p` (azimuthal 
        velocity), `U_z` (axial velocity) and `Zeta` (axial vorticity).
    :rtype: base.LabeledCollection
    
    .. note::

        Reduced formulation does not give magnetic components of the eigenmode
        in itself! The eigensolution and the expansion does not determine these.
        The magnetic components can only be recovered by plugging the solution
        back into the full PG equation.
    """
    # Build conjugate fields
    field_in = base.LabeledCollection(
        xpd_recipe.rad_xpd.fields._field_names,
        **{fname: xpd_recipe.rad_xpd[fname]*xpd_recipe.fourier_xpd.bases
           for fname in xpd_recipe.rad_xpd.fields._field_names})
    field_f = base.LabeledCollection(
        field_in._field_names + ["U_s", "U_p", "U_z", "Zeta"], 
        **{fname: field_in[fname] for fname in field_in._field_names})
    field_f.U_s = core.U_pg[0].subs({core.pgvar.Psi: field_f.Psi})
    field_f.U_p = core.U_pg[1].subs({core.pgvar.Psi: field_f.Psi})
    field_f.U_z = core.U_pg[2].subs({core.pgvar.Psi: field_f.Psi})
    # field_f.Zeta = (diff(core.s*field_f.U_p, core.s) - diff(field_f.U_s, core.p))/core.s
    field_f.Zeta = -(diff(s/H*diff(field_f.Psi, s), s) + diff(field_f.Psi, (p, 2))/s/H)/s
    # Substitution and symbolic evaluation
    field_f.apply(
        lambda expr: expr.subs(xpd_recipe.base_expr).subs(phys_par).subs({H: H_s}).doit().simplify(), 
        inplace=True, metadata=False
    )
    # Lambdify
    field_f.apply(
        lambda expr: lambdify(
            (s, p, z, xpd.n_trial, *[cf for cf in xpd_recipe.rad_xpd.coeffs]), 
            expr, modules=[{'jacobi_u': specfun.eval_jacobi}, "scipy", "numpy"]
        ), 
        inplace=True, metadata=False
    )
    return field_f


def filter_sort(vals: np.ndarray, 
    filter_op: Callable[[np.ndarray], np.ndarray] = lambda x, y: np.full(x.shape, True), 
    threshold: Optional[float] = None,
    transform_filter: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    transform_sort: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    reversed: bool = False, 
    remove_zero: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    """
    idx_out = np.arange(vals.size)
    idx_out = idx_out[filter_op(transform_filter(vals), threshold)]
    vals_out = vals[idx_out]
    idx_sort = np.argsort(transform_sort(vals_out))
    if reversed:
        idx_sort = np.flip(idx_sort)
    vals_out = vals_out[idx_sort]
    idx_out = idx_out[idx_sort]
    if remove_zero is not None:
        nonzero = np.abs(vals_out) > remove_zero
        vals_out = vals_out[nonzero]
        idx_out = idx_out[nonzero]
    return vals_out, idx_out


def arr_cyl_2_canonical(*args: np.ndarray, rank: int = 1, sym: bool = True):
    """Convert tensors in cylindrical components to canonical components
    """
    if rank == 1:
        cyl_s, cyl_p = args
        can_p = (cyl_s + 1j*cyl_p)/np.sqrt(2)
        can_m = (cyl_s - 1j*cyl_p)/np.sqrt(2)
        return can_p, can_m
    
    if rank == 2:
        if sym:
            cyl_ss, cyl_pp, cyl_sp = args
            can_pp = (cyl_ss - cyl_pp + 2j*cyl_sp)/2
            can_mm = (cyl_ss - cyl_pp - 2j*cyl_sp)/2
            can_1 = (cyl_ss + cyl_pp)/2
            return can_pp, can_mm, can_1
        else:
            cyl_ss, cyl_pp, cyl_sp, cyl_ps = args
            can_pp = (cyl_ss - cyl_pp + 1j*(cyl_sp + cyl_ps))/2
            can_mm = (cyl_ss - cyl_pp - 1j*(cyl_sp + cyl_ps))/2
            can_pm = (cyl_ss + cyl_pp - 1j*(cyl_sp - cyl_ps))/2
            can_mp = (cyl_ss + cyl_pp + 1j*(cyl_sp - cyl_ps))/2
            return can_pp, can_mm, can_pm, can_mp
    
    raise NotImplementedError


def arr_canonical_2_cyl(*args: np.ndarray, rank: int = 1, sym: bool = True):
    """Convert tensors in canonical components to cylindrical components
    """
    if rank == 1:
        can_p, can_m = args
        cyl_s = (can_p + can_m)/np.sqrt(2)
        cyl_p = (can_p - can_m)/np.sqrt(2)/1j
        return cyl_s, cyl_p
    
    if rank == 2:
        if sym:
            can_pp, can_mm, can_1 = args
            cyl_ss = (+(can_pp + can_mm) + 2*can_1)/2
            cyl_pp = (-(can_pp + can_mm) + 2*can_1)/2
            cyl_sp = (can_pp - can_mm)/2j
            return cyl_ss, cyl_pp, cyl_sp
        else:
            can_pp, can_mm, can_pm, can_mp = args
            cyl_ss = (+(can_pp + can_mm) + (can_pm + can_mp))/2
            cyl_pp = (-(can_pp + can_mm) + (can_pm + can_mp))/2
            cyl_sp = (can_pp - can_mm - (can_pm - can_mp))/2j
            cyl_ps = (can_pp - can_mm + (can_pm - can_mp))/2j
            return cyl_ss, cyl_pp, cyl_sp, cyl_ps
    
    raise NotImplementedError


def arr_pg_2_conj(pg_comp: base.CollectionPG) -> base.CollectionConjugate:
    """Convert arrays of PG fields to conjugate counterparts.
    
    :param base.CollectionPG pg_comp: PG components to be converted
    :returns: collection of conjugate quantities
    """
    # Decide how to form the conjugate object
    # The method assumes all entries are of the same type
    cg_comp = base.CollectionConjugate()
    
    # No conversion
    cg_comp.Psi = pg_comp.Psi
    cg_comp.Br_b = pg_comp.Br_b
    # cg_comp.Bz_e = pg_comp.Bz_e
    cg_comp.Bz_p = pg_comp.Bz_p
    cg_comp.Bz_m = pg_comp.Bz_m
    cg_comp.Mzz = pg_comp.Mzz
    
    # Moments
    if (pg_comp.Mss is not None) and (pg_comp.Mpp is not None) and (pg_comp.Msp is not None):
        M_p, M_m, M_1 = arr_cyl_2_canonical(pg_comp.Mss, pg_comp.Mpp, pg_comp.Msp, rank=2, sym=True)
        cg_comp.M_1 = M_1
        cg_comp.M_p = M_p
        cg_comp.M_m = M_m

    if (pg_comp.zMsz is not None) and (pg_comp.zMpz is not None):
        zM_zp, zM_zm = arr_cyl_2_canonical(pg_comp.zMsz, pg_comp.zMpz, rank=1)
        cg_comp.zM_zp = zM_zp
        cg_comp.zM_zm = zM_zm
    
    if (pg_comp.z2Mss is not None) and (pg_comp.z2Mpp is not None) and (pg_comp.z2Msp is not None):
        z2M_p, z2M_m, z2M_1 = arr_cyl_2_canonical(pg_comp.z2Mss, pg_comp.z2Mpp, pg_comp.z2Msp, rank=2, sym=True)
        cg_comp.z2M_1 = z2M_1
        cg_comp.z2M_p = z2M_p
        cg_comp.z2M_m = z2M_m
    
    # if (pg_comp.Msz is not None) and (pg_comp.Mpz is not None):
    #     M_zp, M_zm = arr_cyl_2_canonical(pg_comp.Msz, pg_comp.Mpz, rank=1)
    #     cg_comp.M_zp = M_zp
    #     cg_comp.M_zm = M_zm
    
    # if (pg_comp.zMss is not None) and (pg_comp.zMpp is not None) and (pg_comp.zMsp is not None):
    #     zM_p, zM_m, zM_1 = arr_cyl_2_canonical(pg_comp.zMss, pg_comp.zMpp, pg_comp.zMsp, rank=2, sym=True)
    #     cg_comp.zM_1 = zM_1
    #     cg_comp.zM_p = zM_p
    #     cg_comp.zM_m = zM_m
    
    # # Equatorial fields
    # if (pg_comp.Bs_e is not None) and (pg_comp.Bp_e is not None):
    #     B_ep, B_em = arr_cyl_2_canonical(pg_comp.Bs_e, pg_comp.Bp_e, rank=1)
    #     cg_comp.B_ep = B_ep
    #     cg_comp.B_em = B_em
    
    # if (pg_comp.dBs_dz_e is not None) and (pg_comp.dBp_dz_e is not None):
    #     dB_dz_ep, dB_dz_em = arr_cyl_2_canonical(pg_comp.dBs_dz_e, pg_comp.dBp_dz_e, rank=1)
    #     cg_comp.dB_dz_ep = dB_dz_ep
    #     cg_comp.dB_dz_em = dB_dz_em
    
    # Boundary terms
    if (pg_comp.Bs_p is not None) and (pg_comp.Bp_p is not None):
        B_pp, B_pm = arr_cyl_2_canonical(pg_comp.Bs_p, pg_comp.Bp_p, rank=1)
        cg_comp.B_pp = B_pp
        cg_comp.B_pm = B_pm
    
    if (pg_comp.Bs_m is not None) and (pg_comp.Bp_m is not None):
        B_mp, B_mm = arr_cyl_2_canonical(pg_comp.Bs_m, pg_comp.Bp_m, rank=1)
        cg_comp.B_mp = B_mp
        cg_comp.B_mm = B_mm
    
    return cg_comp


def arr_conj_2_pg(cg_comp: base.CollectionConjugate) -> base.CollectionPG:
    """Convert conjugate quantities to PG counterparts
    
    :param base.CollectionConjugate cg_comp: conjugate components to be converted
    :returns: collection object with PG quantities
    """
    pg_comp = base.CollectionPG()
    
    # No conversion
    pg_comp.Psi = cg_comp.Psi
    pg_comp.Br_b = cg_comp.Br_b
    # pg_comp.Bz_e = cg_comp.Bz_e
    pg_comp.Bz_p = cg_comp.Bz_p
    pg_comp.Bz_m = cg_comp.Bz_m
    pg_comp.Mzz = cg_comp.Mzz
    
    # Moments
    if (cg_comp.M_1 is not None) and (cg_comp.M_p is not None) and (cg_comp.M_m is not None):
        Mss, Mpp, Msp = arr_canonical_2_cyl(cg_comp.M_p, cg_comp.M_m, cg_comp.M_1, rank=2, sym=True)
        pg_comp.Mss = Mss
        pg_comp.Mpp = Mpp
        pg_comp.Msp = Msp

    if (cg_comp.zM_zp is not None) and (cg_comp.zM_zm is not None):
        zMsz, zMpz = arr_canonical_2_cyl(cg_comp.zM_zp, cg_comp.zM_zm, rank=1)
        pg_comp.zMsz = zMsz
        pg_comp.zMpz = zMpz
    
    if (cg_comp.z2M_p is not None) and (cg_comp.z2M_m is not None) and (cg_comp.z2M_1 is not None):
        z2Mss, z2Mpp, z2Msp = arr_canonical_2_cyl(cg_comp.z2M_p, cg_comp.z2M_m, cg_comp.z2M_1, rank=2, sym=True)
        pg_comp.z2Mss = z2Mss
        pg_comp.z2Mpp = z2Mpp
        pg_comp.z2Msp = z2Msp
        
    # if (cg_comp.M_zp is not None) and (cg_comp.M_zm is not None):
    #     Msz, Mpz = arr_canonical_2_cyl(cg_comp.M_zp, cg_comp.M_zm, rank=1)
    #     pg_comp.Msz = Msz
    #     pg_comp.Mpz = Mpz
    
    # if (cg_comp.zM_p is not None) and (cg_comp.zM_m is not None) and (cg_comp.zM_1 is not None):
    #     zMss, zMpp, zMsp = arr_canonical_2_cyl(cg_comp.zM_p, cg_comp.zM_m, cg_comp.zM_1, rank=2, sym=True)
    #     pg_comp.zMss = zMss
    #     pg_comp.zMpp = zMpp
    #     pg_comp.zMsp = zMsp
    
    # # Equatorial fields
    # if (cg_comp.B_ep is not None) and (cg_comp.B_em is not None):
    #     Bs_e, Bp_e = arr_canonical_2_cyl(cg_comp.B_ep, cg_comp.B_em, rank=1)
    #     pg_comp.Bs_e = Bs_e
    #     pg_comp.Bp_e = Bp_e
    
    # if (cg_comp.dB_dz_ep is not None) and (cg_comp.dB_dz_em is not None):
    #     dBs_dz_e, dBp_dz_e = arr_canonical_2_cyl(cg_comp.dB_dz_ep, cg_comp.dB_dz_em, rank=1)
    #     pg_comp.dBs_dz_e = dBs_dz_e
    #     pg_comp.dBp_dz_e = dBp_dz_e
    
    # Boundary terms
    if (cg_comp.B_pp is not None) and (cg_comp.B_pm is not None):
        Bs_p, Bp_p = arr_canonical_2_cyl(cg_comp.B_pp, cg_comp.B_pm, rank=1)
        pg_comp.Bs_p = Bs_p
        pg_comp.Bp_p = Bp_p
    
    if (cg_comp.B_mp is not None) and (cg_comp.B_mm is not None):
        Bs_m, Bp_m = arr_canonical_2_cyl(cg_comp.B_mp, cg_comp.B_mm, rank=1)
        pg_comp.Bs_m = Bs_m
        pg_comp.Bp_m = Bp_m
    
    return pg_comp
