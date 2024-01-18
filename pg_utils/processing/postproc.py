# -*- coding: utf-8 -*-
"""
Utilities for post-processing and visualization of the results

Jingtao Min @ ETH Zurich, 2023
"""

import numpy as np
import os, h5py

from typing import Optional, List, Callable, Tuple
from sympy import S, diff, lambdify

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
            expr, modules=["scipy", "numpy"]
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
            expr, modules=["scipy", "numpy"]
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
            expr, modules=["scipy", "numpy"]
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


def classify_eigens_criteria(eigenvals: np.ndarray, 
    criteria: List[Callable[[np.ndarray], np.ndarray]]):
    """
    """
    return [criterion(eigenvals) for criterion in criteria]
