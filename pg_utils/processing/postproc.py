# -*- coding: utf-8 -*-
"""Utilities for post-processing and visualization of the results
Jingtao Min @ ETH Zurich, 2023
"""

import numpy as np
import os, h5py

from typing import Optional
from sympy import S, diff, lambdify

from ..pg_model import *
from ..pg_model import base, core
from ..pg_model import expansion as xpd

from ..numerics import matrices as nmatrix


def get_eigen_field_function(xpd_recipe: xpd.ExpansionRecipe, 
    phys_par: dict) -> base.LabeledCollection:
    """Get lambdified function of field variables
    for fast evaluation
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
    """Get lambdified function of field variables
    for fast evaluation
    """
    # Build fields
    field_in = base.CollectionConjugate(
        **{fname: xpd_recipe.rad_xpd[fname]*xpd_recipe.fourier_xpd.bases
           for fname in xpd_recipe.rad_xpd.fields._field_names})
    for fname in field_in._field_names:
        if field_in[fname] is None:
            field_in[fname] = S.Zero
    field_in = core.conjugate_to_PG(field_in)
    field_f = base.LabeledCollection(
        base.CollectionPG.pg_field_names + ["U_s", "U_p", "U_z", "Zeta"], 
        **{fname: field_in[fname] for fname in field_in._field_names})
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

