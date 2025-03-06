# -*- coding: utf-8 -*-
"""
Utilities for PG model
Jingtao Min @ ETH-EPM, 09.2023
"""

import sympy
from ..sympy_supp import vector_calculus_3d as v3d
from .core import *
from . import base
from typing import Optional


def integrate_sym(field):
    """Equatorially symmetric / even integral
    
    :param sympy.Expr field: function to be integrated.
        This should be a function of cylindrical coordinates,
        especially a function of :data:`pg_utils.pg_model.core.z`
    :returns: integrated quantity
    :rtype: sympy.Expr
    """
    return sympy.integrate(field, (z, -H, H))


def integrate_asym(field):
    """equatorially antisymmetric / odd integral
    
    :param sympy.Expr field: function to be integrated.
        This should be a function of cylindrical coordinates,
        especially a function of :data:`pg_utils.pg_model.core.z`
    :returns: integrated quantity
    :rtype: sympy.Expr
    """
    return sympy.integrate(field, (z, 0, H)) + sympy.integrate(field, (z, 0, -H))


def field_to_moment(B_field):
    """Convert magnetic field to integrated quadratic moments.
    
    :param B_field: tuple, magnetic field components
        B_field should be given in cylindrical components, 
        and the field components should be functions of cylindrical coordinates,
        i.e. B_field = (B_s(s, p, z), B_p(s, p, z), B_z(s, p, z))
    :returns: tuple of the sympy expressions of the eight moments,
        Mss, Mpp, Msp, Msz, Mpz, zMss, zMpp, zMsp
    """
    assert isinstance(B_field, v3d.Vector3D) or len(B_field) == 3
    Mss = integrate_sym(B_field[0]*B_field[0])
    Mpp = integrate_sym(B_field[1]*B_field[1])
    Msp = integrate_sym(B_field[0]*B_field[1])
    Msz = integrate_asym(B_field[0]*B_field[2])
    Mpz = integrate_asym(B_field[1]*B_field[2])
    zMss = integrate_asym(z*B_field[0]*B_field[0])
    zMpp = integrate_asym(z*B_field[1]*B_field[1])
    zMsp = integrate_asym(z*B_field[0]*B_field[1])
    return Mss, Mpp, Msp, Msz, Mpz, zMss, zMpp, zMsp


def assemble_background(B0, Psi0=None, mode="PG", sub_H=False):
    """Assemble background fields
    
    :param array-like B0: indexable
        background magnetic field in cylindrical coordinates
    :param sympy.Function Psi0: the background stream function
        Because velocity cannot in general be converted to Psi,
        if one wants to specify the stream function this has to
        be done separately. Note that when the nonlinear term is 
        absent from the vorticity equation, Psi0 will in general
        not be involved in any of these equations. Background
        velocity in all induction equations uses U instead of Psi
    :param str mode: the mode, what kind of background fields to
        assemble. Supports Plesio-Geostrophy "PG", or conjugate 
        variables "CG". Default is PG.
    
    :returns: collection of background PG fields / conjugate fields
    :rtype: base.CollectionPG or base.CollectionConjugate
    
    .. note:: The components of B0 and the field Psi0 need to be
        "directly evaluable", i.e. they should all be `sympy.Expr`
        containing no derivative of undefined functions.
        Otherwise, substitution will return incorrect results.
        The background field at the boundary are not immediately 
        evaluated at :math:`z=\\pm H`, since their z-derivatives are 
        present in the induction equations. One has to evaluate 
        after substitution and simplification.
    """
    moments_bg = field_to_moment(B0)
    pg_background = base.CollectionPG(
        # Vorticity
        Psi = Psi0 if Psi0 is not None else sympy.S.Zero,
        # Magnetic moments
        Mss = moments_bg[0],
        Mpp = moments_bg[1],
        Msp = moments_bg[2],
        Msz = moments_bg[3],
        Mpz = moments_bg[4],
        zMss = moments_bg[5],
        zMpp = moments_bg[6],
        zMsp = moments_bg[7],
        # Magnetic field in the equatorial plane
        Bs_e = B0[0].subs({z: 0}),
        Bp_e = B0[1].subs({z: 0}),
        Bz_e = B0[2].subs({z: 0}),
        dBs_dz_e = diff(B0[0], z).doit().subs({z: 0}),
        dBp_dz_e = diff(B0[1], z).doit().subs({z: 0}),
        # Magnetic field at the boundary
        Br_b = s*B0[0] + z*B0[2],
        Bs_p = B0[0].subs({z: +H}),
        Bp_p = B0[1].subs({z: +H}),
        Bz_p = B0[2].subs({z: +H}),
        Bs_m = B0[0].subs({z: -H}),
        Bp_m = B0[1].subs({z: -H}),
        Bz_m = B0[2].subs({z: -H})
    )
    if sub_H:
        pg_background.apply(
            lambda x: x.subs({H: H_s}) if isinstance(x, sympy.Expr) else x,
            inplace=True
        )
    if mode.lower() == "pg":
        return pg_background
    elif mode.lower() == "cg":
        return PG_to_conjugate(pg_background)


def linearize(expr, *subs_maps, perturb_var=eps):
    """Linearize expression
    
    :param sympy.Expr expr: expressions to be linearized
    :param dict *subs_maps: subtitution maps, takes the form
        ``{A: A0 + eps*A1, B: B0 + eps*B1, ...}``
    :param sympy.Symbol perturb_var: perturbation number, 
        default to be the symbol eps from pg_fields module
    :returns: linearized expression
    """
    expr_lin = expr
    for subs_map in subs_maps:
        expr_lin = expr_lin.subs(subs_map)
    expr_lin = expr_lin.doit().expand().coeff(perturb_var, 1)
    return expr_lin


def fields_in_term(expr: sympy.Expr, field_collection: base.LabeledCollection):
    """Extract all fields in a term from a collection.
    
    :param sympy.Expr expr: the term from which the field will be extracted
    :param base.LabeledCollection field_collection: range of fields
    :returns: set of fields
    """
    set_tmp = {field for field in field_collection}
    expr_fields = expr.atoms(sympy.Function)
    expr_fields = tuple(field for field in expr_fields if field in set_tmp)
    return expr_fields


def extract_symbols(var_collection: base.LabeledCollection):
    """Return a collection of symbols, whose names are specified
    by the input collection items
    """
    symb_collection = base.LabeledCollection(
        var_collection._field_names, 
        **{fname: sympy.Symbol(var_collection[fname].name) 
           for fname in var_collection._field_names}
    )
    return symb_collection


def forcing_term(psi_eqn: sympy.Eq, dyn_vars: base.LabeledCollection):
    """Retrieve body force term expression in terms of dynamical variables
    """
    F_ext = sympy.S.Zero
    for term in psi_eqn.rhs.expand().args:
        for func in term.atoms(sympy.Function):
            if func in dyn_vars:
                F_ext += term
                break
    return F_ext


def slope_subs(expr: sympy.Expr, slope_val: sympy.Expr = -s/H, max_iter: Optional[int] = None):
    """Substition of the slope factor dH/ds
    """
    slope = sympy.diff(H, s)
    slope_dict = {slope: slope_val}
    expr = expr.doit()
    i_iter = 0
    while expr.has(slope):
        if max_iter is not None and i_iter >= max_iter:
            break
        expr = expr.subs(slope_dict).doit()
        i_iter += 1
    
    return expr
