# -*- coding: utf-8 -*-
"""
Utilities for PG model
Jingtao Min @ ETH-EPM, 09.2023
"""

import sympy
from ..sympy_supp import vector_calculus_3d as v3d
from .core import *


def integrate_sym(field):
    """equatorially symmetric / even integral
    Note: input field has to be a function of cylindrical coordinates
    """
    return sympy.integrate(field, (z, -H, H))


def integrate_asym(field):
    """equatorially asymmetric / odd integral
    Note: input field has to be a function of cylindrical coordinates
    """
    return sympy.integrate(field, (z, 0, H)) + sympy.integrate(field, (z, 0, -H))


def field_to_moment(B_field):
    """Convert magnetic field to integrated moments.
    
    :param B_field: tuple, magnetic field components
        B_field should be given in cylindrical components, 
        and the field components should be functions of cylindrical coordinates,
        i.e. B_field = (
            B_s(s, p, z),
            B_p(s, p, z),
            B_z(s, p, z)
        )
    :return: tuple of the sympy expressions of the eight moments,
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


def assemble_background(B0, Psi0=None, mode="PG"):
    """Assemble background PG fields
    
    :param B0: array-like or Vector3D, iterable and indexable
        background magnetic field in cylindrical coordinates
    :param Psi0: sympy.Function, the background stream function
        Because velocity cannot in general be converted to Psi,
        if one wants to specify the stream function this has to
        be done separately. Note that when the nonlinear term is 
        absent from the vorticity equation, Psi0 will in general
        not be involved in any of these equations. Background
        velocity in all induction equations uses U instead of Psi
    :param mode: str, the mode, what kind of background fields to
        assemble. Supports Plesio-Geostrophy "PG", or conjugate 
        variables "CG". Default is PG.
    
    :returns: CollectionPG, collection of background PG fields
    
    .. note:: The components of B0 and the field Psi0 need to be
        "directly evaluable", i.e. they should all be sympy expr
        containing no derivative of undefined functions.
        Otherwise, substitution will return incorrect results.
        The background field at the boundary are not immediately 
        evaluated at $z=\pm H$, since their z-derivatives are 
        present in the induction equations. One has to evaluate 
        after substitution and simplification.
    """
    moments_bg = field_to_moment(B0)
    pg_background = base.CollectionPG(
        # Vorticity
        Psi = Psi0 if Psi0 is not None else sympy.S.Zero,
        # Magnetic moments
        Mss = moments_bg[0].subs({H: H_s}),
        Mpp = moments_bg[1].subs({H: H_s}),
        Msp = moments_bg[2].subs({H: H_s}),
        Msz = moments_bg[3].subs({H: H_s}),
        Mpz = moments_bg[4].subs({H: H_s}),
        zMss = moments_bg[5].subs({H: H_s}),
        zMpp = moments_bg[6].subs({H: H_s}),
        zMsp = moments_bg[7].subs({H: H_s}),
        # Magnetic field in the equatorial plane
        Bs_e = B0[0].subs({z: 0}),
        Bp_e = B0[1].subs({z: 0}),
        Bz_e = B0[2].subs({z: 0}),
        dBs_dz_e = diff(B0[0], z).doit().subs({z: 0}),
        dBp_dz_e = diff(B0[0], z).doit().subs({z: 0}),
        # Magnetic field at the boundary
        Br_b = s*B0[0] + z*B0[2],
        Bs_p = B0[0],
        Bp_p = B0[1],
        Bz_p = B0[2],
        Bs_m = B0[0],
        Bp_m = B0[1],
        Bz_m = B0[2]
    )
    if mode.lower() == "pg":
        return pg_background
    elif mode.lower() == "cg":
        return PG_to_conjugate(pg_background)


def linearize(expr, *subs_maps, perturb_var=eps):
    """Linearize expression
    
    :param expr: sympy expression, expression to be linearized
    :param *subs_maps: dict, subtitution maps, takes the form
        {A: A0 + eps*A1, B: B0 + eps*B1, ...}
    :param perturb_var: perturbation number, 
        default to be the symbol eps from pg_fields module
    :return: sympy expression, linearized expression
    """
    expr_lin = expr
    for subs_map in subs_maps:
        expr_lin = expr_lin.subs(subs_map)
    expr_lin = expr_lin.doit().expand().coeff(perturb_var, 1)
    return expr_lin


def fields_in_term(expr: sympy.Expr, field_collection: base.LabeledCollection):
    """Extract all fields in a term from a collection.
    
    :param expr: the term from which the field will be extracted
    :param field_collection: range of fields
    :returns: set of fields
    """
    set_tmp = {field for field in field_collection}
    expr_fields = expr.atoms(sympy.Function)
    expr_fields = tuple(field for field in expr_fields if field in set_tmp)
    return expr_fields

