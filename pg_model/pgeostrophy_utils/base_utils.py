# -*- coding: utf-8 -*-
"""
Utilities for PG model
Jingtao Min @ ETH-EPM, 09.2023
"""

import sympy
from ..sympy_supp import vector_calculus_3d as v3d
from .pg_fields import H, H_s, s, p, z, cyl_op, eps


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
    
    B_field should be given in cylindrical components, 
    and the field components should be functions of cylindrical coordinates,
    i.e. B_field = (
        B_s(s, p, z),
        B_p(s, p, z),
        B_z(s, p, z)
    )
    """
    assert len(B_field) == 3
    Mss = integrate_sym(B_field[0]*B_field[0])
    Mpp = integrate_sym(B_field[1]*B_field[1])
    Msp = integrate_sym(B_field[0]*B_field[1])
    Msz = integrate_asym(B_field[0]*B_field[2])
    Mpz = integrate_asym(B_field[1]*B_field[2])
    zMss = integrate_asym(z*B_field[0]*B_field[0])
    zMpp = integrate_asym(z*B_field[1]*B_field[1])
    zMsp = integrate_asym(z*B_field[0]*B_field[1])
    return Mss, Mpp, Msp, Msz, Mpz, zMss, zMpp, zMsp


def linearize(expr, *subs_maps, perturb_var=eps):
    expr_lin = expr
    for subs_map in subs_maps:
        expr_lin = expr_lin.subs(subs_map)
    expr_lin = expr_lin.simplify().expand().coeff(perturb_var, 1)
    return expr_lin
