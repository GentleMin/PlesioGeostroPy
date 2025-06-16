# -*- coding: utf-8 -*-
"""
Expansion configuration file for revised PG formulation
Expansion for the canonical variables [extended with boundary terms]
    for equatorially symmetric ("quadrupolar" symmetry) b field

This formulation should only be used in solving the eigenvalue problems

Jingtao Min @ ETH Zurich 2025
"""

import numpy as np
from sympy import S, Symbol, Function, Rational, Abs, jacobi
from .core import s, p, t, H, H_s, cgvar_ptb
from .expansion import n, m, xi, xi_s, s_xi
from . import expansion, base

identifier = "expand_eigen_canonical_ext-bound_sym"


# Which equations to use
field_names = [
    fname for fname in base.CollectionConjugate.cg_field_names if fname != "Br_b"
]
field_indexer = np.array([
    True if fname in field_names else False for fname in base.CollectionConjugate.cg_field_names
])
subscript_str = [
    r"\Psi", 
    "M1", "M++", "M--", "Mzz", "zMz+", "zMz-", "z2M1", "z2M++", "z2M--", 
    "+,+", "-,+", "z,+", "+,-", "-,-", "z,-"
]


"""Fourier expansion"""

# fields
fields = cgvar_ptb.generate_collection(field_indexer)

# Fourier expansion
fourier_expand = expansion.FourierExpansions(
    expansion.omega*t + expansion.m*p, fields, 
    expansion.cgvar_s.generate_collection(field_indexer))


"""Radial expansion"""

# bases (placeholder)
bases_s = base.LabeledCollection(field_names,
    **{fname: Function(r"\Phi_{%s}^{mn}" % subscript_str[idx])(s) 
       for idx, fname in enumerate(field_names)})

# explicit expression for the radial bases

bases_s_expression = base.LabeledCollection(
    field_names,
    
    Psi = expansion.jacobi_2_side(3, Abs(m), Rational(3, 2), Abs(m)),
    
    # Magnetic moments
    M_1 = expansion.jacobi_2_side(1, Abs(m), S.Half, Abs(m) - S.Half),
    M_p = expansion.jacobi_2_side(1, Abs(m + 2), S.Half, Abs(m + 2) - S.Half),
    M_m = expansion.jacobi_2_side(1, Abs(m - 2), S.Half, Abs(m - 2) - S.Half),
    Mzz = expansion.jacobi_2_side(1, Abs(m), S.Half, Abs(m) - S.Half),
    zM_zp = expansion.jacobi_2_side(3, Abs(m + 1), 3 - S.Half, Abs(m + 1) - S.Half),
    zM_zm = expansion.jacobi_2_side(3, Abs(m + 1), 3 - S.Half, Abs(m - 1) - S.Half),
    z2M_1 = expansion.jacobi_2_side(3, Abs(m), 3 - S.Half, Abs(m) - S.Half),
    z2M_p = expansion.jacobi_2_side(3, Abs(m + 2), 3 - S.Half, Abs(m + 2) - S.Half),
    z2M_m = expansion.jacobi_2_side(3, Abs(m - 2), 3 - S.Half, Abs(m - 2) - S.Half),
    
    # Boundary fields
    B_pp = expansion.jacobi_2_side(0, Abs(m + 1), -S.Half, Abs(m + 1) - S.Half),
    B_pm = expansion.jacobi_2_side(0, Abs(m - 1), -S.Half, Abs(m - 1) - S.Half),
    Bz_p = expansion.jacobi_2_side(1, Abs(m), S.Half, Abs(m) - S.Half),
    B_mp = expansion.jacobi_2_side(0, Abs(m + 1), -S.Half, Abs(m + 1) - S.Half),
    B_mm = expansion.jacobi_2_side(0, Abs(m - 1), -S.Half, Abs(m - 1) - S.Half),
    Bz_m = expansion.jacobi_2_side(1, Abs(m), S.Half, Abs(m) - S.Half),
)

# coefficients
coeff_s = base.LabeledCollection(field_names,
    **{fname: Symbol(r"C_{%s}^{mn}" % subscript_str[idx])
       for idx, fname in enumerate(field_names)})

rad_expand = expansion.RadialExpansions(fourier_expand.coeffs, bases_s, coeff_s,
    Psi = coeff_s.Psi*bases_s.Psi,
    
    # Conjugate moments: no coupling
    M_1 = coeff_s.M_1*bases_s.M_1,
    M_p = coeff_s.M_p*bases_s.M_p,
    M_m = coeff_s.M_m*bases_s.M_m,
    Mzz = coeff_s.Mzz*bases_s.Mzz,
    zM_zp = coeff_s.zM_zp*bases_s.zM_zp,
    zM_zm = coeff_s.zM_zm*bases_s.zM_zm,
    z2M_1 = coeff_s.z2M_1*bases_s.z2M_1,
    z2M_p = coeff_s.z2M_p*bases_s.z2M_p,
    z2M_m = coeff_s.z2M_m*bases_s.z2M_m,
    
    # Boundary fields
    B_pp = coeff_s.B_pp*bases_s.B_pp,
    B_pm = coeff_s.B_pm*bases_s.B_pm,
    Bz_p = coeff_s.Bz_p*bases_s.Bz_p,
    B_mp = coeff_s.B_mp*bases_s.B_mp,
    B_mm = coeff_s.B_mm*bases_s.B_mm,
    Bz_m = coeff_s.Bz_m*bases_s.Bz_m,
)

"""Test functions"""

# Even when the test functions are the same with the trial functions
# It is required that they use different indices (or different placeholders)
# so that the two can be distinguished.
test_s = base.LabeledCollection(field_names,
    **{fname: Function(r"\Phi_{%s}^{mn'}" % subscript_str[idx])(s) 
       for idx, fname in enumerate(field_names)})


"""Inner products"""

inner_prod_op = expansion.RadialInnerProducts(
    field_names,
    Psi=expansion.InnerProductOp1D(s, S.One, (S.Zero, S.One)),
    **{fname: expansion.InnerProductOp1D(s, 1/H, (S.Zero, S.One))
       for fname in field_names[1:]}
)

recipe = expansion.ExpansionRecipe(
    identifier=identifier,
    fourier_expand=fourier_expand,
    rad_expand=rad_expand,
    rad_test=test_s,
    inner_prod_op=inner_prod_op,
    base_expr=bases_s_expression.subs({n: expansion.n_trial}),
    test_expr=bases_s_expression.subs({n: expansion.n_test})
)

