# -*- coding: utf-8 -*-
"""
Expansion configuration file - 
Expansion for the transformed variables [extended with boundary terms]
    for anti-symmetric ("dipolar" symmetry) b field

This formulation should only be used in solving the eigenvalue problems

Jingtao Min @ ETH Zurich 2025
"""

import numpy as np
from sympy import S, Symbol, Function, Rational, Abs, jacobi
from .core import s, p, t, H, H_s, cgvar_ptb
from .expansion import n, m, xi, xi_s, s_xi
from . import expansion, base

identifier = "expand_conjugate_eigen_ext-bound_antisym"


# Which equations to use
field_names = [
    fname for fname in base.CollectionConjugate.cg_field_names if fname != "Br_b"
]
field_indexer = np.array([
    True if fname in field_names else False for fname in base.CollectionConjugate.cg_field_names
])
subscript_str = [
    r"\Psi", "M1", "M+", "M-", "z+", "z-",
    "zM1", "zM+", "zM-", "e+", "e-", "ez", "e+,z", "e-,z", 
    "+,+", "-,+", "z,+", "+,-", "-,-", "z,-"]


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
    Psi = H**3*s**Abs(m)*jacobi(n, Rational(3, 2), Abs(m), xi_s),
    # Coupling for magnetic moments
    M_1 = expansion.orth_pref_jacobi(1, Abs(m)),
    M_p = expansion.orth_pref_jacobi(1, Abs(m + 2)),
    M_m = expansion.orth_pref_jacobi(1, Abs(m - 2)),
    M_zp = expansion.orth_pref_jacobi(2, Abs(m + 1)),
    M_zm = expansion.orth_pref_jacobi(2, Abs(m - 1)),
    zM_1 = expansion.orth_pref_jacobi(2, Abs(m)),
    zM_p = expansion.orth_pref_jacobi(2, Abs(m + 2)),
    zM_m = expansion.orth_pref_jacobi(2, Abs(m - 2)),
    # Coupling for equatorial fields
    B_ep = expansion.orth_pref_jacobi(0, Abs(m + 1)),
    B_em = expansion.orth_pref_jacobi(0, Abs(m - 1)),
    Bz_e = expansion.orth_pref_jacobi(0, Abs(m)),
    dB_dz_ep = expansion.orth_pref_jacobi(0, Abs(m + 1)),
    dB_dz_em = expansion.orth_pref_jacobi(0, Abs(m - 1)),
    # Boundary fields
    B_pp = expansion.orth_pref_jacobi(1, Abs(m + 1)),
    B_pm = expansion.orth_pref_jacobi(1, Abs(m - 1)),
    Bz_p = expansion.orth_pref_jacobi(0, Abs(m)),
    B_mp = expansion.orth_pref_jacobi(1, Abs(m + 1)),
    B_mm = expansion.orth_pref_jacobi(1, Abs(m - 1)),
    Bz_m = expansion.orth_pref_jacobi(0, Abs(m))
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
    M_zp = coeff_s.M_zp*bases_s.M_zp,
    M_zm = coeff_s.M_zm*bases_s.M_zm,
    zM_1 = coeff_s.zM_1*bases_s.zM_1,
    zM_p = coeff_s.zM_p*bases_s.zM_p,
    zM_m = coeff_s.zM_m*bases_s.zM_m,
    # Equatorial fields: no coupling
    B_ep = coeff_s.B_ep*bases_s.B_ep,
    B_em = coeff_s.B_em*bases_s.B_em,
    Bz_e = coeff_s.Bz_e*bases_s.Bz_e,
    dB_dz_ep = coeff_s.dB_dz_ep*bases_s.dB_dz_ep,
    dB_dz_em = coeff_s.dB_dz_em*bases_s.dB_dz_em,
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

inner_prod_op = expansion.RadialInnerProducts(field_names,
    **{fname: expansion.InnerProductOp1D(s, S.One, (S.Zero, S.One))
       for fname in field_names}
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

