# -*- coding: utf-8 -*-
"""
Expansion configuration file - 
Expansion of PG variable, implementing partial coupling, as documented in Daria's dissertation
Augmented with spectral expansion of boundary quantities

Jingtao Min @ ETH Zurich 2024
"""

import numpy as np
from sympy import *
from .core import s, p, t, H, H_s, pgvar_ptb
from .expansion import n, m, xi, xi_s, s_xi
from . import expansion, base

identifier = "expand_pg_eigen_ext-bound"


# Which equations to use
field_names = [
    fname for fname in base.CollectionPG.pg_field_names if fname != "Br_b"
]
field_indexer = np.array([
    True if fname != "Br_b" else False for fname in base.CollectionPG.pg_field_names
])
subscript_str = [
    r"\Psi", "ss", r"\phi\phi", r"s\phi", "sz", r"\phi z",
    "zss", r"z\phi\phi", r"zs\phi", "es", r"e\phi", "ez", "es,z", r"e\phi,z", 
    "s,+", r"\phi,+", "z,+", "s,-", r"\phi,-", "z,-"
]


"""Fourier expansion"""

# fields
fields = pgvar_ptb.generate_collection(field_indexer)

# Fourier expansion
fourier_expand = expansion.FourierExpansions(
    expansion.omega*t + expansion.m*p, fields, 
    expansion.pgvar_s.generate_collection(field_indexer))


"""Radial expansion"""

# bases (placeholder)
bases_s = base.LabeledCollection(field_names,
    **{fname: Function(r"\Phi_{%s}^{mn}" % subscript_str[idx])(s) 
       for idx, fname in enumerate(field_names)})

# explicit expression for the radial bases
m_abs = Abs(m)
pow_m = Abs(Abs(m_abs - S.One) - S.One)
# beta_m = pow_m - Rational(1, 2)

coupling_p = Piecewise(
    (S.Zero, Eq(m, 0)),
    (S.One, m_abs > 0)
)
coupling_pp = Piecewise(
    (S.One, Eq(m, S.Zero)), 
    (S.Zero, Eq(m_abs, S.One)), 
    (-S.One, m_abs > S.One)
)
coupling_sp = Piecewise(
    (S.Zero, m_abs <= S.One), 
    (S.One, m_abs > S.One)
)
# coupling_p = Abs(m_abs + Rational(1, 2)) - Abs(m_abs - Rational(1, 2))
# coupling_pp = 1 - (Abs(m_abs + 2) - Abs(m_abs - 2))/2
# coupling_sp = (1 + Abs(m_abs - 1) - Abs(m_abs - 2))/2


bases_s_expression = base.LabeledCollection(field_names,
    Psi = H**3*s**m_abs*jacobi(n, Rational(3, 2), m_abs, xi_s),
    # Coupling for magnetic moments
    Mss = expansion.orth_pref_jacobi(1, pow_m),
    Mpp = expansion.orth_pref_jacobi(1, pow_m + 2*Abs(coupling_pp)),
    Msp = expansion.orth_pref_jacobi(1, Abs(m_abs - 2) + 2*Abs(coupling_sp)),
    Msz = expansion.orth_pref_jacobi(2, Abs(m_abs - 1)),
    Mpz = expansion.orth_pref_jacobi(2, Abs(m_abs - 1) + 2*Abs(coupling_p)),
    zMss = expansion.orth_pref_jacobi(2, pow_m),
    zMpp = expansion.orth_pref_jacobi(2, pow_m + 2*Abs(coupling_pp)),
    zMsp = expansion.orth_pref_jacobi(2, Abs(m_abs - 2) + 2*Abs(coupling_sp)),
    # Coupling for equatorial fields
    Bs_e = expansion.orth_pref_jacobi(0, Abs(m_abs - 1)),
    Bp_e = expansion.orth_pref_jacobi(0, Abs(m_abs - 1) + 2*Abs(coupling_p)),
    Bz_e = expansion.orth_pref_jacobi(0, m_abs),
    dBs_dz_e = expansion.orth_pref_jacobi(0, Abs(m_abs - 1)),
    dBp_dz_e = expansion.orth_pref_jacobi(0, Abs(m_abs - 1) + 2*Abs(coupling_p)),
    # Boundary fields
    Bs_p = expansion.orth_pref_jacobi(0, Abs(m_abs - 1)),
    Bp_p = expansion.orth_pref_jacobi(0, Abs(m_abs - 1) + 2*Abs(coupling_p)),
    Bz_p = expansion.orth_pref_jacobi(0, Abs(m)),
    Bs_m = expansion.orth_pref_jacobi(0, Abs(m_abs - 1)),
    Bp_m = expansion.orth_pref_jacobi(0, Abs(m_abs - 1) + 2*Abs(coupling_p)),
    Bz_m = expansion.orth_pref_jacobi(0, Abs(m))
)

# coefficients
coeff_s = base.LabeledCollection(field_names,
    **{fname: Symbol(r"C_{%s}^{mn}" % subscript_str[idx])
       for idx, fname in enumerate(field_names)})

rad_expand = expansion.RadialExpansions(fourier_expand.coeffs, bases_s, coeff_s,
    Psi = coeff_s.Psi*bases_s.Psi,
    # Lowest-order coupling between Mss, Mpp and Msp
    Mss = coeff_s.Mss*bases_s.Mss,
    Mpp = coeff_s.Mpp*bases_s.Mpp \
        + coupling_pp*coeff_s.Mss*bases_s.Mss,
    Msp = coeff_s.Msp*bases_s.Msp \
        + coupling_sp*I*sign(m)*coeff_s.Mss*bases_s.Mss,
    # Lowest-order coupling between Msz and Mpz
    Msz = coeff_s.Msz*bases_s.Msz,
    Mpz = coeff_s.Mpz*bases_s.Mpz \
        + coupling_p*I*sign(m)*coeff_s.Msz*bases_s.Msz,
    # Lowest-order coupling between zMss, zMpp and zMsp
    zMss = coeff_s.zMss*bases_s.zMss,
    zMpp = coeff_s.zMpp*bases_s.zMpp \
        + coupling_pp*coeff_s.zMss*bases_s.zMss,
    zMsp = coeff_s.zMsp*bases_s.zMsp \
        + coupling_sp*I*sign(m)*coeff_s.zMss*bases_s.zMss,
    # Lowest-order coupling between Bs_e and Bp_e
    Bs_e = coeff_s.Bs_e*bases_s.Bs_e,
    Bp_e = coeff_s.Bp_e*bases_s.Bp_e \
        + coupling_p*I*sign(m)*coeff_s.Bs_e*bases_s.Bs_e,
    Bz_e = coeff_s.Bz_e*bases_s.Bz_e,
    # Lowest-order coupling between dBs_dz_e and dBp_dz_e
    dBs_dz_e = coeff_s.dBs_dz_e*bases_s.dBs_dz_e,
    dBp_dz_e = coeff_s.dBp_dz_e*bases_s.dBp_dz_e \
        + coupling_p*I*sign(m)*coeff_s.dBs_dz_e*bases_s.dBs_dz_e,
    # Lowest-order coupling between Bs_[pm] and Bp_[pm]
    Bs_p = coeff_s.Bs_p*bases_s.Bs_p,
    Bp_p = coeff_s.Bp_p*bases_s.Bp_p \
        + coupling_p*I*sign(m)*coeff_s.Bs_p*bases_s.Bs_p,
    Bz_p = coeff_s.Bz_p*bases_s.Bz_p,
    Bs_m = coeff_s.Bs_m*bases_s.Bs_m,
    Bp_m = coeff_s.Bp_m*bases_s.Bp_m \
        + coupling_p*I*sign(m)*coeff_s.Bs_m*bases_s.Bs_m,
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

