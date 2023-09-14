# -*- coding: utf-8 -*-
"""
Spectral expansion for PG fields
Jingtao Min @ ETH-EPM, 09.2023
"""


import sympy
import types
from sympy import jacobi
from .pg_fields import *


# Symbolic expressions
basis_psi = sympy.Function(r"\widehat{\psi}")(s, m, n)
basis_Mss = sympy.Function(r"\widehat{M_{ss}}")(s, m, n)
basis_Mpp = sympy.Function(r"\widehat{M_{\phi\phi}}")(s, m, n)
basis_Msp = sympy.Function(r"\widehat{M_{s\phi}}")(s, m, n)
basis_Msz = sympy.Function(r"\widehat{M_{sz}}")(s, m, n)
basis_Mpz = sympy.Function(r"\widehat{M_{\phi s}}")(s, m, n)
basis_zMss = sympy.Function(r"\widehat{zM_{ss}}")(s, m, n)
basis_zMpp = sympy.Function(r"\widehat{zM_{\phi\phi}}")(s, m, n)
basis_zMsp = sympy.Function(r"\widehat{zM_{\phi s}}")(s, m, n)
basis_Bs_e = sympy.Function(r"\widehat{B_{es}}")(s, m, n)
basis_Bp_e = sympy.Function(r"\widehat{B_{e\phi}}")(s, m, n)
basis_Bz_e = sympy.Function(r"\widehat{B_{ez}}")(s, m, n)
basis_dBs_dz_e = sympy.Function(r"\widehat{B_{es, z}}")(s, m, n)
basis_dBp_dz_e = sympy.Function(r"\widehat{B_{e\phi, z}}")(s, m, n)
basis_V = sympy.Function("V")(r, theta, p)


# transformation to [-1, 1]
xi = 2*s**2 - 1
# symmetry to m=0
m_abs = sympy.Abs(m)


"""Standard expansions"""
Basis_std = types.SimpleNamespace()

# m-wise prefactors
prefactor_m = sympy.Piecewise((1, sympy.Eq(m, 0)), (s, sympy.Eq(m_abs, 1)), (s**(m_abs - 2), m_abs > 1))
beta_m = sympy.Piecewise((-sympy.Rational(1, 2), sympy.Eq(m, 0)), (sympy.Rational(1, 2), sympy.Eq(m_abs, 1)), (m_abs - sympy.Rational(5, 2), m_abs > 1))

Basis_std.trial_psi = H_s**3*s**m_abs*jacobi(n, sympy.Rational(3, 2), m_abs, xi)

Basis_std.trial_Mss = H_s*prefactor_m*jacobi(n, 1, beta_m, xi)
Basis_std.trial_Mpp = H_s*prefactor_m*jacobi(n, 1, beta_m, xi)
Basis_std.trial_Msp = H_s**2*s**(sympy.Abs(m_abs - 2))*jacobi(n, 2, sympy.Abs(m_abs - 2) - sympy.Rational(1, 2), xi)

Basis_std.trial_Msz = H_s**2*s**(sympy.Abs(m_abs - 1))*jacobi(n, 2, sympy.Abs(m_abs - 1) - sympy.Rational(1, 2), xi)
Basis_std.trial_Mpz = H_s**2*s**(sympy.Abs(m_abs - 1))*jacobi(n, 2, sympy.Abs(m_abs - 1) - sympy.Rational(1, 2), xi)

Basis_std.trial_zMss = H_s**2*prefactor_m*jacobi(n, 2, beta_m, xi)
Basis_std.trial_zMpp = H_s**2*prefactor_m*jacobi(n, 2, beta_m, xi)
Basis_std.trial_zMsp = H_s**2*s**(sympy.Abs(m_abs - 2))*jacobi(n, 2, sympy.Abs(m_abs - 2) - sympy.Rational(1, 2), xi)

Basis_std.trial_Bs_e = H_s**2*s**(sympy.Abs(m_abs - 1))*jacobi(n, 2, sympy.Abs(m_abs - 1) - sympy.Rational(1, 2), xi)
Basis_std.trial_Bp_e = H_s**2*s**(sympy.Abs(m_abs - 1))*jacobi(n, 2, m_abs - sympy.Rational(1, 2), xi)
Basis_std.trial_Bz_e = H_s**2*s**m_abs*jacobi(n, 2, m_abs - sympy.Rational(1, 2), xi)

Basis_std.trial_dBs_dz_e = H_s**2*s**(sympy.Abs(m_abs - 1))*jacobi(n, 2, sympy.Abs(m_abs - 1) - sympy.Rational(1, 2), xi)
Basis_std.trial_dBp_dz_e = H_s**2*s**(sympy.Abs(m_abs - 1))*jacobi(n, 2, sympy.Abs(m_abs - 1) - sympy.Rational(1, 2), xi)



