# -*- coding: utf-8 -*-
"""
Spectral expansion for PG fields
Jingtao Min @ ETH-EPM, 09.2023
"""


import sympy
import types
from sympy import jacobi
from .pg_fields import *


# Symbolic expressions (placeholders)
basis_psi = sympy.Function(r"\psi^{mn}")(s)
basis_Mss = sympy.Function(r"\overline{M_{ss}}^{mn}")(s)
basis_Mpp = sympy.Function(r"\overline{M_{\phi\phi}}^{mn}")(s)
basis_Msp = sympy.Function(r"\overline{M_{s\phi}}^{mn}")(s)
basis_Msz = sympy.Function(r"\widetilde{M_{sz}}^{mn}")(s)
basis_Mpz = sympy.Function(r"\widetilde{M_{\phi z}}^{mn}")(s)
basis_zMss = sympy.Function(r"\widetilde{zM_{ss}}^{mn}")(s)
basis_zMpp = sympy.Function(r"\widetilde{zM_{\phi\phi}}^{mn}")(s)
basis_zMsp = sympy.Function(r"\widetilde{zM_{\phi s}}^{mn}")(s)
basis_Bs_e = sympy.Function(r"B_{es}^{mn}")(s)
basis_Bp_e = sympy.Function(r"B_{e\phi}^{mn}")(s)
basis_Bz_e = sympy.Function(r"B_{ez}^{mn}")(s)
basis_dBs_dz_e = sympy.Function(r"B_{es, z}^{mn}")(s)
basis_dBp_dz_e = sympy.Function(r"B_{e\phi, z}^{mn}")(s)
basis_V = sympy.Function(r"V^{mn}")(r, theta, p)

list_s_basis = [basis_psi, basis_Mss, basis_Mpp, basis_Msp, basis_Msz, basis_Mpz, 
    basis_zMss, basis_zMpp, basis_zMsp, basis_Bs_e, basis_Bp_e, basis_Bz_e, basis_dBs_dz_e, basis_dBp_dz_e, basis_V]

C_psi = sympy.Symbol(r"C_{\psi}^{mn}")
C_Mss = sympy.Symbol(r"C_{ss}^{mn}")
C_Mpp = sympy.Symbol(r"C_{\phi\phi}^{mn}")
C_Msp = sympy.Symbol(r"C_{s\phi}^{mn}")
C_Msz = sympy.Symbol(r"C_{sz}^{mn}")
C_Mpz = sympy.Symbol(r"C_{\phi z}^{mn}")
C_zMss = sympy.Symbol(r"C_{zss}^{mn}")
C_zMpp = sympy.Symbol(r"C_{z\phi\phi}^{mn}")
C_zMsp = sympy.Symbol(r"C_{zs\phi}^{mn}")
C_Bs_e = sympy.Symbol(r"C_{es}^{mn}")
C_Bp_e = sympy.Symbol(r"C_{e\phi}^{mn}")
C_Bz_e = sympy.Symbol(r"C_{ez}^{mn}")
C_dBs_dz_e = sympy.Symbol(r"C_{es,z}^{mn}")
C_dBp_dz_e = sympy.Symbol(r"C_{e\phi,z}^{mn}")
C_V = sympy.Symbol(r"\iota^{mn}")

list_coeffs_fields = [C_psi, C_Mss, C_Mpp, C_Msp, C_Msz, C_Mpz,
    C_zMss, C_zMpp, C_zMsp, C_Bs_e, C_Bp_e, C_Bz_e, C_dBs_dz_e, C_dBp_dz_e, C_V]

def sort_coeffs(expr):
    expr_sorted = sympy.S.Zero
    for coeff in list_coeffs_fields:
        expr_sorted += coeff*expr.coeff(coeff, 1)
    return expr_sorted
    


# transformation to [-1, 1]
xi = 2*s**2 - 1
# symmetry to m=0
m_abs = sympy.Abs(m)


"""Standard expansions"""
Basis_std = types.SimpleNamespace()

# m-wise prefactors
prefactor_m = sympy.Piecewise(
    (1, sympy.Eq(m, 0)), 
    (s, sympy.Eq(m_abs, 1)), 
    (s**(m_abs - 2), m_abs > 1))
beta_m = sympy.Piecewise(
    (-sympy.Rational(1, 2), sympy.Eq(m, 0)), 
    (sympy.Rational(1, 2), sympy.Eq(m_abs, 1)), 
    (m_abs - sympy.Rational(5, 2), m_abs > 1))

# Field expansions
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


# Fourier ansatz
fourier_basis = sympy.exp(sympy.I*(omega*t + m*p))
fourier_ansatz = {list_perturb_fields[i_field]: list_coeffs_fields[i_field]*list_s_basis[i_field]*fourier_basis 
    for i_field in range(len(list_perturb_fields))}

def fourier_domain(expr):
    expr_fourier = expr.subs(fourier_ansatz).doit()/fourier_basis
    return expr_fourier
