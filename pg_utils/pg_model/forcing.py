# -*- coding: utf-8 -*-
"""
Forcings in Plesio-Geostrophy Model
Jingtao Min @ ETH-EPM, 09.2023
"""

import sympy
from sympy import diff
from sympy import Derivative as diff_u
from .core import *
from .base_utils import linearize


"""Lorentz force"""

# Symbols for Lorentz forces
Ls_sym = sympy.Function(r"\overline{L_s}")(s, p, t)
Lp_sym = sympy.Function(r"\overline{L_\phi}")(s, p, t)
Lz_asym = sympy.Function(r"\widetilde{L_z}")(s, p, t)
Le_p = sympy.Function(r"L_{e\phi}")(s, p, t)

# Explicit expressions for Lorentz force
Ls_sym_expr = 1/s*diff(s*pgvar.Mss, s) + 1/s*diff(pgvar.Msp, p) - pgvar.Mpp/s \
    + (pgvar.Bs_p*pgvar.Bz_p - pgvar.Bs_m*pgvar.Bz_m) \
    + s/H*(pgvar.Bs_p*pgvar.Bs_p + pgvar.Bs_m*pgvar.Bs_m)
Lp_sym_expr = 1/s*diff(s*pgvar.Msp, s) + 1/s*diff(pgvar.Mpp, p) + pgvar.Msp/s \
    + (pgvar.Bp_p*pgvar.Bz_p - pgvar.Bp_m*pgvar.Bz_m) \
    + s/H*(pgvar.Bs_p*pgvar.Bp_p + pgvar.Bs_m*pgvar.Bp_m)
Lz_asym_expr = 1/s*diff(s*pgvar.Msz, s) + 1/s*diff(pgvar.Mpz, p) \
    + (pgvar.Bz_p*pgvar.Bz_p + pgvar.Bz_m*pgvar.Bz_m - 2*pgvar.Bz_e*pgvar.Bz_e) \
    + s/H*(pgvar.Bs_p*pgvar.Bz_p - pgvar.Bs_m*pgvar.Bz_m)
Le_p_expr = pgvar.Bs_e*diff(pgvar.Bp_e, s) + 1/s*pgvar.Bp_e*diff(pgvar.Bp_e, p) \
    + pgvar.Bz_e*pgvar.dBp_dz_e + 1/s*pgvar.Bs_e*pgvar.Bp_e

# Convert to conjugate quantities
pg_cg_map = map_pg_to_conjugate(pgvar, cgvar)
Ls_sym_cg = Ls_sym_expr.subs(pg_cg_map)
Lp_sym_cg = Lp_sym_expr.subs(pg_cg_map)
Lz_asym_cg = Lz_asym_expr.subs(pg_cg_map)
Le_p_cg = Le_p_expr.subs(pg_cg_map)

# Mapping for placeholder symbol - explicit expressions for forces
force_explicit = {
    Ls_sym: Ls_sym_expr,
    Lp_sym: Lp_sym_expr,
    Lz_asym: Lz_asym_expr,
    Le_p: Le_p_expr
}

force_explicit_cg = {
    Ls_sym: Ls_sym_cg,
    Lp_sym: Lp_sym_cg,
    Lz_asym: Lz_asym_cg,
    Le_p: Le_p_cg
}


"""Linearized Lorentz force"""

# Linearized in terms of magnetic fields (for $L_{e\phi}$) 
# or in terms of magnetic moments (for integrated forces).

# Linearized form of Lorentz force in the equatorial plane $L_{e\phi}$
# $L_{e\phi}$ is quadratic in the magnetic field components in the equatorial plane.
# Linearized form involves cross terms 
# between background and perturbational magnetic fields.
Le_p_lin = linearize(Le_p_expr, pg_linmap, perturb_var=eps)

# For the integrated quantities, the Lorentz force IS a linear function 
# of magnetic moments. Essentially no linearization required.
# However, the boundary terms and the equatorial terms are quadratic 
# in magnetic fields. These terms need to be linearized.

# Linearized form for $\overline{L_s}$
Ls_sym_lin = linearize(Ls_sym_expr, pg_linmap, perturb_var=eps)

# Linearized form for $\overline{L_\phi}$
Lp_sym_lin = linearize(Lp_sym_expr, pg_linmap, perturb_var=eps)

# Linearized form for $\widetilde{L_z}$
Lz_asym_lin = linearize(Lz_asym_expr, pg_linmap, perturb_var=eps)

# Curl of horizontal components $\nabla \times \mathbf{L}_e$
# Curl is linear, linearize (curl (field)) = curl (linearize (field))
# curl_L = cyl.curl((Ls_sym_lin, Lp_sym_lin, 0))[2]

# Conjugate expressions
Ls_sym_lin_cg = linearize(Ls_sym_cg, cg_linmap, perturb_var=eps)
Lp_sym_lin_cg = linearize(Lp_sym_cg, cg_linmap, perturb_var=eps)
Lz_asym_lin_cg = linearize(Lz_asym_cg, cg_linmap, perturb_var=eps)
Le_p_lin_cg = linearize(Le_p_cg, cg_linmap, perturb_var=eps)


# Mapping for placeholder symbol - explicit expressions for linearized forces
force_explicit_lin = {
    Ls_sym: Ls_sym_lin,
    Lp_sym: Lp_sym_lin,
    Lz_asym: Lz_asym_lin,
    Le_p: Le_p_lin
}

force_explicit_lin_cg = {
    Ls_sym: Ls_sym_lin_cg,
    Lp_sym: Lp_sym_lin_cg,
    Lz_asym: Lz_asym_lin_cg,
    Le_p: Le_p_lin_cg
}
