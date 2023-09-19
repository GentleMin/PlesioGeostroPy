# -*- coding: utf-8 -*-
"""
Forcings in Plesio-Geostrophy Model
Jingtao Min @ ETH-EPM, 09.2023
"""

import sympy
from sympy import diff
from sympy import Derivative as diff_u
from .pg_fields import *
from .base_utils import linearize


"""Lorentz force"""

# Symbols for Lorentz forces
Ls_sym = sympy.Function(r"\overline{L_s}")(s, p, t)
Lp_sym = sympy.Function(r"\overline{L_\phi}")(s, p, t)
Lz_asym = sympy.Function(r"\widetilde{L_z}")(s, p, t)
Le_p = sympy.Function(r"L_{e\phi}")(s, p, t)

# Explicit expressions for Lorentz force
Ls_sym_expr = 1/s*diff(s*Mss, s) + 1/s*diff(Msp, p) - Mpp/s + (Msz_p_expr - Msz_m_expr) + s/H*(Mss_p_expr + Mss_m_expr)
Lp_sym_expr = 1/s*diff(s*Msp, s) + 1/s*diff(Mpp, p) + Msp/s + (Mpz_p_expr - Mpz_m_expr) + s/H*(Msp_p_expr + Msp_m_expr)
Lz_asym_expr = 1/s*diff(s*Msz, s) + 1/s*diff(Mpz, p) + (Mzz_p_expr + Mzz_m_expr - 2*Bz_e*Bz_e) + s/H*(Msz_p_expr - Msz_m_expr)
Le_p_expr = Bs_e*diff(Bp_e, s) + 1/s*Bp_e*diff(Bp_e, p) + Bz_e*dBp_dz_e + 1/s*Bs_e*Bp_e



"""Linearized Lorentz force"""

# Linearized in terms of magnetic fields (for $L_{e\phi}$) or in terms of magnetic moments (for integrated forces).

# Linearized form of Lorentz force in the equatorial plane $L_{e\phi}$
# $L_{e\phi}$ is quadratic in the magnetic field components in the equatorial plane.
# Linearized form involves cross terms between background and perturbational magnetic fields.
Le_p_lin = linearize(Le_p_expr, linearization_subs_map, perturb_var=eps)

# For the integrated quantities, the Lorentz force IS a linear function of magnetic moments. Essentially no linearization required.
# However, the boundary terms and the equatorial terms are quadratic in magnetic fields. These terms need to be linearized.

# Linearized form for $\overline{L_s}$
Ls_sym_lin = linearize(Ls_sym_expr, linearization_subs_map, perturb_var=eps)

# Linearized form for $\overline{L_\phi}$
Lp_sym_lin = linearize(Lp_sym_expr, linearization_subs_map, perturb_var=eps)

# Linearized form for $\widetilde{L_z}$
Lz_asym_lin = linearize(Lz_asym_expr, linearization_subs_map, perturb_var=eps)

# Curl of horizontal components $\nabla \times \mathbf{L}_e$
# Curl is linear, linearize (curl (field)) = curl (linearize (field))
curl_L = cyl_op.curl((Ls_sym_lin, Lp_sym_lin, 0))[2]

