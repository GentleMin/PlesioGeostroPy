# -*- coding: utf-8 -*-
"""
Fields and symbols concerned in Plesio-Geostrophy Model
Jingtao Min @ ETH-EPM, 09.2023
"""


import sympy
from sympy import diff
from ..sympy_supp import vector_calculus_3d as v3d



"""Variables"""

# Coordinates
x, y, z, t = sympy.symbols("x, y, z, t", real=True)
s, p, r, theta = sympy.symbols(r"s, \phi, r, \theta", positive=True)

# Integers
n, m = sympy.symbols("n, m", integer=True)

# Half cylinder height (symbol + expression)
H = sympy.Function("H")(s)
H_s = sympy.sqrt(1 - s**2)

# Coordinate systems
cyl_op = v3d.CylindricalCoordinates((s, p, z))
sph_op = v3d.SphericalCoordinates((r, theta, p))

# Angular frequency
omega = sympy.Symbol(r"\omega")


"""Complete fields"""

Psi = sympy.Function(r"\Psi")(s, p, t)

Mss = sympy.Function(r"\overline{M_{ss}}")(s, p, t)
Msp = sympy.Function(r"\overline{M_{s\phi}}")(s, p, t)
Mpp = sympy.Function(r"\overline{M_{\phi\phi}}")(s, p, t)

Msz = sympy.Function(r"\widetilde{M_{sz}}")(s, p, t)
Mpz = sympy.Function(r"\widetilde{M_{\phi z}}")(s, p, t)
zMss = sympy.Function(r"\widetilde{zM_{ss}}")(s, p, t)
zMpp = sympy.Function(r"\widetilde{zM_{\phi\phi}}")(s, p, t)
zMsp = sympy.Function(r"\widetilde{zM_{s\phi}}")(s, p, t)

Bs_e = sympy.Function(r"B_{es}")(s, p, t)
Bp_e = sympy.Function(r"B_{e\phi}")(s, p, t)
Bz_e = sympy.Function(r"B_{ez}")(s, p, t)

dBs_dz_e = sympy.Function(r"B_{es, z}")(s, p, t)
dBp_dz_e = sympy.Function(r"B_{e\phi, z}")(s, p, t)

Br = sympy.Function(r"B_r")(theta, p, t)

list_fields = [Psi, Mss, Mpp, Msp, Msz, Mpz, zMss, zMpp, zMsp, Bs_e, Bp_e, Bz_e, dBs_dz_e, dBp_dz_e, Br]


"""Boundary fields"""

Bs_p = sympy.Function(r"B_s^+")(s, p, t)
Bp_p = sympy.Function(r"B_\phi^+")(s, p, t)
Bz_p = sympy.Function(r"B_z^+")(s, p, t)
Bs_m = sympy.Function(r"B_s^-")(s, p, t)
Bp_m = sympy.Function(r"B_\phi^-")(s, p, t)
Bz_m = sympy.Function(r"B_z^-")(s, p, t)

Msz_p_expr = Bs_p*Bz_p
Mpz_p_expr = Bp_p*Bz_p
Mss_p_expr = Bs_p*Bs_p
Msp_p_expr = Bs_p*Bp_p
Mzz_p_expr = Bz_p*Bz_p
Msz_m_expr = Bs_m*Bz_m
Mpz_m_expr = Bp_m*Bz_m
Mss_m_expr = Bs_m*Bs_m
Msp_m_expr = Bs_m*Bp_m
Mzz_m_expr = Bz_m*Bz_m

list_boundaries = [Bs_p, Bp_p, Bz_p, Bs_m, Bp_m, Bz_m]


"""Auxiliary fields"""

us = sympy.Function(r"u_s")(s, p, t)
up = sympy.Function(r"u_\phi")(s, p, t)
uz = sympy.Function(r"u_z")(s, p, z, t)

us_Psi = 1/(s*H)*diff(Psi, p)
up_Psi = -1/H*diff(Psi, s)
uz_Psi = z/H*diff(H, s)*us_Psi

ur = sympy.Function(r"u_r")(r, theta, p)
ut = sympy.Function(r"u_\theta")(r, theta, p)
up_sph = sympy.Function(r"u_\phi")(r, theta, p)



# For linearization


"""Background fields"""

Psi_0 = sympy.Function(r"\Psi^0")(s, p, t)
us_0 = sympy.Function(r"U_s^0")(s, p, z, t)
up_0 = sympy.Function(r"U_\phi^0")(s, p, z, t)
uz_0 = sympy.Function(r"U_z^0")(s, p, z, t)

Mss_0 = sympy.Function(r"\overline{M_{ss}}^0")(s, p, t)
Msp_0 = sympy.Function(r"\overline{M_{s\phi}}^0")(s, p, t)
Mpp_0 = sympy.Function(r"\overline{M_{\phi\phi}}^0")(s, p, t)

Msz_0 = sympy.Function(r"\widetilde{M_{sz}}^0")(s, p, t)
Mpz_0 = sympy.Function(r"\widetilde{M_{\phi z}}^0")(s, p, t)
zMss_0 = sympy.Function(r"\widetilde{zM_{ss}}^0")(s, p, t)
zMpp_0 = sympy.Function(r"\widetilde{zM_{\phi\phi}}^0")(s, p, t)
zMsp_0 = sympy.Function(r"\widetilde{zM_{s\phi}}^0")(s, p, t)

Bs_e_0 = sympy.Function(r"B_{es}^0")(s, p, t)
Bp_e_0 = sympy.Function(r"B_{e\phi}^0")(s, p, t)
Bz_e_0 = sympy.Function(r"B_{ez}^0")(s, p, t)

dBs_dz_e_0 = sympy.Function(r"B_{es, z}^0")(s, p, t)
dBp_dz_e_0 = sympy.Function(r"B_{e\phi, z}^0")(s, p, t)

Br_0 = sympy.Function(r"B_r^0")(theta, p, t)

list_bg_fields = [Psi_0, Mss_0, Mpp_0, Msp_0, Msz_0, Mpz_0, zMss_0, zMpp_0, zMsp_0, Bs_e_0, Bp_e_0, Bz_e_0, dBs_dz_e_0, dBp_dz_e_0, Br_0]

# Boundary terms

Bs_p_0 = sympy.Function(r"B_s^{0+}")(s, p, t)
Bp_p_0 = sympy.Function(r"B_\phi^{0+}")(s, p, t)
Bz_p_0 = sympy.Function(r"B_z^{0+}")(s, p, t)
Bs_m_0 = sympy.Function(r"B_s^{0-}")(s, p, t)
Bp_m_0 = sympy.Function(r"B_\phi^{0-}")(s, p, t)
Bz_m_0 = sympy.Function(r"B_z^{0-}")(s, p, t)

list_bg_boundaries = [Bs_p_0, Bp_p_0, Bz_p_0, Bs_m_0, Bp_m_0, Bz_m_0]


"""Perturbation fields"""

psi = sympy.Function(r"\psi")(s, p, t)
us_psi = 1/(s*H)*diff(psi, p)
up_psi = -1/H*diff(psi, s)
uz_psi = z/H*diff(H, s)*us_psi

mss = sympy.Function(r"\overline{m_{ss}}")(s, p, t)
msp = sympy.Function(r"\overline{m_{s\phi}}")(s, p, t)
mpp = sympy.Function(r"\overline{m_{\phi\phi}}")(s, p, t)

msz = sympy.Function(r"\widetilde{m_{sz}}")(s, p, t)
mpz = sympy.Function(r"\widetilde{m_{\phi z}}")(s, p, t)
zmss = sympy.Function(r"\widetilde{zm_{ss}}")(s, p, t)
zmpp = sympy.Function(r"\widetilde{zm_{\phi\phi}}")(s, p, t)
zmsp = sympy.Function(r"\widetilde{zm_{s\phi}}")(s, p, t)

bs_e = sympy.Function(r"b_{es}")(s, p, t)
bp_e = sympy.Function(r"b_{e\phi}")(s, p, t)
bz_e = sympy.Function(r"b_{ez}")(s, p, t)

dbs_dz_e = sympy.Function(r"b_{es, z}")(s, p, t)
dbp_dz_e = sympy.Function(r"b_{e\phi, z}")(s, p, t)

br = sympy.Function(r"b_r")(theta, p, t)

list_perturb_fields = [psi, mss, mpp, msp, msz, mpz, zmss, zmpp, zmsp, bs_e, bp_e, bz_e, dbs_dz_e, dbp_dz_e, br]

# Boundary terms

bs_p = sympy.Function(r"b_s^+")(s, p, t)
bp_p = sympy.Function(r"b_\phi^+")(s, p, t)
bz_p = sympy.Function(r"b_z^+")(s, p, t)
bs_m = sympy.Function(r"b_s^-")(s, p, t)
bp_m = sympy.Function(r"b_\phi^-")(s, p, t)
bz_m = sympy.Function(r"b_z^-")(s, p, t)

list_perturb_boundaries = [bs_p, bp_p, bz_p, bs_m, bp_m, bz_m]


# Introduce a small quantity $\epsilon$
eps = sympy.Symbol("\epsilon")

# Substitutions / expansions
linearization_subs_map = {
    list_fields[idx_field]: list_bg_fields[idx_field] + eps*list_perturb_fields[idx_field] for idx_field in range(len(list_fields))
}
linearization_subs_map.update({
    list_boundaries[idx_bound]: list_bg_boundaries[idx_bound] + eps*list_perturb_boundaries[idx_bound] for idx_bound in range(len(list_boundaries))
})

