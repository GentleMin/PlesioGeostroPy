# -*- coding: utf-8 -*-
"""
Core utilities: coordinates, fields, quantities 
and controlling parameters in PG model
Jingtao Min @ ETH-EPM, 09.2023
"""


import sympy
from sympy import diff
from ..sympy_supp import vector_calculus_3d as v3d
from . import base



"""Independent variables"""

# Coordinates
x, y, z, t = sympy.symbols("x, y, z, t", real=True)
s, p, r, theta = sympy.symbols(r"s, \phi, r, \theta", positive=True)
# Cylindrical coordinates
cyl = v3d.CylindricalCoordinates(s, p, z)
# Spherical coordinates
sph = v3d.SphericalCoordinates(r, theta, p)

# Half cylinder height (symbol + expression)
H = sympy.Function("H")(s)
H_s = sympy.sqrt(1 - s**2)


"""PG-independent physical fields"""

# Magnetic field
B_vec = v3d.Vector3D(
    [
        sympy.Function(r"B_s")(s, p, z, t),
        sympy.Function(r"B_\phi")(s, p, z, t),
        sympy.Function(r"B_z")(s, p, z, t)
    ], 
    coord_sys=cyl
)
# In spherical coordinates
B_sph = v3d.Vector3D(
    [
        sympy.Function(r"B_r")(r, theta, p, t),
        sympy.Function(r"B_\theta")(r, theta, p, t),
        sympy.Function(r"B_\phi")(r, theta, p, t)
    ], 
    coord_sys=sph
)
# Velocity field
U_vec = v3d.Vector3D(
    [
        sympy.Function(r"U_s")(s, p, z, t),
        sympy.Function(r"U_\phi")(s, p, z, t),
        sympy.Function(r"U_z")(s, p, z, t)
    ], 
    coord_sys=cyl
)
# In spherical coordinates
U_sph = v3d.Vector3D(
    [
        sympy.Function(r"U_r")(r, theta, p, t),
        sympy.Function(r"U_\theta")(r, theta, p, t),
        sympy.Function(r"U_\phi")(r, theta, p, t)
    ],
    coord_sys=sph
)


"""Complete fields in PG"""

pgvar = base.CollectionPG(
    # Vorticity
    Psi = sympy.Function(r"\Psi")(s, p, t),
    # Magnetic moments
    Mss = sympy.Function(r"\overline{M_{ss}}")(s, p, t),
    Msp = sympy.Function(r"\overline{M_{s\phi}}")(s, p, t),
    Mpp = sympy.Function(r"\overline{M_{\phi\phi}}")(s, p, t),
    Msz = sympy.Function(r"\widetilde{M_{sz}}")(s, p, t),
    Mpz = sympy.Function(r"\widetilde{M_{\phi z}}")(s, p, t),
    zMss = sympy.Function(r"\widetilde{zM_{ss}}")(s, p, t),
    zMpp = sympy.Function(r"\widetilde{zM_{\phi\phi}}")(s, p, t),
    zMsp = sympy.Function(r"\widetilde{zM_{s\phi}}")(s, p, t),
    # Magnetic field on the equatorial plane
    Bs_e = sympy.Function(r"B_{es}")(s, p, t),
    Bp_e = sympy.Function(r"B_{e\phi}")(s, p, t),
    Bz_e = sympy.Function(r"B_{ez}")(s, p, t),
    dBs_dz_e = sympy.Function(r"B_{es, z}")(s, p, t),
    dBp_dz_e = sympy.Function(r"B_{e\phi, z}")(s, p, t),
    # Boundary magnetic fields
    Br_b = sympy.Function(r"B_{r1}")(theta, p, t),
    Bs_p = sympy.Function(r"B_s^+")(s, p, t),
    Bp_p = sympy.Function(r"B_\phi^+")(s, p, t),
    Bz_p = sympy.Function(r"B_z^+")(s, p, t),
    Bs_m = sympy.Function(r"B_s^-")(s, p, t),
    Bp_m = sympy.Function(r"B_\phi^-")(s, p, t),
    Bz_m = sympy.Function(r"B_z^-")(s, p, t)
)

# PG velocity ansatz
U_pg = v3d.Vector3D(
    [
        1/(s*H)*diff(pgvar.Psi, p),
        -1/H*diff(pgvar.Psi, s),
        z/(s*H**2)*diff(H, s)*diff(pgvar.Psi, p)
    ], 
    coord_sys=cyl
)



"""Background fields"""

# Magnetic field
B0_vec = v3d.Vector3D(
    [
        sympy.Function(r"B_s^0")(s, p, z),
        sympy.Function(r"B_\phi^0")(s, p, z),
        sympy.Function(r"B_z^0")(s, p, z)
    ], 
    coord_sys=cyl
)
# In spherical coordinates
B0_sph = v3d.Vector3D(
    [
        sympy.Function(r"B_r^0")(r, theta, p),
        sympy.Function(r"B_\theta^0")(r, theta, p),
        sympy.Function(r"B_\phi^0")(r, theta, p)
    ], 
    coord_sys=sph
)
# Velocity field
U0_vec = v3d.Vector3D(
    [
        sympy.Function(r"U_s^0")(s, p, z),
        sympy.Function(r"U_\phi^0")(s, p, z),
        sympy.Function(r"U_z^0")(s, p, z)
    ], 
    coord_sys=cyl
)
# In spherical coordinates
U0_sph = v3d.Vector3D(
    [
        sympy.Function(r"U_r^0")(r, theta, p),
        sympy.Function(r"U_\theta^0")(r, theta, p),
        sympy.Function(r"U_\phi^0")(r, theta, p)
    ],
    coord_sys=sph
)

pgvar_bg = base.CollectionPG(
    # Vorticity
    Psi = sympy.Function(r"\Psi^0")(s, p),
    # Magnetic moments
    Mss = sympy.Function(r"\overline{M_{ss}}^0")(s, p),
    Msp = sympy.Function(r"\overline{M_{s\phi}}^0")(s, p),
    Mpp = sympy.Function(r"\overline{M_{\phi\phi}}^0")(s, p),
    Msz = sympy.Function(r"\widetilde{M_{sz}}^0")(s, p),
    Mpz = sympy.Function(r"\widetilde{M_{\phi z}}^0")(s, p),
    zMss = sympy.Function(r"\widetilde{zM_{ss}}^0")(s, p),
    zMpp = sympy.Function(r"\widetilde{zM_{\phi\phi}}^0")(s, p),
    zMsp = sympy.Function(r"\widetilde{zM_{s\phi}}^0")(s, p),
    # Equatorial magnetic field
    Bs_e = sympy.Function(r"B_{es}^0")(s, p),
    Bp_e = sympy.Function(r"B_{e\phi}^0")(s, p),
    Bz_e = sympy.Function(r"B_{ez}^0")(s, p),
    dBs_dz_e = sympy.Function(r"B_{es, z}^0")(s, p),
    dBp_dz_e = sympy.Function(r"B_{e\phi, z}^0")(s, p),
    # Boundary magnetic fields
    Br_b = sympy.Function(r"B_{r1}^0")(theta, p),
    Bs_p = sympy.Function(r"B_s^{0+}")(s, p),
    Bp_p = sympy.Function(r"B_\phi^{0+}")(s, p),
    Bz_p = sympy.Function(r"B_z^{0+}")(s, p),
    Bs_m = sympy.Function(r"B_s^{0-}")(s, p),
    Bp_m = sympy.Function(r"B_\phi^{0-}")(s, p),
    Bz_m = sympy.Function(r"B_z^{0-}")(s, p)
)



"""Perturbation fields"""

b_vec = v3d.Vector3D(
    [
        sympy.Function(r"b_s")(s, p, z, t),
        sympy.Function(r"b_\phi")(s, p, z, t),
        sympy.Function(r"b_z")(s, p, z, t)
    ], 
    coord_sys=cyl
)
# In spherical coordinates
b_sph = v3d.Vector3D(
    [
        sympy.Function(r"b_r")(r, theta, p, t),
        sympy.Function(r"b_\theta")(r, theta, p, t),
        sympy.Function(r"b_\phi")(r, theta, p, t)
    ], 
    coord_sys=sph
)
# Velocity field
u_vec = v3d.Vector3D(
    [
        sympy.Function(r"u_s")(s, p, z, t),
        sympy.Function(r"u_\phi")(s, p, z, t),
        sympy.Function(r"u_z")(s, p, z, t)
    ], 
    coord_sys=cyl
)
# In spherical coordinates
u_sph = v3d.Vector3D(
    [
        sympy.Function(r"u_r")(r, theta, p, t),
        sympy.Function(r"u_\theta")(r, theta, p, t),
        sympy.Function(r"u_\phi")(r, theta, p, t)
    ],
    coord_sys=sph
)

pgvar_ptb = base.CollectionPG(
    # Vorticity
    Psi = sympy.Function(r"\psi")(s, p, t),
    # Magnetic moments
    Mss = sympy.Function(r"\overline{m_{ss}}")(s, p, t),
    Msp = sympy.Function(r"\overline{m_{s\phi}}")(s, p, t),
    Mpp = sympy.Function(r"\overline{m_{\phi\phi}}")(s, p, t),
    Msz = sympy.Function(r"\widetilde{m_{sz}}")(s, p, t),
    Mpz = sympy.Function(r"\widetilde{m_{\phi z}}")(s, p, t),
    zMss = sympy.Function(r"\widetilde{zm_{ss}}")(s, p, t),
    zMpp = sympy.Function(r"\widetilde{zm_{\phi\phi}}")(s, p, t),
    zMsp = sympy.Function(r"\widetilde{zm_{s\phi}}")(s, p, t),
    # Magnetic field in the equatorial plane
    Bs_e = sympy.Function(r"b_{es}")(s, p, t),
    Bp_e = sympy.Function(r"b_{e\phi}")(s, p, t),
    Bz_e = sympy.Function(r"b_{ez}")(s, p, t),
    dBs_dz_e = sympy.Function(r"b_{es, z}")(s, p, t),
    dBp_dz_e = sympy.Function(r"b_{e\phi, z}")(s, p, t),
    # Magnetic field at the boundary
    Br_b = sympy.Function(r"b_{r1}")(theta, p, t),
    Bs_p = sympy.Function(r"b_s^+")(s, p, t),
    Bp_p = sympy.Function(r"b_\phi^+")(s, p, t),
    Bz_p = sympy.Function(r"b_z^+")(s, p, t),
    Bs_m = sympy.Function(r"b_s^-")(s, p, t),
    Bp_m = sympy.Function(r"b_\phi^-")(s, p, t),
    Bz_m = sympy.Function(r"b_z^-")(s, p, t)
)



"""Linearization utilities"""

# Introduce a small quantity $\epsilon$
eps = sympy.Symbol("\epsilon")

u_linmap = {U_vec[idx]: U0_vec[idx] + eps*U_pg[idx].subs(pgvar.Psi, pgvar_ptb.Psi)
    for idx in range(U_vec.ndim)}
u_linmap.update({U_sph[idx]: U0_sph[idx] + eps*u_sph[idx] 
    for idx in range(U_sph.ndim)})
b_linmap = {B_vec[idx]: B0_vec[idx] + eps*b_vec[idx] 
    for idx in range(B_vec.ndim)}
b_linmap.update({B_sph[idx]: B0_sph[idx] + eps*b_sph[idx] 
    for idx in range(B_sph.ndim)})
pg_linmap = {pgvar[idx]: pgvar_bg[idx] + eps*pgvar_ptb[idx] 
    for idx in range(pgvar.n_fields)}

def pg_ansatz(expr, formal_vel=U_vec, PG_Psi=pgvar.Psi):
    """Replace formal velocity with PG ansatz
    """
    expr_psi = expr.subs({
        formal_vel[0]: U_pg[0].subs(pgvar.Psi, PG_Psi),
        formal_vel[1]: U_pg[1].subs(pgvar.Psi, PG_Psi),
        formal_vel[2]: U_pg[2].subs(pgvar.Psi, PG_Psi)
    })
    return expr_psi
