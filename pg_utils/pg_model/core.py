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
    # Stream function
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
    # Stream function
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
    # Stream function
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



"""Conjugate variables"""

cgvar = base.CollectionConjugate(
    # Stream function, unchanged
    Psi = pgvar.Psi,
    # Conjugate variables for magnetic moments
    M_1 = sympy.Function(r"\overline{M_1}")(s, p, t),
    M_p = sympy.Function(r"\overline{M_+}")(s, p, t),
    M_m = sympy.Function(r"\overline{M_-}")(s, p, t),
    M_zp = sympy.Function(r"\widetilde{M_{z+}}")(s, p, t),
    M_zm = sympy.Function(r"\widetilde{M_{z-}}")(s, p, t),
    zM_1 = sympy.Function(r"\widetilde{zM_1}")(s, p, t),
    zM_p = sympy.Function(r"\widetilde{zM_+}")(s, p, t),
    zM_m = sympy.Function(r"\widetilde{zM_-}")(s, p, t),
    # Conjugate variables for magnetic fields in equatorial plane
    B_ep = sympy.Function(r"B_{e+}")(s, p, t),
    B_em = sympy.Function(r"B_{e-}")(s, p, t),
    Bz_e = pgvar.Bz_e,
    dB_dz_ep = sympy.Function(r"B_{e+, z}")(s, p, t),
    dB_dz_em = sympy.Function(r"B_{e-, z}")(s, p, t),
    # Magnetic field at the boundary
    Br_b = pgvar.Br_b,
    Bs_p = pgvar.Bs_p,
    Bp_p = pgvar.Bp_p,
    Bz_p = pgvar.Bz_p,
    Bs_m = pgvar.Bs_m,
    Bp_m = pgvar.Bp_m,
    Bz_m = pgvar.Bz_m
)

cgvar_ptb = base.CollectionConjugate(
    # Stream function, unchanged
    Psi = pgvar_ptb.Psi,
    # Conjugate variables for magnetic moments
    M_1 = sympy.Function(r"\overline{m_1}")(s, p, t),
    M_p = sympy.Function(r"\overline{m_+}")(s, p, t),
    M_m = sympy.Function(r"\overline{m_-}")(s, p, t),
    M_zp = sympy.Function(r"\widetilde{m_{z+}}")(s, p, t),
    M_zm = sympy.Function(r"\widetilde{m_{z-}}")(s, p, t),
    zM_1 = sympy.Function(r"\widetilde{zm_1}")(s, p, t),
    zM_p = sympy.Function(r"\widetilde{zm_+}")(s, p, t),
    zM_m = sympy.Function(r"\widetilde{zm_-}")(s, p, t),
    # Conjugate variables for magnetic fields in equatorial plane
    B_ep = sympy.Function(r"b_{e+}")(s, p, t),
    B_em = sympy.Function(r"b_{e-}")(s, p, t),
    Bz_e = pgvar_ptb.Bz_e,
    dB_dz_ep = sympy.Function(r"b_{e+, z}")(s, p, t),
    dB_dz_em = sympy.Function(r"b_{e-, z}")(s, p, t),
    # Magnetic field at the boundary
    Br_b = pgvar_ptb.Br_b,
    Bs_p = pgvar_ptb.Bs_p,
    Bp_p = pgvar_ptb.Bp_p,
    Bz_p = pgvar_ptb.Bz_p,
    Bs_m = pgvar_ptb.Bs_m,
    Bp_m = pgvar_ptb.Bp_m,
    Bz_m = pgvar_ptb.Bz_m
)


# Conversion between PG and conjugate quantities
def PG_to_conjugate(pg_comp: base.CollectionPG) -> base.CollectionConjugate:
    """Convert PG collection to conjugate counterparts.
    
    :param pg_comp: PG components to be converted
    :returns: base.CollectionConjugate object with conjugate quantities
    """
    # Decide how to form the conjugate object
    # The method assumes all entries are of the same type
    if isinstance(pg_comp.Psi, sympy.Expr):
        cg_comp = base.CollectionConjugate(
            Psi = pg_comp.Psi,
            # Moments: conversion
            M_1 = pg_comp.Mss + pg_comp.Mpp,
            M_p = pg_comp.Mss - pg_comp.Mpp + 2*sympy.I*pg_comp.Msp,
            M_m = pg_comp.Mss - pg_comp.Mpp - 2*sympy.I*pg_comp.Msp,
            M_zp = pg_comp.Msz + sympy.I*pg_comp.Mpz,
            M_zm = pg_comp.Msz - sympy.I*pg_comp.Mpz,
            zM_1 = pg_comp.zMss + pg_comp.zMpp,
            zM_p = pg_comp.zMss - pg_comp.zMpp + 2*sympy.I*pg_comp.zMsp,
            zM_m = pg_comp.zMss - pg_comp.zMpp - 2*sympy.I*pg_comp.zMsp,
            # Equatorial fields: conversion
            B_ep = pg_comp.Bs_e + sympy.I*pg_comp.Bp_e,
            B_em = pg_comp.Bs_e - sympy.I*pg_comp.Bp_e,
            Bz_e = pg_comp.Bz_e,
            dB_dz_ep = pg_comp.dBs_dz_e + sympy.I*pg_comp.dBp_dz_e,
            dB_dz_em = pg_comp.dBs_dz_e - sympy.I*pg_comp.dBp_dz_e,
            # Boundary: unchanged
            Br_b = pg_comp.Br_b,
            Bs_p = pg_comp.Bs_p,
            Bp_p = pg_comp.Bp_p,
            Bz_p = pg_comp.Bz_p,
            Bs_m = pg_comp.Bs_m,
            Bp_m = pg_comp.Bp_m,
            Bz_m = pg_comp.Bz_m,
        )
        return cg_comp
    elif isinstance(pg_comp.Psi, sympy.Eq):
        cg_lhs = PG_to_conjugate(pg_comp.apply(lambda eq: eq.lhs, inplace=False))
        cg_rhs = PG_to_conjugate(pg_comp.apply(lambda eq: eq.rhs, inplace=False))
        cg_comp = base.CollectionConjugate(
            **{fname: sympy.Eq(cg_lhs[fname], cg_rhs[fname]) 
               for fname in base.CollectionConjugate.cg_field_names})
        return cg_comp
    else:
        raise TypeError


def conjugate_to_PG(cg_comp: base.CollectionConjugate) -> base.CollectionPG:
    """Convert conjugate quantities to PG counterparts
    
    :param cg_comp: conjugate components to be converted
    :returns: base.CollectionPG object with PG quantities
    """
    if isinstance(cg_comp.Psi, sympy.Expr):
        pg_comp = base.CollectionPG(
            Psi = cg_comp.Psi,
            # Moments conversion
            Mss = (2*cg_comp.M_1 + cg_comp.M_p + cg_comp.M_m)/sympy.Integer(4),
            Mpp = (2*cg_comp.M_1 - cg_comp.M_p - cg_comp.M_m)/sympy.Integer(4),
            Msp = (cg_comp.M_p - cg_comp.M_m)/sympy.Integer(4)/sympy.I,
            Msz = (cg_comp.M_zp + cg_comp.M_zm)/sympy.Integer(2),
            Mpz = (cg_comp.M_zp - cg_comp.M_zm)/sympy.Integer(2)/sympy.I,
            zMss = (2*cg_comp.zM_1 + cg_comp.zM_p + cg_comp.zM_m)/sympy.Integer(4),
            zMpp = (2*cg_comp.zM_1 - cg_comp.zM_p - cg_comp.zM_m)/sympy.Integer(4),
            zMsp = (cg_comp.zM_p - cg_comp.zM_m)/sympy.Integer(4)/sympy.I,
            # Equatorial magnetic field conversion
            Bs_e = (cg_comp.B_ep + cg_comp.B_em)/sympy.Integer(2),
            Bp_e = (cg_comp.B_ep - cg_comp.B_em)/sympy.Integer(2)/sympy.I,
            Bz_e = cg_comp.Bz_e,
            dBs_dz_e = (cg_comp.dB_dz_ep + cg_comp.dB_dz_em)/sympy.Integer(2),
            dBp_dz_e = (cg_comp.dB_dz_ep - cg_comp.dB_dz_em)/sympy.Integer(2)/sympy.I,
            # Boundary unchanged
            Br_b = cg_comp.Br_b,
            Bs_p = cg_comp.Bs_p,
            Bp_p = cg_comp.Bp_p,
            Bz_p = cg_comp.Bz_p,
            Bs_m = cg_comp.Bs_m,
            Bp_m = cg_comp.Bp_m,
            Bz_m = cg_comp.Bz_m,
        )
        return pg_comp
    elif isinstance(cg_comp.Psi, sympy.Eq):
        cg_lhs = conjugate_to_PG(cg_comp.apply(lambda eq: eq.lhs, inplace=False))
        cg_rhs = conjugate_to_PG(cg_comp.apply(lambda eq: eq.rhs, inplace=False))
        pg_comp = base.CollectionPG(
            **{fname: sympy.Eq(cg_lhs[fname], cg_rhs[fname]) 
               for fname in base.CollectionPG.pg_field_names})
        return pg_comp
    else:
        raise TypeError


def map_pg_to_conjugate(pg_comp: base.CollectionPG, 
    cg_comp: base.CollectionConjugate) -> dict:
    """Build a dictionary that maps PG quantities to conjugates
    
    :param pg_comp: PG quantities collection
        each entry should be a symbol, or at least an expression
    :param cg_comp: conjugate quantities collection
        each entry should be a symbol, or at least an expression
    :returns: dict<PG quantity, expression in conjugates>
    """
    pg_expr = conjugate_to_PG(cg_comp)
    return base.map_collection(pg_comp, pg_expr)


def map_conjugate_to_pg(cg_comp: base.CollectionConjugate, 
    pg_comp: base.CollectionPG) -> dict:
    """Build a dictionary that maps conjugate quantities to PG quantities
    
    :param cg_comp: conjugate quantities collection
        each entry should be a symbol, or at least an expression
    :param pg_comp: PG quantities collection
        each entry should be a symbol, or at least an expression
    :returns: dict<conjugate quantity, expression in PG>
    """
    cg_expr = PG_to_conjugate(pg_comp)
    return base.map_collection(cg_comp, cg_expr)



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
