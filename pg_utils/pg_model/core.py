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



# ================== Independent variables ====================

#: Spatial and time coordinates (Cartesian)
x, y, z, t = sympy.symbols("x, y, z, t", real=True)
#: Cylindrical and spherical coordinates
s, p, r, theta = sympy.symbols(r"s, \phi, r, \theta", positive=True)
#: Cylindrical coordinate system
cyl = v3d.CylindricalCoordinates(s, p, z)
#: Spherical coordinate system
sph = v3d.SphericalCoordinates(r, theta, p)

#: Half height as a function of polar radius (symbol)
H = sympy.Function("H")(s)
#: Half height as a function of polar radius (expr)
H_s = sympy.sqrt(1 - s**2)


# ============== PG-independent physical fields ===============

#: Vector magnetic field (cylindrical coordinates)
B_vec = v3d.Vector3D(
    [
        sympy.Function(r"B_s")(s, p, z, t),
        sympy.Function(r"B_\phi")(s, p, z, t),
        sympy.Function(r"B_z")(s, p, z, t)
    ], 
    coord_sys=cyl
)

#: Vector magnetic field (spherical coordinates)
B_sph = v3d.Vector3D(
    [
        sympy.Function(r"B_r")(r, theta, p, t),
        sympy.Function(r"B_\theta")(r, theta, p, t),
        sympy.Function(r"B_\phi")(r, theta, p, t)
    ], 
    coord_sys=sph
)

#: Vector velocity field (cylindrical coordinates)
U_vec = v3d.Vector3D(
    [
        sympy.Function(r"U_s")(s, p, z, t),
        sympy.Function(r"U_\phi")(s, p, z, t),
        sympy.Function(r"U_z")(s, p, z, t)
    ], 
    coord_sys=cyl
)

#: Equatorial velocity (cylindrical coordinates)
v_e = (U_vec.s, U_vec.p, 0)

#: Vector velocity field (spherical coordinates)
U_sph = v3d.Vector3D(
    [
        sympy.Function(r"U_r")(r, theta, p, t),
        sympy.Function(r"U_\theta")(r, theta, p, t),
        sympy.Function(r"U_\phi")(r, theta, p, t)
    ],
    coord_sys=sph
)


# =================== Complete fields in PG ===================

#: Collection of PG variables
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
    Bs_e = sympy.Function(r"B_{s}^e")(s, p, t),
    Bp_e = sympy.Function(r"B_{\phi}^e")(s, p, t),
    Bz_e = sympy.Function(r"B_{z}^e")(s, p, t),
    dBs_dz_e = sympy.Function(r"B_{s, z}^e")(s, p, t),
    dBp_dz_e = sympy.Function(r"B_{\phi, z}^e")(s, p, t),
    # Boundary magnetic fields
    Br_b = sympy.Function(r"B_{r1}")(theta, p, t),
    Bs_p = sympy.Function(r"B_s^+")(s, p, t),
    Bp_p = sympy.Function(r"B_\phi^+")(s, p, t),
    Bz_p = sympy.Function(r"B_z^+")(s, p, t),
    Bs_m = sympy.Function(r"B_s^-")(s, p, t),
    Bp_m = sympy.Function(r"B_\phi^-")(s, p, t),
    Bz_m = sympy.Function(r"B_z^-")(s, p, t)
)

#: Quasi-geostrophic velocity ansatz
U_pg = v3d.Vector3D(
    [
        1/(s*H)*diff(pgvar.Psi, p),
        -1/H*diff(pgvar.Psi, s),
        z/(s*H**2)*diff(H, s)*diff(pgvar.Psi, p)
    ], 
    coord_sys=cyl
)



# ===================== Background fields =====================

#: Background magnetic field (cylindrical)
B0_vec = v3d.Vector3D(
    [
        sympy.Function(r"B_s^0")(s, p, z),
        sympy.Function(r"B_\phi^0")(s, p, z),
        sympy.Function(r"B_z^0")(s, p, z)
    ], 
    coord_sys=cyl
)
#: Background magnetic field (spherical)
B0_sph = v3d.Vector3D(
    [
        sympy.Function(r"B_r^0")(r, theta, p),
        sympy.Function(r"B_\theta^0")(r, theta, p),
        sympy.Function(r"B_\phi^0")(r, theta, p)
    ], 
    coord_sys=sph
)
#: Background velocity field (cylindrical)
U0_vec = v3d.Vector3D(
    [
        sympy.Function(r"U_s^0")(s, p, z),
        sympy.Function(r"U_\phi^0")(s, p, z),
        sympy.Function(r"U_z^0")(s, p, z)
    ], 
    coord_sys=cyl
)
#: Background velocity field (spherical)
U0_sph = v3d.Vector3D(
    [
        sympy.Function(r"U_r^0")(r, theta, p),
        sympy.Function(r"U_\theta^0")(r, theta, p),
        sympy.Function(r"U_\phi^0")(r, theta, p)
    ],
    coord_sys=sph
)

#: Collection of background PG fields
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
    Bs_e = sympy.Function(r"B_{s}^{0e}")(s, p),
    Bp_e = sympy.Function(r"B_{\phi}^{0e}")(s, p),
    Bz_e = sympy.Function(r"B_{z}^{0e}")(s, p),
    dBs_dz_e = sympy.Function(r"B_{s, z}^{0e}")(s, p),
    dBp_dz_e = sympy.Function(r"B_{\phi, z}^{0e}")(s, p),
    # Boundary magnetic fields
    Br_b = sympy.Function(r"B_{r1}^0")(theta, p),
    Bs_p = sympy.Function(r"B_s^{0+}")(s, p),
    Bp_p = sympy.Function(r"B_\phi^{0+}")(s, p),
    Bz_p = sympy.Function(r"B_z^{0+}")(s, p),
    Bs_m = sympy.Function(r"B_s^{0-}")(s, p),
    Bp_m = sympy.Function(r"B_\phi^{0-}")(s, p),
    Bz_m = sympy.Function(r"B_z^{0-}")(s, p)
)



# ==================== Perturbation fields ====================

#: Perturbation of magnetic field (cylindrical)
b_vec = v3d.Vector3D(
    [
        sympy.Function(r"b_s")(s, p, z, t),
        sympy.Function(r"b_\phi")(s, p, z, t),
        sympy.Function(r"b_z")(s, p, z, t)
    ], 
    coord_sys=cyl
)
#: Perturbation of magnetic field (spherical)
b_sph = v3d.Vector3D(
    [
        sympy.Function(r"b_r")(r, theta, p, t),
        sympy.Function(r"b_\theta")(r, theta, p, t),
        sympy.Function(r"b_\phi")(r, theta, p, t)
    ], 
    coord_sys=sph
)
#: Perturbation of velocity field (cylindrical)
u_vec = v3d.Vector3D(
    [
        sympy.Function(r"u_s")(s, p, z, t),
        sympy.Function(r"u_\phi")(s, p, z, t),
        sympy.Function(r"u_z")(s, p, z, t)
    ], 
    coord_sys=cyl
)
#: Perturbation of magnetic field (spherical)
u_sph = v3d.Vector3D(
    [
        sympy.Function(r"u_r")(r, theta, p, t),
        sympy.Function(r"u_\theta")(r, theta, p, t),
        sympy.Function(r"u_\phi")(r, theta, p, t)
    ],
    coord_sys=sph
)

#: Perturbation of PG fields
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
    Bs_e = sympy.Function(r"b_{s}^e")(s, p, t),
    Bp_e = sympy.Function(r"b_{\phi}^e")(s, p, t),
    Bz_e = sympy.Function(r"b_{z}^e")(s, p, t),
    dBs_dz_e = sympy.Function(r"b_{s, z}^e")(s, p, t),
    dBp_dz_e = sympy.Function(r"b_{\phi, z}^e")(s, p, t),
    # Magnetic field at the boundary
    Br_b = sympy.Function(r"b_{r1}")(theta, p, t),
    Bs_p = sympy.Function(r"b_s^+")(s, p, t),
    Bp_p = sympy.Function(r"b_\phi^+")(s, p, t),
    Bz_p = sympy.Function(r"b_z^+")(s, p, t),
    Bs_m = sympy.Function(r"b_s^-")(s, p, t),
    Bp_m = sympy.Function(r"b_\phi^-")(s, p, t),
    Bz_m = sympy.Function(r"b_z^-")(s, p, t)
)



# =================== Conjugate variables =====================

#: Collection of conjugate variables
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
    B_ep = sympy.Function(r"B_{+}^e")(s, p, t),
    B_em = sympy.Function(r"B_{-}^e")(s, p, t),
    Bz_e = pgvar.Bz_e,
    dB_dz_ep = sympy.Function(r"B_{+, z}^e")(s, p, t),
    dB_dz_em = sympy.Function(r"B_{-, z}^e")(s, p, t),
    # Magnetic field at the boundary
    Br_b = pgvar.Br_b,
    B_pp = sympy.Function(r"B_+^+")(s, p, t),
    B_pm = sympy.Function(r"B_-^+")(s, p, t),
    Bz_p = pgvar.Bz_p,
    B_mp = sympy.Function(r"B_+^-")(s, p, t),
    B_mm = sympy.Function(r"B_-^-")(s, p, t),
    Bz_m = pgvar.Bz_m
)

#: Collection of background conjugate quantities
cgvar_bg = base.CollectionConjugate(
    # Stream function, unchanged
    Psi = pgvar.Psi,
    # Conjugate variables for magnetic moments
    M_1 = sympy.Function(r"\overline{M_1}^0")(s, p, t),
    M_p = sympy.Function(r"\overline{M_+}^0")(s, p, t),
    M_m = sympy.Function(r"\overline{M_-}^0")(s, p, t),
    M_zp = sympy.Function(r"\widetilde{M_{z+}}^0")(s, p, t),
    M_zm = sympy.Function(r"\widetilde{M_{z-}}^0")(s, p, t),
    zM_1 = sympy.Function(r"\widetilde{zM_1}^0")(s, p, t),
    zM_p = sympy.Function(r"\widetilde{zM_+}^0")(s, p, t),
    zM_m = sympy.Function(r"\widetilde{zM_-}^0")(s, p, t),
    # Conjugate variables for magnetic fields in equatorial plane
    B_ep = sympy.Function(r"B_{+}^{0e}")(s, p, t),
    B_em = sympy.Function(r"B_{-}^{0e}")(s, p, t),
    Bz_e = pgvar.Bz_e,
    dB_dz_ep = sympy.Function(r"B_{+, z}^{0e}")(s, p, t),
    dB_dz_em = sympy.Function(r"B_{-, z}^{0e}")(s, p, t),
    # Magnetic field at the boundary
    Br_b = pgvar.Br_b,
    B_pp = sympy.Function(r"B_+^{0+}")(s, p, t),
    B_pm = sympy.Function(r"B_-^{0+}")(s, p, t),
    Bz_p = pgvar.Bz_p,
    B_mp = sympy.Function(r"B_+^{0-}")(s, p, t),
    B_mm = sympy.Function(r"B_-^{0-}")(s, p, t),
    Bz_m = pgvar.Bz_m
)

#: Perturbation in conjugate fields
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
    B_ep = sympy.Function(r"b_{+}^e")(s, p, t),
    B_em = sympy.Function(r"b_{-}^e")(s, p, t),
    Bz_e = pgvar_ptb.Bz_e,
    dB_dz_ep = sympy.Function(r"b_{+, z}^e")(s, p, t),
    dB_dz_em = sympy.Function(r"b_{-, z}^e")(s, p, t),
    # Magnetic field at the boundary
    Br_b = pgvar_ptb.Br_b,
    B_pp = sympy.Function(r"b_+^+")(s, p, t),
    B_pm = sympy.Function(r"b_-^+")(s, p, t),
    Bz_p = pgvar_ptb.Bz_p,
    B_mp = sympy.Function(r"b_+^-")(s, p, t),
    B_mm = sympy.Function(r"b_-^-")(s, p, t),
    Bz_m = pgvar_ptb.Bz_m
)


# Conversion between PG and conjugate quantities
def PG_to_conjugate(pg_comp: base.CollectionPG) -> base.CollectionConjugate:
    """Convert PG collection to conjugate counterparts.
    
    :param base.CollectionPG pg_comp: PG components to be converted
    :returns: collection of conjugate quantities
    """
    # Decide how to form the conjugate object
    # The method assumes all entries are of the same type
    if isinstance(pg_comp.Mss, sympy.Expr):
        cg_comp = base.CollectionConjugate(
            Psi = pg_comp.Psi,
            # Moments: conversion
            M_1 = (pg_comp.Mss + pg_comp.Mpp)/2,
            M_p = (pg_comp.Mss - pg_comp.Mpp + 2*sympy.I*pg_comp.Msp)/2,
            M_m = (pg_comp.Mss - pg_comp.Mpp - 2*sympy.I*pg_comp.Msp)/2,
            M_zp = (pg_comp.Msz + sympy.I*pg_comp.Mpz)/sympy.sqrt(2),
            M_zm = (pg_comp.Msz - sympy.I*pg_comp.Mpz)/sympy.sqrt(2),
            zM_1 = (pg_comp.zMss + pg_comp.zMpp)/2,
            zM_p = (pg_comp.zMss - pg_comp.zMpp + 2*sympy.I*pg_comp.zMsp)/2,
            zM_m = (pg_comp.zMss - pg_comp.zMpp - 2*sympy.I*pg_comp.zMsp)/2,
            # Equatorial fields: conversion
            B_ep = (pg_comp.Bs_e + sympy.I*pg_comp.Bp_e)/sympy.sqrt(2),
            B_em = (pg_comp.Bs_e - sympy.I*pg_comp.Bp_e)/sympy.sqrt(2),
            Bz_e = pg_comp.Bz_e,
            dB_dz_ep = (pg_comp.dBs_dz_e + sympy.I*pg_comp.dBp_dz_e)/sympy.sqrt(2),
            dB_dz_em = (pg_comp.dBs_dz_e - sympy.I*pg_comp.dBp_dz_e)/sympy.sqrt(2),
            # Boundary: unchanged
            Br_b = pg_comp.Br_b,
            B_pp = (pg_comp.Bs_p + sympy.I*pg_comp.Bp_p)/sympy.sqrt(2),
            B_pm = (pg_comp.Bs_p - sympy.I*pg_comp.Bp_p)/sympy.sqrt(2),
            Bz_p = pg_comp.Bz_p,
            B_mp = (pg_comp.Bs_m + sympy.I*pg_comp.Bp_m)/sympy.sqrt(2),
            B_mm = (pg_comp.Bs_m - sympy.I*pg_comp.Bp_m)/sympy.sqrt(2),
            Bz_m = pg_comp.Bz_m,
        )
        return cg_comp
    elif isinstance(pg_comp.Mss, sympy.Eq):
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
    
    :param base.CollectionConjugate cg_comp: conjugate components to be converted
    :returns: collection object with PG quantities
    """
    if isinstance(cg_comp.Psi, sympy.Expr):
        pg_comp = base.CollectionPG(
            Psi = cg_comp.Psi,
            # Moments conversion
            Mss = (2*cg_comp.M_1 + cg_comp.M_p + cg_comp.M_m)/2,
            Mpp = (2*cg_comp.M_1 - cg_comp.M_p - cg_comp.M_m)/2,
            Msp = (cg_comp.M_p - cg_comp.M_m)/2/sympy.I,
            Msz = (cg_comp.M_zp + cg_comp.M_zm)/sympy.sqrt(2),
            Mpz = (cg_comp.M_zp - cg_comp.M_zm)/sympy.sqrt(2)/sympy.I,
            zMss = (2*cg_comp.zM_1 + cg_comp.zM_p + cg_comp.zM_m)/2,
            zMpp = (2*cg_comp.zM_1 - cg_comp.zM_p - cg_comp.zM_m)/2,
            zMsp = (cg_comp.zM_p - cg_comp.zM_m)/2/sympy.I,
            # Equatorial magnetic field conversion
            Bs_e = (cg_comp.B_ep + cg_comp.B_em)/sympy.sqrt(2),
            Bp_e = (cg_comp.B_ep - cg_comp.B_em)/sympy.sqrt(2)/sympy.I,
            Bz_e = cg_comp.Bz_e,
            dBs_dz_e = (cg_comp.dB_dz_ep + cg_comp.dB_dz_em)/sympy.sqrt(2),
            dBp_dz_e = (cg_comp.dB_dz_ep - cg_comp.dB_dz_em)/sympy.sqrt(2)/sympy.I,
            # Boundary unchanged
            Br_b = cg_comp.Br_b,
            Bs_p = (cg_comp.B_pp + cg_comp.B_pm)/sympy.sqrt(2),
            Bp_p = (cg_comp.B_pp - cg_comp.B_pm)/sympy.sqrt(2)/sympy.I,
            Bz_p = cg_comp.Bz_p,
            Bs_m = (cg_comp.B_mp + cg_comp.B_mm)/sympy.sqrt(2),
            Bp_m = (cg_comp.B_mp - cg_comp.B_mm)/sympy.sqrt(2)/sympy.I,
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
    """Build a dictionary that maps PG quantities to their conjugates
    
    :param base.CollectionPG pg_comp: PG quantities collection
        each entry should be a symbol, or at least an expression
    :param base.CollectionConjugate cg_comp: conjugate quantities collection
        each entry should be a symbol, or at least an expression
    :returns: dict<PG quantity, expression in conjugate>
    """
    pg_expr = conjugate_to_PG(cg_comp)
    return base.map_collection(pg_comp, pg_expr)


def map_conjugate_to_pg(cg_comp: base.CollectionConjugate, 
    pg_comp: base.CollectionPG) -> dict:
    """Build a dictionary that maps conjugate quantities to PG quantities
    
    :param base.CollectionConjugate cg_comp: conjugate quantities collection
        each entry should be a symbol, or at least an expression
    :param base.CollectionPG pg_comp: PG quantities collection
        each entry should be a symbol, or at least an expression
    :returns: dict<conjugate quantity, expression in PG>
    """
    cg_expr = PG_to_conjugate(pg_comp)
    return base.map_collection(cg_comp, cg_expr)



# ====================== Reduced system =======================

#: Reduced system variables
reduced_var = base.LabeledCollection(
    ["Psi", "F_ext"], 
    Psi = pgvar_ptb.Psi, 
    F_ext = sympy.Function(r"F_\mathrm{ext}")(s, p, t)
)


# ==================== Force placeholders =====================

#: Symmetric integral of radial force
fs_sym = sympy.Function(r"\overline{f_s}")(s, p, t)
#: Symmetric integral of azimuthal force
fp_sym = sympy.Function(r"\overline{f_\phi}")(s, p, t)
#: Anti-symmetric integral of axial force
fz_asym = sympy.Function(r"\widetilde{f_z}")(s, p, t)
#: Azimuthal force in the equatorial plane
fe_p = sympy.Function(r"f_{\phi}^e")(s, p, t)


# ================= Linearization utilities ===================

#: Small quantity for linearization :math:`\epsilon`
eps = sympy.Symbol(r"\epsilon")

#: First-order perturbation map of velocity field
u_linmap = {U_vec[idx]: U0_vec[idx] + eps*U_pg[idx].subs(pgvar.Psi, pgvar_ptb.Psi)
    for idx in range(U_vec.ndim)}
u_linmap.update({U_sph[idx]: U0_sph[idx] + eps*u_sph[idx] 
    for idx in range(U_sph.ndim)})
#: First-order perturbation map of magnetic field
b_linmap = {B_vec[idx]: B0_vec[idx] + eps*b_vec[idx] 
    for idx in range(B_vec.ndim)}
b_linmap.update({B_sph[idx]: B0_sph[idx] + eps*b_sph[idx] 
    for idx in range(B_sph.ndim)})
#: First-order perturbation map of PG variables
pg_linmap = {pgvar[idx]: pgvar_bg[idx] + eps*pgvar_ptb[idx] 
    for idx in range(pgvar.n_fields)}
#: First-order perturbation map of conjugate variables
cg_linmap = {cgvar[idx]: cgvar_bg[idx] + eps*cgvar_ptb[idx]
    for idx in range(cgvar.n_fields)}

#: First-order perturbation in external forcing
force_linmap = {
    fp_sym: eps*fp_sym,
    fs_sym: eps*fs_sym,
    fz_asym: eps*fz_asym,
    fe_p: eps*fe_p
}

def pg_ansatz(expr, formal_vel=U_vec, PG_Psi=pgvar.Psi):
    """Replace formal velocity with PG ansatz
    """
    expr_psi = expr.subs({
        formal_vel[0]: U_pg[0].subs(pgvar.Psi, PG_Psi),
        formal_vel[1]: U_pg[1].subs(pgvar.Psi, PG_Psi),
        formal_vel[2]: U_pg[2].subs(pgvar.Psi, PG_Psi)
    })
    return expr_psi
