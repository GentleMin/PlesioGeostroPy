# -*- coding: utf-8 -*-
"""
Equations for Plesio-Geostrophy Model
Jingtao Min @ ETH-EPM, 09.2023
"""

import sympy
from sympy import diff
from sympy import Derivative as diff_u
from . import base
from .base import CollectionPG, CollectionConjugate
from .core import *
from .base_utils import linearize



"""Collecting equations"""

eqs_pg = CollectionPG()

"""Vorticity equation"""
eqs_pg.Psi = sympy.Eq(
    diff_u(s/H*diff(pgvar.Psi, t, s), s)
    + (1/(s*H) - 1/(2*H**2)*diff(H, s))*diff(pgvar.Psi, t, (p, 2)), 
    -2/H**2*diff(H, s)*diff(pgvar.Psi, p)
    + diff(H, s)*(s/H*fe_p + 1/(2*H)*diff(fz_asym, p)) 
    - s/(2*H)*cyl.curl((fs_sym, fp_sym, 0))[2])

"""Induction equations for magnetic moments"""
eqs_pg.Mss = sympy.Eq(
    diff_u(pgvar.Mss, t), 
    - H*v3d.dot(v_e, cyl.grad(pgvar.Mss/H, evaluate=False))
    + 2*diff_u(U_vec.s, s)*pgvar.Mss
    + 2/s*diff_u(U_vec.s, p)*pgvar.Msp)

eqs_pg.Mpp = sympy.Eq(
    diff_u(pgvar.Mpp, t),
    - 1/H*v3d.dot(v_e, cyl.grad(H*pgvar.Mpp, evaluate=False))
    - 2*diff_u(U_vec.s, s)*pgvar.Mpp
    + 2*s*diff_u(U_vec.p/s, s)*pgvar.Msp)

eqs_pg.Msp = sympy.Eq(
    diff_u(pgvar.Msp, t), 
    - v3d.dot(v_e, cyl.grad(pgvar.Msp, evaluate=False))
    + s*diff_u(U_vec.p/s, s)*pgvar.Mss
    + 1/s*diff_u(U_vec.s, p)*pgvar.Mpp)

eqs_pg.Msz = sympy.Eq(
    diff_u(pgvar.Msz, t), 
    - v3d.dot(v_e, cyl.grad(pgvar.Msz, evaluate=False))
    + (diff_u(U_vec.s, s) + 2*diff_u(U_vec.z, z))*pgvar.Msz
    + 1/s*diff_u(U_vec.s, p)*pgvar.Mpz
    + diff_u(U_vec.s/H*diff_u(H, s), s)*pgvar.zMss
    + 1/(s*H)*diff_u(H, s)*diff_u(U_vec.s, p)*pgvar.zMsp)

eqs_pg.Mpz = sympy.Eq(
    diff_u(pgvar.Mpz, t), 
    - v3d.dot(v_e, cyl.grad(pgvar.Mpz, evaluate=False))
    + (diff(U_vec.z, z) - diff(U_vec.s, s))*pgvar.Mpz
    + s*diff_u(U_vec.p/s, s)*pgvar.Msz
    + diff_u(U_vec.s/H*diff(H, s), s)*pgvar.zMsp
    + 1/(s*H)*diff(H, s)*diff(U_vec.s, p)*pgvar.zMpp)

eqs_pg.zMss = sympy.Eq(
    diff_u(pgvar.zMss, t), 
    - v3d.dot(v_e, cyl.grad(pgvar.zMss, evaluate=False))
    + 2*(diff(U_vec.s, s) + diff(U_vec.z, z))*pgvar.zMss
    + 2/s*diff(U_vec.s, p)*pgvar.zMsp)

eqs_pg.zMpp = sympy.Eq(
    diff_u(pgvar.zMpp, t), 
    - v3d.dot(v_e, cyl.grad(pgvar.zMpp, evaluate=False))
    - 2*diff(U_vec.s, s)*pgvar.zMpp
    + 2*s*diff_u(U_vec.p/s, s)*pgvar.zMsp)

eqs_pg.zMsp = sympy.Eq(
    diff_u(pgvar.zMsp, t), 
    - v3d.dot(v_e, cyl.grad(pgvar.zMsp, evaluate=False))
    + diff(U_vec.z, z)*pgvar.zMsp
    + s*diff_u(U_vec.p/s, s)*pgvar.zMss
    + 1/s*diff(U_vec.s, p)*pgvar.zMpp)

"""Induction equation for magnetic field in the equatorial plane"""
eqs_pg.Bs_e = sympy.Eq(
    diff(pgvar.Bs_e, t),
    + pgvar.Bs_e*diff(U_vec.s, s) + 1/s*pgvar.Bp_e*diff(U_vec.s, p)
    - U_vec.s*diff(pgvar.Bs_e, s) - 1/s*U_vec.p*diff(pgvar.Bs_e, p))

eqs_pg.Bp_e = sympy.Eq(
    diff(pgvar.Bp_e, t), 
    + pgvar.Bs_e*diff(U_vec.p, s) + 1/s*pgvar.Bp_e*diff(U_vec.p, p)
    - U_vec.s*diff(pgvar.Bp_e, s) - 1/s*U_vec.p*diff(pgvar.Bp_e, p)
    + (pgvar.Bp_e*U_vec.s - U_vec.p*pgvar.Bs_e)/s)

eqs_pg.Bz_e = sympy.Eq(
    diff(pgvar.Bz_e, t), 
    - U_vec.s*diff(pgvar.Bz_e, s) - 1/s*U_vec.p*diff(pgvar.Bz_e, p)
    + diff(U_vec.z, z)*pgvar.Bz_e)

eqs_pg.dBs_dz_e = sympy.Eq(
    diff(pgvar.dBs_dz_e, t),
    + pgvar.dBs_dz_e*diff(U_vec.s, s) + 1/s*pgvar.dBp_dz_e*diff(U_vec.s, p)
    - U_vec.s*diff(pgvar.dBs_dz_e, s) - 1/s*U_vec.p*diff(pgvar.dBs_dz_e, p)
    - diff(U_vec.z, z)*pgvar.dBs_dz_e)

eqs_pg.dBp_dz_e = sympy.Eq(
    diff(pgvar.dBp_dz_e, t),
    + pgvar.dBs_dz_e*diff(U_vec.p, s) + 1/s*pgvar.dBp_dz_e*diff(U_vec.p, p)
    - U_vec.s*diff(pgvar.dBp_dz_e, s) - 1/s*U_vec.p*diff(pgvar.dBp_dz_e, p)
    + (pgvar.dBp_dz_e*U_vec.s - U_vec.p*pgvar.dBs_dz_e)/s
    - diff(U_vec.z, z)*pgvar.dBp_dz_e)

"""Induction: boundary stirring"""
# In non-linearized form, boundary induction equation 
# must be written in Br to be closed
eqs_pg.Br_b = sympy.Eq(
    diff(pgvar.Br_b, t),
    -sph.surface_div((pgvar.Br_b*U_sph.t, pgvar.Br_b*U_sph.p), evaluate=False)
)

# Boundary induction in cylindrical coordinates 
# involves magnetic fields in the volume, and is not closed at the surface
eqs_pg.Bs_p = sympy.Eq(
    diff(pgvar.Bs_p, t),
    + pgvar.Bs_p*diff(U_vec.s, s) - U_vec.s*diff(B_vec.s, s)
    + pgvar.Bp_p/s*diff(U_vec.s, p) - U_vec.p/s*diff(B_vec.s, p)
    + pgvar.Bz_p*diff(U_vec.s, z)  - U_vec.z*diff(B_vec.s, z))

eqs_pg.Bp_p = sympy.Eq(
    diff(pgvar.Bp_p, t),
    + pgvar.Bs_p*diff(U_vec.p, s) - U_vec.s*diff(B_vec.p, s) 
    + pgvar.Bp_p/s*diff(U_vec.p, p) - U_vec.p/s*diff(B_vec.p, p) 
    + pgvar.Bz_p*diff(U_vec.p, z) - U_vec.z*diff(B_vec.p, z)
    + (pgvar.Bp_p*U_vec.s - U_vec.p*pgvar.Bs_p)/s)

eqs_pg.Bz_p = sympy.Eq(
    diff(pgvar.Bz_p, t),
    + pgvar.Bs_p*diff(U_vec.z, s) - U_vec.s*diff(B_vec.z, s)  
    + pgvar.Bp_p/s*diff(U_vec.z, p) - U_vec.p/s*diff(B_vec.z, p) 
    + pgvar.Bz_p*diff(U_vec.z, z)- U_vec.z*diff(B_vec.z, z))

eqs_pg.Bs_m = sympy.Eq(
    diff(pgvar.Bs_m, t),
    + pgvar.Bs_m*diff(U_vec.s, s) - U_vec.s*diff(B_vec.s, s)
    + pgvar.Bp_m/s*diff(U_vec.s, p) - U_vec.p/s*diff(B_vec.s, p)
    + pgvar.Bz_m*diff(U_vec.s, z)  - U_vec.z*diff(B_vec.s, z))

eqs_pg.Bp_m = sympy.Eq(
    diff(pgvar.Bp_m, t),
    + pgvar.Bs_m*diff(U_vec.p, s) - U_vec.s*diff(B_vec.p, s) 
    + pgvar.Bp_m/s*diff(U_vec.p, p) - U_vec.p/s*diff(B_vec.p, p) 
    + pgvar.Bz_m*diff(U_vec.p, z) - U_vec.z*diff(B_vec.p, z)
    + (pgvar.Bp_m*U_vec.s - U_vec.p*pgvar.Bs_m)/s)

eqs_pg.Bz_m = sympy.Eq(
    diff(pgvar.Bz_m, t),
    + pgvar.Bs_m*diff(U_vec.z, s) - U_vec.s*diff(B_vec.z, s)  
    + pgvar.Bp_m/s*diff(U_vec.z, p) - U_vec.p/s*diff(B_vec.z, p) 
    + pgvar.Bz_m*diff(U_vec.z, z)- U_vec.z*diff(B_vec.z, z))


"""Conjugate equations"""

def eqn_PG_to_conjugate(eqset_pg: CollectionPG, subs_map: dict):
    eqset_pg = eqset_pg.apply(lambda eq: eq.subs(subs_map), inplace=False)
    eqset_cg = PG_to_conjugate(eqset_pg)
    return eqset_cg

pg_cg_subs = map_pg_to_conjugate(pgvar, cgvar)
eqs_cg = eqn_PG_to_conjugate(eqs_pg, pg_cg_subs).apply(
    lambda eq: eq.doit().expand(), inplace=True)


"""Linearized equations"""

# Linearize equation
eqs_pg_lin = CollectionPG()
for idx, eq_tmp in enumerate(eqs_pg):
    eqs_pg_lin[idx] = sympy.Eq(
        linearize(eq_tmp.lhs, pg_linmap),
        linearize(eq_tmp.rhs, u_linmap, b_linmap, pg_linmap, force_linmap))

eqs_cg_lin = CollectionConjugate()
for idx, eq_tmp in enumerate(eqs_cg):
    eqs_cg_lin[idx] = sympy.Eq(
        linearize(eq_tmp.lhs, cg_linmap), 
        linearize(eq_tmp.rhs, u_linmap, b_linmap, cg_linmap, force_linmap))

