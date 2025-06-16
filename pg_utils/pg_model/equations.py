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

beta = diff(H, s)/H

#: Plesio-Geostrophy equations
eqs_pg = CollectionPG()

# Vorticity equation
eqs_pg.Psi = sympy.Eq(
    diff_u(s/H*diff(pgvar.Psi, t, s), s)
    + (1/(s*H) - 1/(3*H**2)*diff(H, s))*diff(pgvar.Psi, t, (p, 2)), 
    # -2/H**2*diff(H, s)*diff(pgvar.Psi, p)
    # + diff(H, s)*(s/H*fe_p + 1/(2*H)*diff(fz_asym, p)) 
    - 2/H**2*diff(H, s)*diff(pgvar.Psi, p)
    - s/(2*H)*cyl.curl((fs_sym, fp_sym, 0))[2]
    - s**2/(2*H**3)*fp_sym
    - s/(2*H**3)*diff_u(zfz_sym, p)
)

# Induction equations for magnetic moments

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

eqs_pg.Mzz = sympy.Eq(
    diff_u(pgvar.Mzz, t),
    - v3d.dot(v_e, cyl.grad(pgvar.Mzz, evaluate=False))
    + 3/H*diff(H, s)*U_vec.s*pgvar.Mzz
    + 2*diff_u(diff(U_vec.z, z), s)*pgvar.zMsz
    + 2*diff(diff(U_vec.z, z), p)/s*pgvar.zMpz
)

eqs_pg.zMsz = sympy.Eq(
    diff_u(pgvar.zMsz, t), 
    - v3d.dot(v_e, cyl.grad(pgvar.zMsz, evaluate=False))
    + (diff_u(U_vec.s, s) + 3*diff(U_vec.z, z))*pgvar.zMsz
    + 1/s*diff(U_vec.s, p)*pgvar.zMpz
    + diff_u(U_vec.s/H*diff(H, s), s)*pgvar.z2Mss
    + 1/(s*H)*diff_u(H, s)*diff(U_vec.s, p)*pgvar.z2Msp
)

eqs_pg.zMpz = sympy.Eq(
    diff_u(pgvar.zMpz, t), 
    - v3d.dot(v_e, cyl.grad(pgvar.zMpz, evaluate=False))
    + (2*diff(U_vec.z, z) - diff_u(U_vec.s, s))*pgvar.zMpz
    + s*diff_u(U_vec.p/s, s)*pgvar.zMsz
    + diff_u(U_vec.s/H*diff(H, s), s)*pgvar.z2Msp
    + 1/(s*H)*diff(H, s)*diff(U_vec.s, p)*pgvar.z2Mpp
)

eqs_pg.z2Mss = sympy.Eq(
    diff_u(pgvar.z2Mss, t), 
    - v3d.dot(v_e, cyl.grad(pgvar.z2Mss, evaluate=False))
    + (2*diff_u(U_vec.s, s) + 3*diff(U_vec.z, z))*pgvar.z2Mss
    + 2/s*diff(U_vec.s, p)*pgvar.z2Msp
)

eqs_pg.z2Mpp = sympy.Eq(
    diff_u(pgvar.z2Mpp, t), 
    - v3d.dot(v_e, cyl.grad(pgvar.z2Mpp, evaluate=False))
    + (diff(U_vec.z, z) - 2*diff_u(U_vec.s, s))*pgvar.z2Mpp
    + 2*s*diff_u(U_vec.p/s, s)*pgvar.z2Msp
)

eqs_pg.z2Msp = sympy.Eq(
    diff_u(pgvar.z2Msp, t), 
    - v3d.dot(v_e, cyl.grad(pgvar.z2Msp, evaluate=False))
    + 2*diff(U_vec.z, z)*pgvar.z2Msp
    + s*diff_u(U_vec.p/s, s)*pgvar.z2Mss
    + 1/s*diff(U_vec.s, p)*pgvar.z2Mpp
)


# Induction: boundary stirring
# In non-linearized form, boundary induction equation 
# must be written in Br to be closed
eqs_pg.Br_b = sympy.Eq(
    diff(pgvar.Br_b, t),
    -sph.div_surface((pgvar.Br_b*U_sph.t, pgvar.Br_b*U_sph.p), evaluate=False)
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


# Conjugate equations

def eqn_PG_to_conjugate(eqset_pg: CollectionPG, subs_map: dict) -> CollectionConjugate:
    """Convert a set of PG equations to transformed form
    
    :param CollectionPG eqset_pg: set of PG equations;
    :param dict subs_map: substitution map that maps 
        PG variables to expressions in transformed variables
    :returns: set of transformed equations.
    """
    eqset_pg = eqset_pg.apply(lambda eq: eq.subs(subs_map), inplace=False)
    eqset_cg = PG_to_conjugate(eqset_pg)
    return eqset_cg

pg_cg_subs = map_pg_to_conjugate(pgvar, cgvar)

#: Conjugate variable equations
eqs_cg = eqn_PG_to_conjugate(eqs_pg, pg_cg_subs).apply(
    lambda eq: eq.doit().expand(), inplace=True)


# ================== Linearized equations =====================

#: Linearized PG equations
eqs_pg_lin = CollectionPG()
for idx, eq_tmp in enumerate(eqs_pg):
    eqs_pg_lin[idx] = sympy.Eq(
        linearize(eq_tmp.lhs, pg_linmap),
        linearize(eq_tmp.rhs, u_linmap, b_linmap, pg_linmap, force_linmap))

#: Linearized conjugate equations
eqs_cg_lin = CollectionConjugate()
for idx, eq_tmp in enumerate(eqs_cg):
    eqs_cg_lin[idx] = sympy.Eq(
        linearize(eq_tmp.lhs, cg_linmap), 
        linearize(eq_tmp.rhs, u_linmap, b_linmap, cg_linmap, force_linmap))

