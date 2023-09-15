# -*- coding: utf-8 -*-
"""
Equations for Plesio-Geostrophy Model
Jingtao Min @ ETH-EPM, 09.2023
"""

import sympy
from sympy import diff
from sympy import Derivative as diff_u
from .pg_fields import *
from .base_utils import linearize

from types import SimpleNamespace


"""Container for equations"""
eqs_mechanics = SimpleNamespace()
eqs_induction = SimpleNamespace()
eqs_mechanics_lin = SimpleNamespace()
eqs_induction_lin = SimpleNamespace()

# Symbols for external forces
fs_sym = sympy.Function(r"\overline{f_s}")(s, p, t)
fp_sym = sympy.Function(r"\overline{f_\phi}")(s, p, t)
fz_asym = sympy.Function(r"\widetilde{f_z}")(s, p, t)
fe_p = sympy.Function(r"f_{e\phi}")(s, p, t)


"""Vorticity equation"""
# vorticity_var = -2*cyl_op.laplacian(diff(Psi, t)) + diff(H, s)*(2/H*diff(Psi, t, s) + 1/s/H*diff(Psi, t, (p, 2)))
# vorticity_forcing = diff(H, s)*4/s/H*diff(Psi, p) - diff(H, s)*(2*fe_p + 1/s*diff(fz_asym, p)) + cyl_op.curl((fs_sym, fp_sym, 0))[2]
# Self-adjoint form
vorticity_var = diff_u(s/H*diff(Psi, t, s), s) + (1/(s*H) - 1/(2*H**2)*diff(H, s))*diff(Psi, t, (p, 2))
vorticity_forcing = -2/H**2*diff(H, s)*diff(Psi, p) + diff(H, s)*(s/H*fe_p + 1/(2*H)*diff(fz_asym, p)) - s/(2*H)*cyl_op.curl((fs_sym, fp_sym, 0))[2]

eqs_mechanics.Psi = sympy.Eq(vorticity_var, vorticity_forcing)


"""Induction equation - the magnetic moments"""
v_e = (us, up, 0)

evo_Mss = cyl_op.grad(Mss/H, evaluate=False)
evo_Mss = -H*v3d.dot(v_e, evo_Mss) + 2*diff_u(us, s)*Mss + 2/s*diff_u(us, p)*Msp

evo_Mpp = cyl_op.grad(H*Mpp, evaluate=False)
evo_Mpp = -1/H*v3d.dot(v_e, evo_Mpp) - 2*diff_u(us, s)*Mpp + 2*s*diff_u(up/s, s)*Msp

evo_Msp = cyl_op.grad(Msp, evaluate=False)
evo_Msp = -v3d.dot(v_e, evo_Msp) + s*diff_u(up/s, s)*Mss + 1/s*diff_u(us, p)*Mpp

evo_Msz = cyl_op.grad(Msz, evaluate=False)
evo_Msz = -v3d.dot(v_e, evo_Msz) + (diff_u(us, s) + 2*diff_u(uz, z))*Msz + 1/s*diff_u(us, p)*Mpz + diff_u(us/H*diff_u(H, s), s)*zMss + 1/(s*H)*diff_u(H, s)*diff_u(us, p)*zMsp

evo_Mpz = cyl_op.grad(Mpz, evaluate=False)
evo_Mpz = -v3d.dot(v_e, evo_Mpz) + (diff(uz, z) - diff(us, s))*Mpz + s*diff_u(up/s, s)*Msz + diff_u(us/H*diff(H, s), s)*zMsp + 1/(s*H)*diff(H, s)*diff(us, p)*zMpp

evo_zMss = cyl_op.grad(zMss, evaluate=False)
evo_zMss = -v3d.dot(v_e, evo_zMss) + 2*(diff(us, s) + diff(uz, z))*zMss + 2/s*diff(us, p)*zMsp

evo_zMpp = cyl_op.grad(zMpp, evaluate=False)
evo_zMpp = -v3d.dot(v_e, evo_zMpp) - 2*diff(us, s)*zMpp + 2*s*diff_u(up/s, s)*zMsp

evo_zMsp = cyl_op.grad(zMsp, evaluate=False)
evo_zMsp = -v3d.dot(v_e, evo_zMsp) + diff(uz, z)*zMsp + s*diff_u(up/s, s)*zMss + 1/s*diff(us, p)*zMpp


"""Induction: magnetic field in the equatorial plane"""

evo_Bs_e = Bs_e*diff(us, s) + 1/s*Bp_e*diff(us, p) - us*diff(Bs_e, s) - 1/s*up*diff(Bs_e, p)
evo_Bp_e = Bs_e*diff(up, s) + 1/s*Bp_e*diff(up, p) - us*diff(Bp_e, s) - 1/s*up*diff(Bp_e, p) + (Bp_e*us - up*Bs_e)/s
evo_Bz_e = -us*diff(Bz_e, s) - 1/s*up*diff(Bz_e, p) + diff(uz, z)*Bz_e
evo_dBs_dz_e = dBs_dz_e*diff(us, s) + 1/s*dBp_dz_e*diff(us, p) - us*diff(dBs_dz_e, s) - 1/s*up*diff(dBs_dz_e, p) - diff(uz, z)*dBs_dz_e
evo_dBp_dz_e = dBs_dz_e*diff(up, s) + 1/s*dBp_dz_e*diff(up, p) - us*diff(dBp_dz_e, s) - 1/s*up*diff(dBp_dz_e, p) + (dBp_dz_e*us - up*dBs_dz_e)/s - diff(uz, z)*dBp_dz_e


"""Induction: boundary stirring"""

evo_Br = -sph_op.surface_div((Br*ut, Br*up_sph), evaluate=False)

# Collecting
eqs_induction.Mss = sympy.Eq(diff(Mss, t), evo_Mss)
eqs_induction.Mpp = sympy.Eq(diff(Mpp, t), evo_Mpp)
eqs_induction.Msp = sympy.Eq(diff(Msp, t), evo_Msp)
eqs_induction.Msz = sympy.Eq(diff(Msz, t), evo_Msz)
eqs_induction.Mpz = sympy.Eq(diff(Mpz, t), evo_Mpz)
eqs_induction.zMss = sympy.Eq(diff(zMss, t), evo_zMss)
eqs_induction.zMpp = sympy.Eq(diff(zMpp, t), evo_zMpp)
eqs_induction.zMsp = sympy.Eq(diff(zMsp, t), evo_zMsp)
eqs_induction.Bs_e = sympy.Eq(diff(Bs_e, t), evo_Bs_e)
eqs_induction.Bp_e = sympy.Eq(diff(Bp_e, t), evo_Bp_e)
eqs_induction.Bz_e = sympy.Eq(diff(Bz_e, t), evo_Bz_e)
eqs_induction.dBs_dz_e = sympy.Eq(diff(dBs_dz_e, t), evo_dBs_dz_e)
eqs_induction.dBp_dz_e = sympy.Eq(diff(dBp_dz_e, t), evo_dBp_dz_e)


"""Linearized vorticity equation"""

force_perturbation = {
    fp_sym: eps*fp_sym,
    fs_sym: eps*fs_sym,
    fz_asym: eps*fz_asym,
    fe_p: eps*fe_p
}

vorticity_var_lin = linearize(vorticity_var, linearization_subs_map, perturb_var=eps)
vorticity_forcing_lin = linearize(vorticity_forcing, linearization_subs_map, force_perturbation, perturb_var=eps)

eqs_mechanics_lin.psi = sympy.Eq(vorticity_var_lin, vorticity_forcing_lin)

"""Linearized induction equation"""
# The induction term further requires perturbation in velocity
velocity_map = {us: us_0 + eps*us_psi, up: up_0 + eps*up_psi, uz: uz_0 + eps*uz_psi}

evo_mss = linearize(evo_Mss, velocity_map, linearization_subs_map, perturb_var=eps)
evo_mpp = linearize(evo_Mpp, velocity_map, linearization_subs_map, perturb_var=eps)
evo_msp = linearize(evo_Msp, velocity_map, linearization_subs_map, perturb_var=eps)
evo_msz = linearize(evo_Msz, velocity_map, linearization_subs_map, perturb_var=eps)
evo_mpz = linearize(evo_Mpz, velocity_map, linearization_subs_map, perturb_var=eps)
evo_zmss = linearize(evo_zMss, velocity_map, linearization_subs_map, perturb_var=eps)
evo_zmpp = linearize(evo_zMpp, velocity_map, linearization_subs_map, perturb_var=eps)
evo_zmsp = linearize(evo_zMsp, velocity_map, linearization_subs_map, perturb_var=eps)
evo_bs_e = linearize(evo_Bs_e, velocity_map, linearization_subs_map, perturb_var=eps)
evo_bp_e = linearize(evo_Bp_e, velocity_map, linearization_subs_map, perturb_var=eps)
evo_bz_e = linearize(evo_Bz_e, velocity_map, linearization_subs_map, perturb_var=eps)
evo_dbs_dz_e = linearize(evo_dBs_dz_e, velocity_map, linearization_subs_map, perturb_var=eps)
evo_dbp_dz_e = linearize(evo_dBp_dz_e, velocity_map, linearization_subs_map, perturb_var=eps)


# Collection
eqs_induction_lin.mss = sympy.Eq(diff(mss, t), evo_mss)
eqs_induction_lin.mpp = sympy.Eq(diff(mpp, t), evo_mpp)
eqs_induction_lin.msp = sympy.Eq(diff(msp, t), evo_msp)
eqs_induction_lin.msz = sympy.Eq(diff(msz, t), evo_msz)
eqs_induction_lin.mpz = sympy.Eq(diff(mpz, t), evo_mpz)
eqs_induction_lin.zmss = sympy.Eq(diff(zmss, t), evo_zmss)
eqs_induction_lin.zmpp = sympy.Eq(diff(zmpp, t), evo_zmpp)
eqs_induction_lin.zmsp = sympy.Eq(diff(zmsp, t), evo_zmsp)
eqs_induction_lin.bs_e = sympy.Eq(diff(bs_e, t), evo_bs_e)
eqs_induction_lin.bp_e = sympy.Eq(diff(bp_e, t), evo_bp_e)
eqs_induction_lin.bz_e = sympy.Eq(diff(bz_e, t), evo_bz_e)
eqs_induction_lin.dbs_dz_e = sympy.Eq(diff(dbs_dz_e, t), evo_dbs_dz_e)
eqs_induction_lin.dbp_dz_e = sympy.Eq(diff(dbp_dz_e, t), evo_dbp_dz_e)
