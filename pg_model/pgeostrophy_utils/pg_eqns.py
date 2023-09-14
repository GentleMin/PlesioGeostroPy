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


"""Linearized vorticity equation"""

force_perturbation = {
    fp_sym: eps*fp_sym,
    fs_sym: eps*fs_sym,
    fz_asym: eps*fz_asym,
    fe_p: eps*fe_p
}

vorticity_var_lin = linearize(vorticity_var, linearization_subs_map, perturb_var=eps)
vorticity_forcing_lin = linearize(vorticity_forcing, linearization_subs_map, force_perturbation, perturb_var=eps)


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


