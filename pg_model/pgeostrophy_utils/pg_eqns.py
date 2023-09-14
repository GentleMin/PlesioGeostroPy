# -*- coding: utf-8 -*-
"""
Equations for Plesio-Geostrophy Model
Jingtao Min @ ETH-EPM, 09.2023
"""

import sympy
from sympy import diff
from sympy import Derivative as diff_u
from .pg_fields import *


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

vorticity_var_perturbed = vorticity_var.subs(linearization_subs_map)
vorticity_var_lin = vorticity_var_perturbed.simplify().expand().coeff(eps, 1)

vorticity_forcing_perturbed = vorticity_forcing.subs(linearization_subs_map)
vorticity_forcing_perturbed = vorticity_forcing_perturbed.subs({
    fp_sym: eps*fp_sym,
    fs_sym: eps*fs_sym,
    fz_asym: eps*fz_asym,
    fe_p: eps*fe_p
})
vorticity_forcing_lin = sympy.collect(vorticity_forcing_perturbed.simplify().expand(), eps).coeff(eps, 1)


"""Linearized induction equation"""
# The induction term further requires perturbation in velocity
velocity_map = {us: us_0 + eps*us_psi, up: up_0 + eps*up_psi, uz: uz_0 + eps*uz_psi}

evo_mss = evo_Mss.subs(velocity_map).subs(linearization_subs_map)
evo_mss = evo_mss.simplify().expand().coeff(eps, 1)

evo_mpp = evo_Mpp.subs(velocity_map).subs(linearization_subs_map)
evo_mpp = evo_mpp.simplify().expand().coeff(eps, 1)

evo_msp = evo_Msp.subs(velocity_map).subs(linearization_subs_map)
evo_msp = evo_msp.simplify().expand().coeff(eps, 1)

evo_msz = evo_Msz.subs(velocity_map).subs(linearization_subs_map)
evo_msz = evo_msz.simplify().expand().coeff(eps, 1)

evo_mpz = evo_Mpz.subs(velocity_map).subs(linearization_subs_map)
evo_mpz = evo_mpz.simplify().expand().coeff(eps, 1)

evo_zmss = evo_zMss.subs(velocity_map).subs(linearization_subs_map)
evo_zmss = evo_zmss.simplify().expand().coeff(eps, 1)

evo_zmpp = evo_zMpp.subs(velocity_map).subs(linearization_subs_map)
evo_zmpp = evo_zmpp.simplify().expand().coeff(eps, 1)

evo_zmsp = evo_zMsp.subs(velocity_map).subs(linearization_subs_map)
evo_zmsp = evo_zmsp.simplify().expand().coeff(eps, 1)

evo_bs_e = evo_Bs_e.subs(velocity_map).subs(linearization_subs_map)
evo_bs_e = evo_bs_e.simplify().expand().coeff(eps, 1)

evo_bp_e = evo_Bp_e.subs(velocity_map).subs(linearization_subs_map)
evo_bp_e = evo_bp_e.simplify().expand().coeff(eps, 1)

evo_bz_e = evo_Bz_e.subs(velocity_map).subs(linearization_subs_map)
evo_bz_e = evo_bz_e.simplify().expand().coeff(eps, 1)

evo_dbs_dz_e = evo_dBs_dz_e.subs(velocity_map).subs(linearization_subs_map)
evo_dbs_dz_e = evo_dbs_dz_e.simplify().expand().coeff(eps, 1)

evo_dbp_dz_e = evo_dBp_dz_e.subs(velocity_map).subs(linearization_subs_map)
evo_dbp_dz_e = evo_dbp_dz_e.simplify().expand().coeff(eps, 1)



