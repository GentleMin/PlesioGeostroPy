# -*- coding: utf-8 -*-
"""
Forcings in Plesio-Geostrophy Model

This module defines the available forces, 
which can be substituted into the placeholder
variables of forces in the vorticity equation

Jingtao Min @ ETH-EPM, 09.2023
"""

import sympy
from sympy import diff
from sympy import Derivative as diff_u
from .core import *
from .base_utils import linearize
from .base import CollectionPG, CollectionConjugate


# ===================== Coriolis force ========================

f_Cor = sympy.Function(r"f_{Cor}")(s, p, t)
f_Cor_expr = -2/H**2*diff(H, s)*diff(pgvar.Psi, p)


# ====================== Lorentz force ========================

#: Placeholder: symmetric integral of the radial Lorentz force
Ls_sym = sympy.Function(r"\overline{L_s}")(s, p, t)

#: Placeholder: symmetric integral of the azimuthal Lorentz force
Lp_sym = sympy.Function(r"\overline{L_\phi}")(s, p, t)

#: Placeholder: symmetric integral of z times the axial Lorentz force
zLz_sym = sympy.Function(r"\overline{zL_z}")(s, p, t)

# #: Placeholder: antisymmetric integral of the axial Lorentz force
# Lz_asym = sympy.Function(r"\widetilde{L_z}")(s, p, t)

# #: Placeholder: the azimuthal Lorentz force in the equatorial plane
# Le_p = sympy.Function(r"L_{\phi}^e")(s, p, t)

# Explicit expressions for Lorentz force

#: Expression: symmetric integral of the radial Lorentz force
Ls_sym_expr = 1/s*diff(s*pgvar.Mss, s) + 1/s*diff(pgvar.Msp, p) - pgvar.Mpp/s \
    + (pgvar.Bs_p*pgvar.Bz_p - pgvar.Bs_m*pgvar.Bz_m) \
    + s/H*(pgvar.Bs_p*pgvar.Bs_p + pgvar.Bs_m*pgvar.Bs_m)

#: Expression: symmetric integral of the azimuthal Lorentz force
Lp_sym_expr = 1/s*diff(s*pgvar.Msp, s) + 1/s*diff(pgvar.Mpp, p) + pgvar.Msp/s \
    + (pgvar.Bp_p*pgvar.Bz_p - pgvar.Bp_m*pgvar.Bz_m) \
    + s/H*(pgvar.Bs_p*pgvar.Bp_p + pgvar.Bs_m*pgvar.Bp_m)

#: Expression: antisymmetric integral of the axial Lorentz force
zLz_sym_expr = 1/s*diff(s*pgvar.zMsz, s) + 1/s*diff(pgvar.zMpz, p) - pgvar.Mzz \
    + H*(pgvar.Bz_p*pgvar.Bz_p + pgvar.Bz_m*pgvar.Bz_m) \
    - H*diff(H, s)*(pgvar.Bs_p*pgvar.Bz_p - pgvar.Bs_m*pgvar.Bz_m)

# #: Expression: antisymmetric integral of the axial Lorentz force
# Lz_asym_expr = 1/s*diff(s*pgvar.Msz, s) + 1/s*diff(pgvar.Mpz, p) \
#     + (pgvar.Bz_p*pgvar.Bz_p + pgvar.Bz_m*pgvar.Bz_m - 2*pgvar.Bz_e*pgvar.Bz_e) \
#     + s/H*(pgvar.Bs_p*pgvar.Bz_p - pgvar.Bs_m*pgvar.Bz_m)

# #: Expression: the azimuthal Lorentz force in the equatorial plane
# Le_p_expr = pgvar.Bs_e*diff(pgvar.Bp_e, s) + 1/s*pgvar.Bp_e*diff(pgvar.Bp_e, p) \
#     + pgvar.Bz_e*pgvar.dBp_dz_e + 1/s*pgvar.Bs_e*pgvar.Bp_e

# Convert to conjugate quantities
pg_cg_map = map_pg_to_conjugate(pgvar, cgvar)
#: Expression: symmetric integral of the radial Lorentz force in conjugate vars
Ls_sym_cg = Ls_sym_expr.subs(pg_cg_map)
#: Expression: symmetric integral of the azimuthal Lorentz force in conjugate vars
Lp_sym_cg = Lp_sym_expr.subs(pg_cg_map)
#: Expression: symmetric integral of the axial Lorentz force in conjugate vars
zLz_sym_cg = zLz_sym_expr.subs(pg_cg_map)

# #: Expression: the azimuthal Lorentz force in the equatorial plane in conjugate vars
# Le_p_cg = Le_p_expr.subs(pg_cg_map)

#: Mapping: placeholder symbols -> explicit exprs for PG vars
force_explicit = {
    Ls_sym: Ls_sym_expr,
    Lp_sym: Lp_sym_expr,
    zLz_sym: zLz_sym_expr,
    # Le_p: Le_p_expr
}

#: Mapping: placeholder symbols -> explicit exprs for conjugate vars
force_explicit_cg = {
    Ls_sym: Ls_sym_cg,
    Lp_sym: Lp_sym_cg,
    zLz_sym: zLz_sym_cg,
    # Le_p: Le_p_cg
}


# ================= Linearized Lorentz force ==================

# Linearized in terms of magnetic fields (for $L_{e\phi}$) 
# or in terms of magnetic moments (for integrated forces).

# $L_{e\phi}$ is quadratic in the magnetic field components in the equatorial plane.
# Linearized form involves cross terms 
# between background and perturbational magnetic fields.
# #: Linearized form of azimuthal Lorentz force in the equatorial plane
# Le_p_lin = linearize(Le_p_expr, pg_linmap, perturb_var=eps)

# For the integrated quantities, the Lorentz force IS a linear function 
# of magnetic moments. Essentially no linearization required.
# However, the boundary terms and the equatorial terms are quadratic 
# in magnetic fields. These terms need to be linearized.

#: Linearized form of :math:`\overline{L_s}`
Ls_sym_lin = linearize(Ls_sym_expr, pg_linmap, perturb_var=eps)

#: Linearized form of :math:`\overline{L_{\phi}}`
Lp_sym_lin = linearize(Lp_sym_expr, pg_linmap, perturb_var=eps)

#: Linearized form of :math:`\widetilde{L_z}`
zLz_sym_lin = linearize(zLz_sym_expr, pg_linmap, perturb_var=eps)

# Curl of horizontal components $\nabla \times \mathbf{L}_e$
# Curl is linear, linearize (curl (field)) = curl (linearize (field))
# curl_L = cyl.curl((Ls_sym_lin, Lp_sym_lin, 0))[2]

# Conjugate expressions

#: Linearized form of :math:`\overline{L_s}` in conjugate vars
Ls_sym_lin_cg = linearize(Ls_sym_cg, cg_linmap, perturb_var=eps)
#: Linearized form of :math:`\overline{L_{\phi}}` in conjugate vars
Lp_sym_lin_cg = linearize(Lp_sym_cg, cg_linmap, perturb_var=eps)
#: Linearized form of :math:`\widetilde{L_z}` in conjugate vars
zLz_sym_lin_cg = linearize(zLz_sym_cg, cg_linmap, perturb_var=eps)
# #: Linearized form of :math:`L_{\phi}^e` in conjugate vars
# Le_p_lin_cg = linearize(Le_p_cg, cg_linmap, perturb_var=eps)


#: Mapping: placeholder symbols -> linearized explicit exprs for PG vars
force_explicit_lin = {
    Ls_sym: Ls_sym_lin,
    Lp_sym: Lp_sym_lin,
    zLz_sym: zLz_sym_lin,
    # Le_p: Le_p_lin
}

#: Mapping: placeholder symbols -> linearized explicit exprs for conjugate vars
force_explicit_lin_cg = {
    Ls_sym: Ls_sym_lin_cg,
    Lp_sym: Lp_sym_lin_cg,
    zLz_sym: zLz_sym_lin_cg,
    # Le_p: Le_p_lin_cg
}


# =================== Magnetic diffusion ======================

Dm_models = dict()
Dm_models_lin = dict()
Dm_models_cg = dict()
Dm_models_cg_lin = dict()

# linear drag model -----------------------

mod_name = "linear drag"

Dm = CollectionPG(**{
    fname: -pgvar[fname] 
    for fname in CollectionPG.pg_field_names if fname != 'Psi'
})
Dm_cg = PG_to_conjugate(Dm)
Dm_cg.apply(
    lambda fname, expr: expr.subs(pg_cg_map).doit().expand() if fname != 'Psi' else expr, 
    inplace=True, metadata=True
)
Dm_lin = Dm.apply(
    lambda field, expr: linearize(expr, pg_linmap, perturb_var=eps) if field != 'Psi' else expr,
    inplace=False, metadata=True
)
Dm_cg_lin = Dm_cg.apply(
    lambda field, expr: linearize(expr, cg_linmap, perturb_var=eps) if field != 'Psi' else expr,
    inplace=False, metadata=True
)
   
Dm_models[mod_name] = Dm
Dm_models_lin[mod_name] = Dm_lin
Dm_models_cg[mod_name] = Dm_cg
Dm_models_cg_lin[mod_name] = Dm_cg_lin
