# -*- coding: utf-8 -*-
"""Script for solving the eigenvalue problem in axisymmetric background field
"""


from pg_utils import eigen
import os, sympy
from pg_utils.pg_model import params, expansion, bg_fields
import sympy
output_dir = "./out/eigen/"


# bg_comps = [bg_fields.Background_S1(), bg_fields.Background_T2()]
# for bg_tmp in bg_comps:
#     bg_params = {bg_tmp.params[i_key]: bg_tmp.params_ref[0][i_key] for i_key in range(len(bg_tmp.params))}
#     bg_tmp.subs(bg_params, inplace=True)
# bg_active = bg_comps[0] + bg_comps[1]
bg_active = bg_fields.Background_S_l2_n1()

eigen.form_equations(
    eq_mode="pg",
    components=["Lorentz"],
    # diff_M="Linear drag",
    timescale="Alfven",
    bg=bg_active,
    bg_sub_H=True,
    # deactivate=["Bp_p", "Bp_m"],
    save_to=os.path.join(output_dir, "S_L2_N1/eqs_pg_ideal_vsH.json"),
    overwrite=False,
    verbose=5
)


from pg_utils.pg_model import expand_pg_partial_ext as xpd_cfg
import numpy as np

eigen.collect_matrix_elements(
    read_from=os.path.join(output_dir, "S_L2_N1/eqs_pg_ideal_vsH.json"),
    manual_select=xpd_cfg.field_indexer,
    expansion_recipe=xpd_cfg.recipe,
    save_to=os.path.join(output_dir, "S_L2_N1/Original/matrix_expr_ideal_vsH.json"),
    overwrite=False,
    verbose=5
)


"""
Numerical solution
"""


# # Malkus field / Hydrodynamic modes
# parameters = {
#     params.Le: sympy.Rational(1, 10000),
#     expansion.m: sympy.Integer(3) 
# }
# # Toroidal quadrupolar field
# parameters = {
#     bg_fields.BackgroundToroidalQuadrupole().params[0]: 3*sympy.sqrt(3)/2,
#     params.Le: sympy.sqrt(2)*sympy.Rational(1, 10000),
#     expansion.m: sympy.Integer(3)
# }
# # Poloidal dipole field
# parameters = {
#     bg_fields.BackgroundPoloidalDipoleTunable().params[0]: sympy.S.One,
#     params.Le: sympy.Rational(1, 10000),
#     # params.Lu: sympy.Integer(1000),
#     expansion.m: sympy.Integer(1)
# }
# # T1 background field
# bg_active = bg_fields.Background_T1()
# parameters = {
#     bg_active.params[0]: bg_active.params_ref[0][0],
#     params.Le: sympy.Rational(1, 10000),
#     expansion.m: sympy.Integer(3)
# }
# S1 background field
# bg_active = bg_fields.Background_S1()
# parameters = {
#     bg_active.params[0]: bg_active.params_ref[0][0],
#     params.Le: sympy.Rational(1, 10000),
#     expansion.m: sympy.Integer(3)
# }
# # S2 background field
# bg_active = bg_fields.Background_S2()
# parameters = {
#     bg_active.params[0]: bg_active.params_ref[0][0],
#     params.Le: sympy.Rational(1, 10000),
#     expansion.m: sympy.Integer(3)
# }
# # T1S1 background field
# bg_active = bg_fields.Background_T1S1()
# parameters = {
#     bg_active.params[0]: bg_active.params_ref[0][0],
#     bg_active.params[1]: bg_active.params_ref[0][1],
#     params.Le: sympy.Rational(1, 10000),
#     expansion.m: sympy.Integer(3)
# }
# # S L=2, N=2, Galerkin basis
# bg_active = bg_fields.Background_S_l2_n2()
# parameters = {
#     bg_active.params[0]: bg_active.params_ref[0][0],
#     params.Le: sympy.Rational(1, 10000),
#     expansion.m: sympy.Integer(3)
# }
# # T2 background field
# bg_active = bg_fields.Background_T2()
# parameters = {
#    bg_active.params[0]: bg_active.params_ref[0][0],
#    params.Le: sympy.Rational(1, 10000),
#    expansion.m: sympy.Integer(3)
# }
# No further parameters in the background field, dimless params only
parameters = {
    params.Le: sympy.Rational(1, 10000),
    expansion.m: sympy.Integer(3)
}

from pg_utils.pg_model import expand_stream_force_hybrid as xpd_cfg

results = eigen.compute_matrix_numerics(
        read_from=os.path.join(output_dir, 
        # "Hydrodynamic/Reduced/matrix_expr.json"
        "T1_SL2N1_std/Reduced/matrix_expr_ideal.json"
        # "T1_SL2N1_std/Original/matrix_expr_ideal.json"
        # "T1_SL2N1_std/Canonical/matrix_expr_ideal.json"
        # "T1_std/Reduced/matrix_expr_ideal_ch.json"
        ),
    xpd_recipe=xpd_cfg.recipe,
    Ntrunc=120,
    par_val=parameters,
    require_all_pars=False,
    quadratic_trunc=False,
    jacobi_rule_opt={"automatic": True, "quadN": None},
    quadrature_opt={
        "backend": "gmpy2", 
        "int_opt": {"n_dps": 33}, 
        "output": "numpy", 
        # "out_opt": {"dps": 33}
    },
    chop=1e-22,
    save_to=os.path.join(output_dir, 
        # "Hydrodynamic/Reduced/matrix_m3_N50_quad-p113_tmp.pkl"
        "T1_SL2N1_std/Reduced/matrix_ideal_m3_Le1e-4_N120_hybrid_p113.h5"
        # "T1_SL2N1_std/Original/matrix_ideal_m3_Le1e-4_N50_p113.h5"
        # "T1_SL2N1_std/Canonical/matrix_ideal_m3_Le1e-4_N50_p113.h5"
        # "T1_std/Reduced/matrix_ideal_m3_Le1e-4_N160_p113_ch.h5"
        # "S_L2_N2/Reduced/matrix_ideal_m3_Le1e-4_N120_prec113_orth.pkl"
        # "Poloidal_Dipole/Reduced/matrix_m3_N50_quad-p113_tmp.pkl"
        ),
    # save_to=None,
    format="hdf5",
    overwrite=False,
    verbose=5
)

eigen.compute_eigen(
    # read_from=results,
    read_from=os.path.join(output_dir, 
        # "Poloidal_Dipole/Reduced/matrix_m3_N50_quad-p113_tmp.pkl"
        "T1_SL2N1_std/Reduced/matrix_ideal_m3_Le1e-4_N120_hybrid_p113.h5"
        # "T1_SL2N1_std/Original/matrix_ideal_m3_Le1e-4_N50_p113.h5"
        # "T1_SL2N1_std/Canonical/matrix_ideal_m3_Le1e-4_N50_p113.h5"
        # "T1_std/Reduced/matrix_ideal_m3_Le1e-4_N160_p113_ch.h5"
        ),
    read_fmt="hdf5",
    save_to=os.path.join(output_dir, 
        "T1_SL2N1_std/Reduced/eigen_ideal_m3_Le1e-4_N120_hybrid_p113.h5"
        # "T1_SL2N1_std/Canonical/eigen_ideal_m3_Le1e-4_N50_p113.h5"
        # "T1_std/Reduced/eigen_ideal_m3_Le1e-4_N160_p113_ch.h5"
        # "Poloidal_Dipole/Reduced/eigen_m3_N50_quad-eigen-p113_tmp2.pkl"
        ),
    save_fmt="hdf5",
    diag=False,
    chop=1e-22,
    # prec=None,
    overwrite=False,
    verbose=5
)

