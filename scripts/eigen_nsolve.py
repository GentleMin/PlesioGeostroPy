# -*- coding: utf-8 -*-

"""
Numerical solution of the eigenvalue problem
"""

import os, sys, sympy
sys.path.append(os.getcwd())

from pg_utils import eigen, tools
from pg_utils.pg_model import core, params, expansion, bg_fields
import sympy
output_dir = "./out/eigen/"


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
# # S L=2, N=1, Galerkin basis
# bg_active = bg_fields.Background_S_l2_n1()
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


bg_dir = 'S1_T2_std'
recipe_name = 'Canonical'
N_trunc = 100
f_mat_expr = os.path.join(output_dir, bg_dir, recipe_name, 'matrix_expr_ideal_vsHjx.json')
f_mat_eval = os.path.join(output_dir, bg_dir, recipe_name, 
    f'matrix_ideal_m{int(parameters[expansion.m])}_Le{float(parameters[params.Le]):.1e}_N{N_trunc}_vsHj_qp.h5')
# f_mat_eval = None
f_eigen = os.path.join(output_dir, bg_dir, recipe_name, 
    f'eigen_ideal_m{int(parameters[expansion.m])}_Le{float(parameters[params.Le]):.1e}_N{N_trunc}.h5')

compute_matrix = True
compute_eigen = False


if recipe_name == 'Original':
    from pg_utils.pg_model import expand_pg_partial_ext as xpd_cfg
elif recipe_name == 'Canonical':
    from pg_utils.pg_model import expand_conjugate_ext as xpd_cfg
elif recipe_name == 'Reduced':
    from pg_utils.pg_model import expand_stream_force_hybrid as xpd_cfg


if __name__ == '__main__':
    
    timer = tools.ProcTimer(start=True)
    
    if compute_matrix:
        results = eigen.compute_matrix_numerics(
            read_from=f_mat_expr,
            xpd_recipe=xpd_cfg.recipe,
            Ntrunc=N_trunc,
            par_val=parameters,
            require_all_pars=False,
            quadratic_trunc=False,
            jacobi_rule_opt={
                "automatic": True, "quadN": None, 
                'prefactors': [core.H, core.s],
                'quadN_redundancy': 10
            },
            quadrature_opt={
                "backend": "gmpy2", 
                "int_opt": {"n_dps": 33}, 
                "output": "numpy", 
                # "out_opt": {"dps": 33}
            },
            chop=1e-22,
            save_to=f_mat_eval,
            format="hdf5",
            overwrite=True,
            verbose=5,
            timer=timer
        )
        
    if compute_eigen:
        if f_mat_eval is None:
            f_mat_eval = results
        eigen.compute_eigen(
            read_from=f_mat_eval,
            read_fmt="hdf5",
            save_to=f_eigen,
            save_fmt="hdf5",
            diag=False,
            chop=1e-22,
            # prec=None,
            overwrite=False,
            verbose=5
        )

