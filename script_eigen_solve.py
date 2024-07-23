# -*- coding: utf-8 -*-
"""Script for solving the eigenvalue problem in axisymmetric background field
"""


from pg_utils import eigen
import os
output_dir = "./out/eigen/"


from pg_utils.pg_model import bg_fields

# eigen.form_equations(
#     eq_mode="cg",
#     components=["Lorentz"],
#     timescale="Alfven",
#     bg=bg_fields.BackgroundToroidalQuadrupole(),
#     # deactivate=["Bp_p", "Bp_m"],
#     save_to=os.path.join(output_dir, "Toroidal_Quadrupole/eqs_cg.json"),
#     overwrite=True,
#     verbose=5
# )


# from pg_utils.pg_model import expand_stream_force_hybrid as xpd_cfg
# import numpy as np

# eigen.collect_matrix_elements(
#     read_from=os.path.join(output_dir, "Poloidal_Dipole/eqs_reduced.json"),
#     manual_select=xpd_cfg.field_indexer,
#     expansion_recipe=xpd_cfg.recipe,
#     save_to=os.path.join(output_dir, "Poloidal_Dipole/Reduced/matrix_expr_hybrid.json"),
#     overwrite=False,
#     verbose=5
# )


from pg_utils.pg_model import expand_stream_force_hybrid as xpd_cfg
from pg_utils.pg_model import params, expansion
import sympy

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
# Poloidal dipole field
parameters = {
    params.Le: sympy.Rational(1, 100000),
    expansion.m: sympy.Integer(1)
}

results = eigen.compute_matrix_numerics(
        read_from=os.path.join(output_dir, 
        # "Hydrodynamic/Reduced/matrix_expr.json"
        # "Malkus/Reduced/matrix_expr.json"
        # "Toroidal_Quadrupole/Reduced/matrix_expr_orth.json"
        "Poloidal_Dipole/Reduced/matrix_expr_hybrid.json"
        # "Poloidal_Dipole/Transformed_ext/matrix_expr.json"
        ),
    xpd_recipe=xpd_cfg.recipe,
    Ntrunc=80,
    par_val=parameters,
    quadratic_trunc=True,
    jacobi_rule_opt={"automatic": True, "quadN": None},
    quadrature_opt={
        "backend": "gmpy2", 
        "int_opt": {"n_dps": 33}, 
        "output": "numpy", 
        # "out_opt": {"dps": 33}
    },
    chop=1e-25,
    save_to=os.path.join(output_dir, 
        # "Hydrodynamic/Reduced/matrix_m3_N50_quad-p113_tmp.pkl"
        # "Malkus/Reduced/matrix_m3_N50_cpt.h5"
        # "Toroidal_Quadrupole/Reduced/matrix_m3_Le-4_N50_orth.h5"
        # "Poloidal_Dipole/Reduced/matrix_m3_N50_p-quad.h5"
        "Poloidal_Dipole/Reduced/matrix_m1_Le1e-5_N80_p113_hybrid.h5"
        # "Poloidal_Dipole/Transformed_ext/matrix_m3_Le1e-4_N40_p113.h5"
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
        # "Hydrodynamic/Reduced/matrix_m3_N50_quad-p113_tmp.pkl"
        # "Malkus/Reduced/matrix_m3_N50_cpt.h5"
        # "Toroidal_Quadrupole/Reduced/matrix_m3_Le-4_N50_orth.h5"
        # "Poloidal_Dipole/Reduced/matrix_m3_N50_quad-p113_tmp.pkl"
        "Poloidal_Dipole/Reduced/matrix_m1_Le1e-5_N80_p113_hybrid.h5"
        # "Poloidal_Dipole/Transformed_ext/matrix_m3_Le1e-4_N40_p113.h5"
        # "../tmp/matrix_tmp.pkl"
        ),
    read_fmt="hdf5",
    save_to=os.path.join(output_dir, 
        # "Hydrodynamic/Reduced/eigen_m3_N50_quad-eigen-p113_tmp.h5"
        # "Malkus/Reduced/eigen_m3_N50_cpt.h5"
        # "Toroidal_Quadrupole/Reduced/eigen_m3_Le-4_N50_orth.h5"
        "Poloidal_Dipole/Reduced/eigen_m1_Le1e-5_N80_p113_hybrid.h5"
        # "Poloidal_Dipole/Transformed_ext/eigen_m3_Le1e-4_N40_p113.h5"
        # "Poloidal_Dipole/Reduced/eigen_m3_N50_quad-eigen-p113_tmp2.pkl"
        ),
    save_fmt="hdf5",
    diag=False,
    chop=1e-25,
    # prec=113,
    overwrite=False,
    verbose=5
)
