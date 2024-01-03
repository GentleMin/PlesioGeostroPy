# -*- coding: utf-8 -*-
"""Script for solving the eigenvalue problem in axisymmetric background field
"""


from pg_utils import eigen
import os
output_dir = "./out/eigen/"


# from pg_utils.pg_model import bg_fields

# eigen.form_equations(
#     eq_mode="reduced",
#     components=["Lorentz"],
#     timescale="Alfven",
#     bg=bg_fields.BackgroundPoloidalDipole(),
#     save_to=os.path.join(output_dir, "Poloidal_Dipole/eqs_reduced.json"),
#     verbose=5
# )


# from pg_utils.pg_model import expand_conjugate as xpd_cfg
# import numpy as np

# eigen.collect_matrix_elements(
#     read_from=os.path.join(output_dir, "Poloidal_Dipole/eqs_cg.json"),
#     manual_select=xpd_cfg.field_indexer,
#     expansion_recipe=xpd_cfg.recipe,
#     save_to=os.path.join(output_dir, "Poloidal_Dipole/Transformed/matrix_expr.json"),
#     verbose=5
# )


from pg_utils.pg_model import expand_conjugate as xpd_cfg
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
    params.Le: sympy.sqrt(2)*sympy.Rational(1, 10000),
    expansion.m: sympy.Integer(3)
}

eigen.compute_matrix_numerics(
    read_from=os.path.join(output_dir, 
        "Poloidal_Dipole/Transformed/matrix_expr.json"),
    xpd_recipe=xpd_cfg.recipe,
    Ntrunc=50,
    par_val=parameters,
    save_to=os.path.join(output_dir, 
        "Poloidal_Dipole/Transformed/matrix_m3_N50.h5"),
    verbose=5
)

eigen.compute_eigen(
    read_from=os.path.join(output_dir, 
        "Poloidal_Dipole/Transformed/matrix_m3_N50.h5"),
    save_to=os.path.join(output_dir, 
        "Poloidal_Dipole/Transformed/eigen_m3_N50.h5"),
    verbose=5
)
