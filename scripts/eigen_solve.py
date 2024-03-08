# -*- coding: utf-8 -*-
"""Script for solving the eigenvalue problem in axisymmetric background field

Note: this script needs to run in the same level as the folder `pg_utils`!
"""


from pg_utils import eigen
import os
output_dir = "./out/tmp/"


"""
===========================================================
Step 1: Form equations and apply background field
===========================================================
"""

from pg_utils.pg_model import bg_fields

eigen.form_equations(
    eq_mode="reduced",
    components=["Lorentz"],
    timescale="Alfven",
    bg=bg_fields.BackgroundPoloidalDipole(),
    # # The commented keyword argument below manually deactivates certain equations
    # deactivate=["Bp_p", "Bp_m"],
    save_to=os.path.join(output_dir, "Poloidal_Dipole/eqs_reduced.json"),
    overwrite=True,
    verbose=5
)


"""
===========================================================
Step 2: Collect symbolic expressions of matrix elements
===========================================================
"""

from pg_utils.pg_model import expand_stream_force_cpt as xpd_cfg
import numpy as np

eigen.collect_matrix_elements(
    read_from=os.path.join(output_dir, "Poloidal_Dipole/eqs_reduced.json"),
    manual_select=xpd_cfg.field_indexer,
    expansion_recipe=xpd_cfg.recipe,
    save_to=os.path.join(output_dir, "Poloidal_Dipole/Reduced/matrix_expr_orth.json"),
    overwrite=False,
    verbose=5
)


"""
===========================================================
Step 3: Expand matrices numerically
===========================================================
"""

from pg_utils.pg_model import expand_stream_force_orth as xpd_cfg
from pg_utils.pg_model import params, expansion
import sympy

# # Malkus field / Hydrodynamic modes
# parameters = {
#     params.Le: sympy.Rational(1, 10000),
#     expansion.m: sympy.Integer(3)
# }
# Toroidal quadrupolar field
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
    read_from=os.path.join(output_dir, "Poloidal_Dipole/Reduced/matrix_expr_orth.json"),
    xpd_recipe=xpd_cfg.recipe,
    Ntrunc=50,
    par_val=parameters,
    jacobi_rule_opt={"automatic": True, "quadN": None},
    # # The commented block below is used for multi-prec evaluation
    # # Note: if saving in multi-prec form needed, use json or pickle instead of hdf5
    # quadrature_opt={
    #     "backend": "gmpy2", 
    #     "int_opt": {"n_dps": 33}, 
    #     "output": "gmpy2", 
    #     "out_opt": {"dps": 33}
    # },
    save_to=os.path.join(output_dir, "Poloidal_Dipole/Reduced/matrix_m3_N50.h5"),
    format="hdf5",
    overwrite=False,
    verbose=5
)


"""
===========================================================
Step 4: Compute eigenvalues and eigenvectors from matrices
===========================================================
"""

eigen.compute_eigen(
    read_from=os.path.join(output_dir, "Poloidal_Dipole/Reduced/matrix_m3_N50.h5"),
    read_fmt="hdf5",
    save_to=os.path.join(output_dir, "Poloidal_Dipole/Reduced/eigen_m3_N50.h5"),
    save_fmt="hdf5",
    diag=True,
    # # The commented keyword argument below chops values smaller than a threshold to 0
    # chop=1e-25,
    # # The commented keyword below tells the program to use a multi-prec eigensolver
    # prec=113,
    overwrite=False,
    verbose=5
)
