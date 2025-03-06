# -*- coding: utf-8 -*-

"""
Symbolic computation of the eigenvalue problem
"""

import os, sys, sympy
sys.path.append(os.getcwd())

from pg_utils import eigen, tools
from pg_utils.pg_model import bg_fields

output_dir = "./out/eigen/"
recipe_aliases = {
    'Original': 'pg',
    'Canonical': 'cg',
    'Reduced': 'reduced'
}


# bg_comps = [bg_fields.Background_S1(), bg_fields.Background_T2()]
# for bg_tmp in bg_comps:
#     bg_params = {bg_tmp.params[i_key]: bg_tmp.params_ref[0][i_key] for i_key in range(len(bg_tmp.params))}
#     bg_tmp.subs(bg_params, inplace=True)
# bg_active = bg_comps[0] + bg_comps[1]

# S L2-N1 poloidal field
bg_active = bg_fields.Background_S_l2_n1()

bg_dir = 'S_L2_N1'


recipe_name = 'Canonical'
recipe_symb = recipe_aliases[recipe_name]
forcings = ['Lorentz']
diff_m_mod = 'None'
time_scale = 'Alfven'

derive_eq = True
fname_eq = os.path.join(output_dir, bg_dir, 'eqs_ideal_%s_vs.json' % recipe_name.lower())

derive_sym_mat = True
fname_mat = os.path.join(output_dir, bg_dir, recipe_name, 'matrix_expr_ideal_vsHjx.json')

if recipe_name == 'Original':
    from pg_utils.pg_model import expand_pg_partial_ext as xpd_cfg
elif recipe_name == 'Canonical':
    from pg_utils.pg_model import expand_conjugate_ext as xpd_cfg
elif recipe_name == 'Reduced':
    from pg_utils.pg_model import expand_stream_force_hybrid as xpd_cfg


if __name__ == '__main__':
    
    timer = tools.ProcTimer(start=True)
    
    if derive_eq:
        eigen.form_equations(
            eq_mode=recipe_symb,
            components=forcings,
            diff_M=diff_m_mod,
            timescale=time_scale,
            bg=bg_active,
            bg_sub_H=True,
            save_to=fname_eq,
            overwrite=False,
            verbose=5,
            timer=timer
        )
    
    if derive_sym_mat:
        eigen.collect_matrix_elements(
            read_from=fname_eq,
            manual_select=xpd_cfg.field_indexer,
            expansion_recipe=xpd_cfg.recipe,
            process_hint='rational-jacobi',
            save_to=fname_mat,
            overwrite=False,
            verbose=5,
            timer=timer
        )

