# -*- coding: utf-8 -*-

import os, json
import numpy as np
from pg_utils import pg_model
from pg_utils.pg_model import expansion as xpd
import matplotlib.pyplot as plt

cwd = os.getcwd()
opath = os.path.join(cwd, 'out/eigen/')


# fpath = os.path.join(opath, 'Poloidal_Dipole/Transformed_ext/matrix_expr_dragl.json')
fpath = os.path.join(opath, 'Toroidal_Quadrupole/Transformed_ext/matrix_expr_dragl.json')
# ofile = os.path.join(opath, 'Poloidal_Dipole/Transformed_ext/mexpr_spy_dragl.png')
ofile = os.path.join(opath, 'Toroidal_Quadrupole/Transformed_ext/mexpr_spy_dragl.png')
# ofile = None

with open(fpath, 'r') as fread:
    matrix_obj = json.load(fread)
xpd_id = matrix_obj['xpd']
cfg = pg_model.xpd_options[xpd_id]
M_expr = xpd.SystemMatrix.deserialize(matrix_obj['M'])
K_expr = xpd.SystemMatrix.deserialize(matrix_obj['K'])

M_shape = M_expr.block_sparsity()
K_shape = K_expr.block_sparsity()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

ax = axes[0]
ax.spy(M_shape)
ax.set_xticks(np.arange(cfg.coeff_s.n_fields))
ax.set_xticklabels([r'$%s$' % cfg.coeff_s[fname].name for fname in cfg.coeff_s._field_names], fontsize=12)
ax.set_yticks(np.arange(cfg.fields.n_fields))
ax.set_yticklabels([r'$%s$' % cfg.fields[fname].name for fname in cfg.fields._field_names], fontsize=12)
ax.set_title("Mass matrix", fontsize=10)

ax = axes[1]
ax.spy(K_shape)
ax.set_xticks(np.arange(cfg.coeff_s.n_fields))
ax.set_xticklabels([r'$%s$' % cfg.coeff_s[fname].name for fname in cfg.coeff_s._field_names], fontsize=12)
ax.set_yticks(np.arange(cfg.fields.n_fields))
ax.set_yticklabels([r'$%s$' % cfg.fields[fname].name for fname in cfg.fields._field_names], fontsize=12)
ax.set_title("Stiffness matrix", fontsize=10)

plt.tight_layout()
if ofile is not None:
    plt.savefig(ofile, format="png", dpi=150, bbox_inches="tight")
plt.show()
