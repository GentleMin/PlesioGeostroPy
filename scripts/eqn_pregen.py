# -*- coding: utf-8 -*-

"""
Pre-generate and store PG equations

Jingtao Min @ ETH-EPM, 12.2024
"""


import sympy as sym
import os, sys

dir_pg = os.getcwd()
sys.path.append(os.getcwd())
import pg_utils.tools as pgtools

proc_timer = pgtools.ProcTimer(start=True)
proc_timer.flag("Equation parsing started.", print_str=True)
import pg_utils.pg_model.equations as pgeq
proc_timer.flag("Equation parsing finished.", print_str=True)


eqs_lib = os.path.join(dir_pg, 'out/symbolic/')
save_streams = [
    ('PG-Original', pgeq.eqs_pg, os.path.join(eqs_lib, 'eqs-pg__boundIE-Bcyl.json')),
    ('PG-Original-lin', pgeq.eqs_pg_lin, os.path.join(eqs_lib, 'eqs-pg__boundIE-Bcyl__lin.json')),
    ('PG-Canonical', pgeq.eqs_cg, os.path.join(eqs_lib, 'eqs-cg__boundIE-Bcyl.json')),
    ('PG-Canonical-lin', pgeq.eqs_cg_lin, os.path.join(eqs_lib, 'eqs-cg__boundIE-Bcyl__lin.json'))
]


if __name__ == '__main__':
    
    for sys_name, eqs_set, o_fname in save_streams:
        if o_fname is None:
            continue
        with open(o_fname, 'x') as fwrite:
            eqs_set.save_json(fwrite, serializer=sym.srepr)
        proc_timer.flag(loginfo=f'System {sys_name} saved to {o_fname}', print_str=True)

