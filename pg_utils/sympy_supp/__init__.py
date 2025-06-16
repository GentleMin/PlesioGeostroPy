# -*- coding: utf-8 -*-
"""
Supplementary functions for symbolic manipulation in sympy

Sub-modules include

* :mod:`~pg_utils.sympy_supp.vector_calculus_3d` functionalities for 3-D vector calculus

"""

import functools
from .functions import jacobi_u
import sympy as sym


func_dict = {
    'jacobi_u': jacobi_u
}

parse_expr_custom = functools.partial(sym.parse_expr, local_dict=func_dict)

