# -*- coding: utf-8 -*-
"""
Expansion configuration file - 
Expansion for the streamfunction-force formulation

..Note: this set of bases and expansions has to be used
    together with the reduced system of equations, 
    where the only equations present are Psi and F_ext

..Note: this is only relevant to eigenvalue problems or
    linearized systems. Nonlinear PG equations do not 
    generally simplify into streamfunction - force 
    formulation, but requires full set of PG variables
    or their conjugate counterparts

Jingtao Min @ ETH Zurich 2023
"""

import numpy as np
from sympy import S, Symbol, Function, Rational, Abs, jacobi
from .core import s, p, t, H, H_s, reduced_var
from .expansion import n, m, xi, xi_s, s_xi
from . import expansion, base

identifier = "expand_reduced_Psi-F"


# Which equations to use
field_names = reduced_var._field_names
field_indexer = np.array([True, True])
subscript_str = [r"\Psi", "F"]


"""Fourier expansion"""

# fields
fields = reduced_var.copy()

# Fourier expansion
fourier_expand = expansion.FourierExpansions(
    expansion.omega*t + expansion.m*p, fields, 
    expansion.reduced_var_s)


"""Radial expansion"""

# bases (placeholder)
bases_s = base.LabeledCollection(field_names,
    **{fname: Function(r"\Phi_{%s}^{mn}" % subscript_str[idx])(s) 
       for idx, fname in enumerate(field_names)})

# explicit expression for the radial bases

bases_s_expression = base.LabeledCollection(field_names,
    # Streamfunction
    Psi = H_s**3*s**Abs(m)*jacobi(n, Rational(3, 2), Abs(m), xi_s),
    # External body force
    F_ext = s**(Abs(m) + 1)*jacobi(n, 0, Abs(m) + Rational(1, 2), xi_s)
)

# coefficients
coeff_s = base.LabeledCollection(field_names,
    **{fname: Symbol(r"C_{%s}^{mn}" % subscript_str[idx])
       for idx, fname in enumerate(field_names)})

rad_expand = expansion.RadialExpansions(fourier_expand.coeffs, bases_s, coeff_s,
    Psi = coeff_s.Psi*bases_s.Psi,
    F_ext = coeff_s.F_ext*bases_s.F_ext
)

"""Test functions"""

# Even when the test functions are the same with the trial functions
# It is required that they use different indices (or different placeholders)
# so that the two can be distinguished.
test_s = base.LabeledCollection(field_names,
    **{fname: Function(r"\Phi_{%s}^{mn'}" % subscript_str[idx])(s) 
       for idx, fname in enumerate(field_names)})


"""Inner products"""

inner_prod_op = expansion.RadialInnerProducts(field_names,
    **{fname: expansion.InnerProductOp1D(s, S.One, (S.Zero, S.One))
       for fname in field_names}
)

recipe = expansion.ExpansionRecipe(
    fourier_expand=fourier_expand,
    rad_expand=rad_expand,
    rad_test=test_s,
    inner_prod_op=inner_prod_op,
    base_expr=bases_s_expression.subs({n: expansion.n_trial}),
    test_expr=bases_s_expression.subs({n: expansion.n_test})
)

