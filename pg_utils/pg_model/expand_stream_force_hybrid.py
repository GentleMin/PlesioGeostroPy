# -*- coding: utf-8 -*-
"""
Expansion configuration file - 
Expansion for the streamfunction-force formulation
with forcing spectrum as compact as streamfunction.

.. note:: This set of bases and expansions has to be used
    together with the reduced system of equations, 
    where the only equations present are Psi and F_ext

.. note:: This is only relevant to eigenvalue problems or
    linearized systems. Nonlinear PG equations do not 
    generally simplify into streamfunction - force 
    formulation, but requires full set of PG variables
    or their conjugate counterparts
    
Using this expansion configuration, the basis functions takes the form

.. math:: 
    
    \\Psi^{nm}(s) = s^{|m|} H^3 J_n^{(\\frac{3}{2}, |m|)}(2s^2 - 1)
    
    F^{nm}(s) = s^{|m|} J_n^{(\\frac{3}{2}, |m|)}(2s^2 - 1)

Unlike :py:mod:~pg_utils.pg_model.expand_stream_force_orth``,
the forcing basis in this configuration is not self-orthogonal.
The orthgonality property reads

.. math:: 
    
    \\int_0^1 \\Psi^{n'm}(s) \\Psi^{nm}(s) \\frac{s}{H^3} ds = N_{\\Psi}^{nm} \\delta_{nn'}
    
    \\int_0^1 \\Psi^{n'm}(s) F^{nm}(s) ds = N_{F}^{nm} \\delta_{nn'}

Hence the two fields share the same set of test functions.

This configuration has the desirable property that 
a compact streamfunction spectrum translates to a compact
forcing spectrum. This is due to the fact that 
the forcing basis is directly linked with the streamfunction basis.
"""

import numpy as np
from sympy import S, Symbol, Function, Rational, Abs, jacobi
from .core import s, p, t, H, H_s, reduced_var
from .expansion import n, m, xi, xi_s, s_xi
from . import expansion, base

identifier = "expand_reduced_Psi-F_compact"


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
    F_ext = s**(Abs(m) + 1)*jacobi(n, Rational(3, 2), Abs(m), xi_s)
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

test_s_expression = base.LabeledCollection(field_names, 
    Psi = bases_s_expression.Psi,
    F_ext = s**(Abs(m) + 1)*jacobi(n, 0, Abs(m) + Rational(1, 2), xi_s)
)


"""Inner products"""

inner_prod_op = expansion.RadialInnerProducts(field_names,
    **{fname: expansion.InnerProductOp1D(s, S.One, (S.Zero, S.One))
       for fname in field_names}
)

recipe = expansion.ExpansionRecipe(
    identifier=identifier,
    fourier_expand=fourier_expand,
    rad_expand=rad_expand,
    rad_test=test_s,
    inner_prod_op=inner_prod_op,
    base_expr=bases_s_expression.subs({n: expansion.n_trial}),
    test_expr=test_s_expression.subs({n: expansion.n_test})
)

