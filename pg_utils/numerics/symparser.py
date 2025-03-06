# -*- coding: utf-8 -*-
"""
Symbolic parser

The bridge between the symbolic expressions and numerical computations
"""

import numpy as np
import sympy
import mpmath
import gmpy2
import sympy.functions
import sympy.functions.special
import sympy.functions.special.polynomials

from . import special
from sympy.printing.pycode import PythonCodePrinter


def powers_of(expr: sympy.Expr, *args: sympy.Symbol, return_expr: bool = False):
    """Retrieve the power of symbols in a given expression.
    
    :param sympy.Expr expr: symbolic expression
    :param sympy.Symbol expr: symbols whose powers are to be estimated
    :param bool return_expr: whether to return the expressions
    :returns: list of powers, optionally with the respective terms (if `return_expr`)
    
    Usage:
    
    Assume we have symbols defined as ``p, q, a, b, n, m = sympy.symbols("p q a b m n")``.
    We can calculate the powers in a monomial::
    
        >>> sample_monomial = p**2*q**(n + 2*m)*jacobi(n, a, b, p**2 + 1)
        >>> powers_of(sample_monomial, p, q)
        [2*n + 2, 2*m + n]
    
    Or we can calculate the powers inccurred in a polynomial::
    
        >>> sample_polynomial = q**2*chebyshev(n, p*q) + p**(n + m)*sp.jacobi(n, a, b, p**2*q**3)
        >>> powers_of(sample_polynomial, p, q)
        [[m + 4*n, 2*n], [n, n + 2]]
    
    .. warning::
    
        This is a very intricate method, and must be used with care, 
        with sanitized input.
        
        If a special function is present in the expression, it will
        be interpreted as a polynomial, whose first argument is the
        degree of the polynomial. This at least works for Jacobi
        polynomials and their special types.
    """
    if isinstance(expr, sympy.Add):
        # When the expression is an addition, collect
        # all powers for each term separately
        powers = [powers_of(term, *args, return_expr=return_expr) for term in expr.args]
        return powers
    elif isinstance(expr, sympy.Function):
        # This is the most intricate part of this method
        # When the expression is a pure special function
        # We assume this Function is a polynomial function,
        # with degree as the first argument
        # and variable as the last argument.
        # Further, the variable needs to be a polynomial in symbol
        powers = [sympy.S.Zero for arg in args]
        for i_symb, symb in enumerate(args):
            arg_deg = sympy.degree(expr.args[-1], gen=symb)
            fun_deg = expr.args[0]
            powers[i_symb] += arg_deg*fun_deg
        return powers
    else:
        expr = expr.factor()
        powers = [sympy.S.Zero for arg in args]
        for i_symb, symb in enumerate(args):
            for arg in expr.args:
                if isinstance(arg, sympy.Pow):
                    # For a power term, simply extract base and exponent
                    base, exp = arg.as_base_exp()
                    if base == symb:
                        powers[i_symb] += exp
                elif isinstance(arg, sympy.Function):
                    # This is the most intricate part of this method
                    # We assume this Function is a polynomial function,
                    # with degree as the first argument
                    # and variable as the last argument.
                    # This applies to all Jacobi polynomials.
                    # Further, the variable needs to be a polynomial in symbol
                    arg_deg = sympy.degree(arg.args[-1], gen=symb)
                    fun_deg = arg.args[0]
                    powers[i_symb] += arg_deg*fun_deg
                elif isinstance(arg, sympy.Symbol):
                    # For a single symbol, simply add one
                    if arg == symb:
                        powers[i_symb] += sympy.S.One
        if return_expr:
            return powers, expr
        else:
            return powers


def leading_powers_factor(expr: sympy.Expr, *args: sympy.Expr):
    """Derive leading powers of a single factor
    """
    if isinstance(expr, sympy.Pow):
        # For a power term, simply extract base and exponent
        base, exp = expr.as_base_exp()
        powers = [exp if base == arg else sympy.S.Zero for arg in args]
        return powers
    if isinstance(expr, sympy.Symbol) or isinstance(expr, sympy.Function):
        # If just a symbol or function: compare if is arg
        powers = [sympy.S.One if expr == arg else sympy.S.Zero for arg in args]
        return powers
    powers = [sympy.S.Zero for arg in args]
    return powers


def leading_powers_of(expr: sympy.Expr, *args: sympy.Expr):
    """Derive leading powers of an expression
    """
    if isinstance(expr, sympy.Add):
        # For a sum, collect all powers for each term separately
        powers = [leading_powers_of(term, *args) for term in expr.args]
        return powers
    if isinstance(expr, sympy.Mul):
        # For a product, add powers of each factor
        powers = [sympy.S.Zero for arg in args]
        for factor in expr.factor().args:
            powers_factor = leading_powers_factor(factor, *args)
            for i_sym in range(len(args)):
                powers[i_sym] += powers_factor[i_sym]
        return powers
    # Otherwise: consider the expression as a single factor
    return leading_powers_factor(expr, *args)


def jacobi_idx_subs(expr: sympy.Expr, arg: sympy.Symbol, 
    arg_a: sympy.Symbol = sympy.Symbol("p"), 
    arg_b: sympy.Symbol = sympy.Symbol("q")) -> sympy.Expr:
    """Jacobi index substitution.
    
    This function replaces the 1-`arg` and 1+`arg` factors in the expression
    to `arg_a` and `arg_b`, respectively, so as to facilitate derivation and 
    calculation of Jacobi-like inner products.
    
    :param sympy.Expr expr: expression
    :param sympy.Symbol arg: the main variable x
    :param sympy.Symbol arg_a: the symbol for 1 - x factors
    :param sympy.Symbol arg_b: the symbol for 1 + x factors
    :returns: processed expression with 1 - x and 1 + x replaced
    
    .. warning::
    
        This function internally uses `xreplace` in order to avoid mixing 
        1 + x and 1 - x terms. However, this also means that unless the expression
        contains the terms in the exact expression, the terms won't be properly
        replaced. The following terms are currently supported:
        * :math:`1 - x`
        * :math:`x - 1`
        * :math:`1/2 - x/2`
        * :math:`x/2 - 1/2`
        * :math:`1 + x`
        * :math:`1/2 + x/2`
    """
    replace_map = {
        1 - arg: arg_a, 
        arg - 1: -arg_a,
        sympy.Rational(1, 2) - arg/2: arg_a/2, 
        arg/2 - sympy.Rational(1, 2): -arg_a/2, 
        1 + arg: arg_b, 
        sympy.Rational(1, 2) + arg/2: arg_b/2
    }
    return expr.xreplace(replace_map)


v_functions_mpmath = {
    'sin': np.vectorize(mpmath.sin, otypes=(object,)),
    'cos': np.vectorize(mpmath.cos, otypes=(object,)),
    'tan': np.vectorize(mpmath.tan, otypes=(object,)),
    'sqrt': np.vectorize(mpmath.sqrt, otypes=(object,))
}
"""Vectorized functions in mpmath"""


v_functions_gmpy2 = {
    'sin': np.vectorize(gmpy2.sin, otypes=(object,)),
    'cos': np.vectorize(gmpy2.cos, otypes=(object,)),
    'tan': np.vectorize(gmpy2.tan, otypes=(object,)),
    'sqrt': np.vectorize(gmpy2.sqrt, otypes=(object,))
}
"""Vectorized functions in gmpy2"""


class Gmpy2Printer(PythonCodePrinter):
    """
    Lambda printer for gmpy2 which maintains precision for floats
    """
    printmethod = "_gmpy2code"
    language = "Python with gmpy2"
    
    def __init__(self, settings = None, dps: int = None, prec: int = None):
        super().__init__(settings)
        self.dps, self.prec = special.transform_dps_prec(dps=dps, prec=prec)
    
    def _print_Float(self, e: sympy.Expr):
        return '{func}("{arg}", {prec})'.format(
            func=self._module_format('gmpy2.mpf'),
            arg=str(e),
            prec=self.prec
        )
    
    def _print_Rational(self, e: sympy.Expr):
        return '{func}({p})/{func}({q})'.format(
            func=self._module_format('gmpy2.mpz'), 
            p=self._print(e.p), 
            q=self._print(e.q)
        )
    
    def _print_Integer(self, e: sympy.Expr):
        return '{func}({arg})'.format(
            func=self._module_format('gmpy2.mpz'),
            arg=str(e)
        )
    
    def _print_Pi(self, e: sympy.Expr):
        return '{func}(precision={prec})'.format(
            func=self._module_format('gmpy2.const_pi'),
            prec=self.prec
        )

