# -*- coding: utf-8 -*-
"""
Supplementary special functions and custom functions
"""

import sympy as sym
from sympy.core.function import ArgumentIndexError


def check_integer(n: sym.Expr):
    """Check if an input expression can be integer
    """
    return isinstance(n, sym.Integer) or (n.is_integer is not False)


class jacobi_2side(sym.Function):
    r"""
    Two-sided Jacobi polynomial 
    
    .. math:: 
    
        P_n^{(k_1, k_2, \alpha, \beta)}(x) 
        = \sqrt{1-x}^{k_1} \sqrt{1+x}^{k_2} P_n^{(\alpha,\beta)}(x)
    """
    
    @classmethod
    def eval(cls, n, k1, k2, a, b, x):
        """Auto-evaluation: defines the input arguments; never evaluate to anything else
        """
        if not (check_integer(n) and check_integer(k1) and check_integer(k2)):
            raise TypeError('n, k1, k2 should be integers')
        
    def _eval_rewrite(self, rule, args, **hints):
        n, k1, k2, a, b, x = args
        if rule == sym.jacobi:
            p = sym.sqrt(1 - x)
            q = sym.sqrt(1 + x)
            expr_rewrite = (p**k1)*(q**k2)*sym.jacobi(n, a, b, x)
            return expr_rewrite
        
    def _eval_evalf(self, prec):
        expr_eval = self.rewrite(sym.jacobi)
        return expr_eval._eval_evalf(prec)
    
    def fdiff(self, argindex=6):
        raise NotImplementedError
    
    def _latex(self, printer, exp=None):
        _n, _k1, _k2, _a, _b, _x = map(printer._print, self.args)
        o_str = r'P_{%s}^{(%s,%s,%s,%s)}(%s)' % (_n, _k1, _k2, _a, _b, _x)
        if exp is None:
            return o_str
        else:
            return r'\left(%s\right)^2' % o_str
    

class jacobi_polar(sym.Function):
    r"""
    Two-sided Jacobi polynomial with polar radius shifting
    
    .. math:: 
    
        Q_n^{(k_1, k_2,\alpha, \beta)}(r) 
        = \sqrt{1-r^2}^{k_1} \sqrt{r}^{k_2} P_n^{(\alpha,\beta)}(2r^2 - 1)
    """
    
    @classmethod
    def eval(cls, n, k1, k2, a, b, r):
        """Auto-evaluation: defines the input arguments; never evaluate to anything else
        """
        if not (check_integer(n) and check_integer(k1) and check_integer(k2)):
            raise TypeError('n, k1, k2 should be integers')
        
    def _eval_rewrite(self, rule, args, **hints):
        n, k1, k2, a, b, r = args
        if rule == sym.jacobi:
            p = sym.sqrt(1 - r**2)
            xi = 2*r**2 - 1
            expr_rewrite = (p**k1)*(r**k2)*sym.jacobi(n, a, b, xi)
            return expr_rewrite
        
    def _eval_evalf(self, prec):
        expr_eval = self._eval_rewrite(sym.jacobi, self.args)
        return expr_eval._eval_evalf(prec)
    
    def fdiff(self, argindex=6):
        if argindex != 6:
            raise ArgumentIndexError
        n, k1, k2, a, b, r = self.args
        return (
            - k1*jacobi_polar(n, k1-2, k2+1, a, b, r) 
            + k2*jacobi_polar(n, k1, k2-1, a, b, r)
            + 2*(n + a + b + 1)*jacobi_polar(n-1, k1, k2+1, a+1, b+1, r)
        )
    
    def _latex(self, printer, exp=None):
        _n, _k1, _k2, _a, _b, _r = map(printer._print, self.args)
        o_str = r'Q_{%s}^{(%s,%s,%s,%s)}(%s)' % (_n, _k1, _k2, _a, _b, _r)
        if exp is None:
            return o_str
        else:
            return r'\left(%s\right)^2' % o_str
    
