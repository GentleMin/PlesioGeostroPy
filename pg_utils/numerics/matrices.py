# -*- coding: utf-8 -*-
"""
Symbolic manipulation and numerical computations 
of the coefficient matrices (mass and stiffness matrices)

The bridge between the symbolic expressions and numerical computations
"""


import warnings

import sympy
from sympy.integrals import quadrature as qdsym

import numpy as np
from scipy import special as specfun

from ..pg_model import core
from ..pg_model import expansion as xpd
from ..pg_model.expansion import xi, n_test, n_trial

from typing import List, Union, Optional


def powers_of(expr: sympy.Expr, *args: sympy.Symbol):
    """Retrieve the power of symbols.
    This is a very intricate method, and must be used with care, 
    with sanitized input.
    """
    if isinstance(expr, sympy.Add):
        # When the expression is an addition, collect
        # all powers for each term separately
        powers = [powers_of(term, *args) for term in expr.args]
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
        return powers



class InnerQuad_Rule:
    """Quadrature of inner product based on certain rule
    Base class for inner product quad evaluators
    """
    
    def __init__(self, inner_prod: xpd.InnerProduct1D) -> None:
        self.inner_prod = inner_prod
        self.int_var = self.inner_prod._int_var
    
    def gram_matrix(self, nrange_trial: List[int], nrange_test: List[int],
        *args, **kwargs) -> Union[np.ndarray, np.matrix, sympy.Matrix]:
        raise NotImplementedError



class InnerQuad_GaussJacobi(InnerQuad_Rule):
    
    def __init__(self, inner_prod: xpd.InnerProduct1D, automatic: bool = False,
        alpha: Union[float, int, sympy.Expr] = -sympy.Rational(1, 2), 
        beta: Union[float, int, sympy.Expr] = -sympy.Rational(1, 2), 
        quadN: Optional[int] = None) -> None:
        """
        """
        super().__init__(inner_prod)
        self.deduce = automatic
        if self.deduce:
            powers_list = self.get_powers(self.int_var, 
                self.inner_prod.integrand().doit())
            if isinstance(powers_list[0], list):
                self.alpha = [powers[0] for powers in powers_list]
                self.beta = [powers[1] for powers in powers_list]
                self.quadN = [powers[2] for powers in powers_list]
            else:
                self.alpha = powers_list[0]
                self.beta = powers_list[1]
                self.quadN = powers_list[2]
        else:
            self.alpha, self.beta = alpha, beta
            self.quadN = quadN
        
    @classmethod
    def get_powers(cls, int_var: sympy.Symbol, expr: sympy.Expr) -> np.ndarray:
        p1 = sympy.Symbol(r"p_1", positive=True)
        p2 = sympy.Symbol(r"p_2", positive=True)
        replace_map = {1 - int_var: p1, int_var - 1: -p1, 
            sympy.Rational(1, 2) - int_var/2: p1/2, 
            int_var/2 - sympy.Rational(1, 2): -p1/2, 
            1 + int_var: p2, sympy.Rational(1, 2) + int_var/2: p2/2}
        expr = expr.xreplace(replace_map).expand()
        return powers_of(expr, p1, p2, int_var)
    
    def deduce_params(self, Ntrial: int, Ntest: int):
        deduce_map = {n_trial: Ntrial, n_test: Ntest}
        if isinstance(self.alpha, list):
            alpha = np.array([tmp.subs(deduce_map) for tmp in self.alpha])
            beta = np.array([tmp.subs(deduce_map) for tmp in self.beta])
            quadN = np.array([tmp.subs(deduce_map) for tmp in self.quadN])
            alpha_min = alpha.min()
            beta_min = beta.min()
            quad_max = quadN.max()
            pow_diff = [not term.is_integer for term in alpha - alpha_min] \
                + [not term.is_integer for term in beta - beta_min]
            if np.array(pow_diff).sum() > 0:
                warnings.warn("Incompatible singularities!"
                    "The quadrature cannot integrate exactly.")
            return alpha_min, beta_min, quad_max
        else:
            return (self.alpha.subs(deduce_map), 
                    self.beta.subs(deduce_map), 
                    self.quadN.subs(deduce_map))
    
    def gram_matrix(self, nrange_trial: List[int], nrange_test: List[int], 
        backend: str="sympy", output: str="sympy", int_opt: dict = {}, 
        simp_opt: dict = {}) -> Union[np.ndarray, np.matrix, sympy.Matrix]:
        """
        """
        alpha, beta, quadN = self.deduce_params(max(nrange_trial), max(nrange_test))
        print(alpha, beta, quadN)
        if backend == "sympy":
            M = self._quad_Jacobi_sympy(nrange_trial, nrange_test, 
                alpha, beta, quadN, **int_opt)
        elif backend == "scipy":
            M = self._quad_Jacobi_scipy(nrange_trial, nrange_test, 
                alpha, beta, quadN)
        else:
            raise AttributeError
        return self._output_form(M, output=output, **simp_opt)
    
    def _quad_Jacobi_sympy(self, nrange_trial: List[int], nrange_test: List[int], 
        alpha: Union[float, int, sympy.Expr], beta: Union[float, int, sympy.Expr], 
        quad_N: int, precision: int = 16) -> sympy.Matrix:
        """
        """
        xi_quad, wt_quad = qdsym.gauss_jacobi(quad_N, alpha, beta, precision)
        integrand = self.inner_prod.integrand()/(1 - xi)**alpha/(1 + xi)**beta
        
        M = list()
        for n_test_val in nrange_test:
            M_row = list()
            for n_trial_val in nrange_trial:
                int_tmp = integrand.subs({n_test: n_test_val, n_trial: n_trial_val}).doit()
                quad_pts = [wt_quad[i]*int_tmp.subs({xi: xi_quad[i]}, n=precision) 
                    for i in range(quad_N)]
                M_row.append(sum(quad_pts))
            M.append(M_row)
        return sympy.Matrix(M)
    
    def _quad_Jacobi_scipy(self, nrange_trial: List[int], nrange_test: List[int], 
        alpha: Union[float, int, sympy.Expr], beta: Union[float, int, sympy.Expr], 
        quad_N: int) -> np.ndarray:
        """
        """
        xi_quad, wt_quad = specfun.roots_jacobi(quad_N, alpha, beta)
        integrand = self.inner_prod.integrand()/(1 - xi)**alpha/(1 + xi)**beta
        int_fun = sympy.lambdify([n_test, n_trial, xi], integrand.doit(), 
            modules=["scipy", "numpy"])
        Ntest, Ntrial, Xi = np.meshgrid(nrange_test, nrange_trial, xi_quad, indexing='ij')
        return np.sum(int_fun(Ntest, Ntrial, Xi)*wt_quad, axis=-1)
    
    def _output_form(self, M_in: Union[np.ndarray, np.matrix, sympy.Matrix], 
        output: str = "sympy", **kwargs) -> Union[np.ndarray, np.matrix, sympy.Matrix]:
        """
        """
        if output == "sympy":
            return sympy.nsimplify(M_in, **kwargs)
        elif output == "numpy":
            if isinstance(M_in, sympy.Matrix):
                return np.array(M_in).astype(np.float64)
            else:
                return M_in
        else:
            raise AttributeError
    
    

class InnerProdQuad:
    """Quadrature of inner product
    class generator for all inner product quadratures in 1D
    
    Compared to the direct quadratures of the integral form,
    calculating quadratures in the notation of inner products
    "allows" one to drastically save of time of basis evaluation.
    When calculating the integral in the form of
        Integral(w(x)*Phi1(l, x)*Phi2(n, x), (x, -1, 1))
    directly calculating the integral using K-point quadrature
    for 0 <= l,n <= N would need KN^2 evaluations of both Phi1
    and Phi2; however, Phi1 and Phi2 in fact only need to be 
    evaluated KN times. This reduces the complexity of evaluation
    from O(KN^2) to O(KN).
    """
    
    def __new__(cls, inner_prod: xpd.InnerProduct1D, 
        quad_method: str, *args, **kwargs) -> InnerQuad_Rule:
        """
        """
        if quad_method == "jacobi":
            return InnerQuad_GaussJacobi(inner_prod, *args, **kwargs)
        else:
            return NotImplementedError
