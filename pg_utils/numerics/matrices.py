# -*- coding: utf-8 -*-
"""
Symbolic manipulation and numerical computations 
of the coefficient matrices (mass and stiffness matrices)

The bridge between the symbolic expressions and numerical computations
"""


import warnings
from typing import List, Union, Optional

import sympy
from sympy.integrals import quadrature as qdsym
from ..pg_model import expansion as xpd
from ..pg_model.expansion import xi, n_test, n_trial

import numpy as np
from scipy import special as specfun
from scipy import sparse


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
    Abstract base class for inner product quad evaluators
    """
    
    def __init__(self, inner_prod: xpd.InnerProduct1D) -> None:
        self.inner_prod = inner_prod
        self.int_var = self.inner_prod._int_var
    
    def gramian(self, nrange_trial: List[int], nrange_test: List[int],
        *args, **kwargs) -> Union[np.ndarray, np.matrix, sympy.Matrix]:
        """Calculate Gram matrix, the matrix formed by 
        the inner products of trial and test functions.
        Abstract method to be overriden for actual realization.
        
        Strictly speaking, Gram matrix may be a abuse of terminology,
        as the inner product is usually not in the form <vi, vj>, but
        rather in the form <ui, L(vj)>, i.e. the test and trial functions
        are different, and the second operand may well involve a linear
        operator on the trial expansion.
        
        :param nrange_trial: range of trial functions, an array of int
            indices to be substituted into n_trial
        :param nrange_test: range of test functions, an array of int
            indices to be substituted into n_test
        """
        raise NotImplementedError



class InnerQuad_GaussJacobi(InnerQuad_Rule):
    """Quadrature of inner product following Gauss-Jacobi quadrature
    """
    
    def __init__(self, inner_prod: xpd.InnerProduct1D, automatic: bool = False,
        alpha: Union[float, int, sympy.Expr] = -sympy.Rational(1, 2), 
        beta: Union[float, int, sympy.Expr] = -sympy.Rational(1, 2), 
        quadN: Optional[Union[int, sympy.Expr]] = n_test + n_trial) -> None:
        """Initialization
        
        :param inner_prod: expansion.InnerProduct1D, inner prod to be evaluated
        :param automatic: bool, whether to automatically deduce the orders
            of Jacobi quadrature and the degree of polynomial to be integrated
        :param alpha: float/int/sympy.Expr, preferably sympy.Expr, alpha index
            of Jacobi quadrature. If automatic is True, the kwarg is ignored;
            if automatic is False but kwarg is not explicitly given, default
            is to use the Chebyshev alpha = -1/2
        :param beta: float/int/sympy.Expr, preferably sympy.Expr, beta index.
            Ignored when automatic is True, default to Chebyshev beta = -1/2 
            when automatic deduction is turned off.
        :param quadN: int/sympy.Expr, the quadrature degree. Ignored when
            automatic deduction is True and the quantity not explicitly given, 
            default to n_test + n_trial when automatic deduction turned off.
            When a valid quadN is given, the input will always be used.
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
        if quadN is not None:
            self.quadN = quadN
        
    @classmethod
    def get_powers(cls, int_var: sympy.Symbol, expr: sympy.Expr) -> np.ndarray:
        """Get the powers of p1=(1 - xi), p2=(1 + xi) and xi
        
        :param int_var: sympy.Symbol, integration variable
        :param expr: sympy.Expr, the expression where the powers are retrieved
        """
        p1 = sympy.Symbol(r"p_1", positive=True)
        p2 = sympy.Symbol(r"p_2", positive=True)
        replace_map = {1 - int_var: p1, int_var - 1: -p1, 
            sympy.Rational(1, 2) - int_var/2: p1/2, 
            int_var/2 - sympy.Rational(1, 2): -p1/2, 
            1 + int_var: p2, sympy.Rational(1, 2) + int_var/2: p2/2}
        expr = expr.xreplace(replace_map).expand()
        return powers_of(expr, p1, p2, int_var)
    
    def deduce_params(self, Ntrial: int, Ntest: int):
        """Determine the parameters of the quadrature
        This method is called to determine the values of the parameters
        during evaluation of Gram matrices.
        
        :param Ntrial: int, maximum value for n_trial
        :param Ntest: int, maximum value for n_test
            we assume that the maximum degree of the function to be
            integrated will be reached at maximum n_trial and n_test
        :returns: alpha, beta, quadN
        """
        deduce_map = {n_trial: Ntrial, n_test: Ntest}
        if isinstance(self.alpha, list):
            # If alpha is a list, this means the integrand contains multiple
            # terms, each term with some alpha, beta factors.
            # The final alpha and beta will be given by the minimum of each
            alpha_list = np.array([tmp.subs(deduce_map) for tmp in self.alpha])
            beta_list = np.array([tmp.subs(deduce_map) for tmp in self.beta])
            alpha = alpha_list.min()
            beta = beta_list.min()
            # If alpha and beta of each term in the summation differ by a 
            # non-integer number, this means there will be endpoint singularities
            # that cannot be simultaneously integrated using Gauss-Jacobi quad
            pow_diff = [not term.is_integer for term in alpha_list - alpha] \
                + [not term.is_integer for term in beta_list - beta]
            if np.array(pow_diff).sum() > 0:
                warnings.warn("Incompatible singularities!"
                    "The quadrature cannot integrate exactly."
                    "Trying splitting the inner product instead.")
        else:
            alpha = self.alpha.subs(deduce_map)
            beta = self.beta.subs(deduce_map)
        if isinstance(self.quadN, list):
            quadN = np.array([tmp.subs(deduce_map) for tmp in self.quadN]).max()
        else:
            quadN = self.quadN.subs(deduce_map)
        return alpha, beta, quadN
    
    def gramian(self, nrange_trial: List[int], nrange_test: List[int], 
        backend: str="sympy", int_opt: dict={}, output: str="sympy", out_opt: dict={}, 
        verbose: bool=True) -> Union[np.ndarray, np.matrix, sympy.Matrix]:
        """Compute Gram matrix, concrete realization for Gauss Jacobi quadrature
        
        :param nrange_trial: idx range for trial func, see InnerQuadRule.gramian
        :param nrange_test: idx range for test func, see InnerQuadRule.gramian
        :param backend: str, which backend to use for integration.
            "sympy": the evaluation will be done using sympy evalf
            "scipy": the evaluation will be conducted using numpy/scipy funcs
                the precision will be limited to platform support for np.float
        :param int_opt: dict, kwargs passed to integration function
        :param output: str, which form of matrix to output.
            "sympy": the output will be cast to a sympy.Matrix
            "numpy": the output will be cast to a numpy.ndarray
        :param out_opt: dict, kwargs passed to _output_form method
        """
        alpha, beta, quadN = self.deduce_params(max(nrange_trial), max(nrange_test))
        if verbose:
            print("Integrating with alpha={}, beta={}, N={}".format(alpha, beta, quadN))
        if backend == "sympy":
            M = self._quad_sympy(nrange_trial, nrange_test, 
                alpha, beta, quadN, **int_opt)
        elif backend == "scipy":
            M = self._quad_scipy(nrange_trial, nrange_test, 
                alpha, beta, quadN)
        else:
            raise AttributeError
        return self._output_form(M, output=output, **out_opt)
    
    def _quad_sympy(self, nrange_trial: List[int], nrange_test: List[int], 
        alpha: Union[float, int, sympy.Expr], beta: Union[float, int, sympy.Expr], 
        quad_N: int, precision: int = 16) -> sympy.Matrix:
        """Quadrature using sympy utilities.
        Concrete realization of the Gauss-Jacobi quadrature
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
    
    def _quad_scipy(self, nrange_trial: List[int], nrange_test: List[int], 
        alpha: Union[float, int, sympy.Expr], beta: Union[float, int, sympy.Expr], 
        quad_N: int) -> np.ndarray:
        """Quadrature using scipy utilities
        Concrete realization of the Gauss-Jacobi quadrature
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
                return np.array(M_in).astype(np.complex_)
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



class QuadRecipe:
    
    def __init__(self, init_opt: dict={}, gram_opt: dict={}) -> None:
        self.init_opt = init_opt
        self.gram_opt = gram_opt



class MatrixExpander:
    
    def __init__(self, matrix: xpd.SystemMatrix, quad_recipes: np.ndarray,
        ranges_trial: List, ranges_test: List) -> None:
        assert matrix._matrix.shape == quad_recipes.shape
        assert matrix._matrix.shape == (len(ranges_test), len(ranges_trial))
        self.matrix = matrix
        self.recipe = quad_recipes
        self.n_trials = ranges_trial
        self.n_tests = ranges_test
    
    def expand(self, sparse=False, verbose=False):
        if sparse:
            return self._expand_sparse(verbose=verbose)
        else:
            return self._expand_dense(verbose=verbose)
    
    def _expand_sparse(self, verbose=False) -> sparse.csr_array:
        n_row = sum([len(nrange_test) for nrange_test in self.n_tests])
        n_col = sum([len(nrange_trial) for nrange_trial in self.n_trials])
        return sparse.csr_array((n_row, n_col))
    
    def _expand_dense(self, verbose=False) -> np.ndarray:
        M_list = list()
        for i_row in range(self.matrix._matrix.shape[0]):
            M_row = list()
            for i_col in range(self.matrix._matrix.shape[1]):
                element = self.matrix[i_row, i_col]
                if element is None or element == sympy.S.Zero or element == 0:
                    M_row.append(np.zeros(
                        (len(self.n_tests[i_row]), len(self.n_trials[i_col])), 
                        dtype=np.complex_))
                elif isinstance(element, xpd.InnerProduct1D):
                    if verbose:
                        print("Element (%s, %s)" % 
                            (self.matrix._row_names[i_row], self.matrix._col_names[i_col]))
                    recipe = self.recipe[i_row, i_col]
                    M_tmp = InnerQuad_GaussJacobi(element, **recipe.init_opt)
                    M_tmp = M_tmp.gramian(self.n_trials[i_col], self.n_tests[i_row], 
                            verbose=verbose, **recipe.gram_opt)
                    M_row.append(np.array(M_tmp).astype(np.complex_))
                else:
                    raise NotImplementedError
            M_list.append(M_row)
        return np.block(M_list)
        
