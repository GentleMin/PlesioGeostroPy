# -*- coding: utf-8 -*-
"""
Numerical computations of coefficient matrices (mass and stiffness matrices)

This module aims to compute quadrature of inner product matrices where
.. math::

    M_{ij} = \\sum_k (w_k f_A(NA_i, \\xi_k) f_B(NB_j, \\xi_k))
"""


import warnings
import functools
from typing import List, Union, Optional

import sympy
from sympy.integrals import quadrature as qdsym
from ..pg_model import expansion
from ..pg_model import expansion as xpd
from ..pg_model.expansion import xi, n_test, n_trial

import numpy as np
import mpmath as mp
import gmpy2 as gp
from scipy import special as specfun
from scipy import sparse

from . import special, symparser, utils


def quad_matrix_sympy(operand_A: sympy.Expr, operand_B: sympy.Expr, 
    nrange_A: List[int], nrange_B: List[int],
    xi_quad: List[Union[float, sympy.Number]], 
    wt_quad: List[Union[float, sympy.Number]], 
    n_dps: int = 16) -> sympy.Matrix:
    """Compute quadrature matrix using sympy,
    where :math:`w_k` is `wt_quad[k]`, :math:`\\xi_k` is `xi_quad[k]`,
    :math:`f_A` is `operand_A` and :math:`f_B` is `operand_B`.
    
    :param sympy.Expr operand_A: operand A
    :param sympy.Expr operand_B: operand B
    :param List[int] nrange_A: range of degrees where operand A is evaluated
    :param List[int] nrange_B: range of degrees where operand B is evaluated
    :param List[Union[float, sympy.Number]] xi_quad: x coordinates where
        the operands are evaluated
    :param List[Union[float, sympy.Number]] wt_quad: weights for summation
    :param int dps: number of decimal places to which the calculation will
        be performed
    
    :returns: calculated matrix in sympy.Matrix
    
    .. warning::
    
        There is a known issue that not all functions can be numerically
        evaluated in sympy. For details, see Ingredients document. Therefore,
        other `quad_matrix` functions should be used in favour of sympy version.
    """
    assert len(xi_quad) == len(wt_quad)
    Phi_test = list()
    for n_tmp in nrange_A:
        opd_tmp = operand_A.subs({n_test: n_tmp}).doit()
        Phi_test.append([
            opd_tmp.evalf(n_dps, subs={xi: xi_tmp})
            for xi_tmp in xi_quad
        ])
    Phi_test = np.array(Phi_test, dtype=object)
    Phi_trial = list()
    for n_tmp in nrange_B:
        opd_tmp = operand_B.subs({n_trial: n_tmp}).doit()
        Phi_trial.append([
            opd_tmp.evalf(n_dps, subs={xi: xi_tmp})
            for xi_tmp in xi_quad
        ])
    Phi_trial = np.array(Phi_trial, dtype=object)
    return sympy.Matrix(list((Phi_test*wt_quad) @ Phi_trial.T))

def quad_matrix_scipy(operand_A: sympy.Expr, operand_B: sympy.Expr, 
    nrange_A: List[int], nrange_B: List[int],
    xi_quad: np.ndarray, wt_quad: np.ndarray) -> np.ndarray:
    """Compute quadrature matrix using scipy, 
    where :math:`w_k` is `wt_quad[k]`, :math:`\\xi_k` is `xi_quad[k]`,
    :math:`f_A` is `operand_A` and :math:`f_B` is `operand_B`.
    
    :param sympy.Expr operand_A: operand A
    :param sympy.Expr operand_B: operand B
    :param List[int] nrange_A: range of degrees where operand A is evaluated
    :param List[int] nrange_B: range of degrees where operand B is evaluated
    :param np.ndarray xi_quad: x coordinates where
        the operands are evaluated
    :param np.ndarray wt_quad: weights for summation
    
    :returns: output matrix in np.ndarray
    """
    f_A = sympy.lambdify([n_test, xi], operand_A.doit(), modules=["scipy", "numpy"])
    f_B = sympy.lambdify([n_trial, xi], operand_B.doit(), modules=["scipy", "numpy"])
    Ntest, Xi_test = np.meshgrid(nrange_A, xi_quad, indexing='ij')
    Phi_A = f_A(Ntest, Xi_test)
    Ntrial, Xi_trial = np.meshgrid(nrange_B, xi_quad, indexing='ij')
    Phi_B = f_B(Ntrial, Xi_trial)
    return (Phi_A*wt_quad) @ Phi_B.T


def quad_matrix_mpmath(operand_A: sympy.Expr, operand_B: sympy.Expr, 
    nrange_A: List[int], nrange_B: List[int],
    xi_quad: np.ndarray, wt_quad: np.ndarray, 
    n_dps: int = 33) -> np.ndarray:
    """Compute quadrature matrix using mpmath, 
    where :math:`w_k` is `wt_quad[k]`, :math:`\\xi_k` is `xi_quad[k]`,
    :math:`f_A` is `operand_A` and :math:`f_B` is `operand_B`.
    
    :param sympy.Expr operand_A: operand A
    :param sympy.Expr operand_B: operand B
    :param List[int] nrange_A: range of degrees where operand A is evaluated
    :param List[int] nrange_B: range of degrees where operand B is evaluated
    :param np.ndarray xi_quad: x coordinates where
        the operands are evaluated
    :param np.ndarray wt_quad: weights for summation
    :param int n_dps: decimal places to which the calculation is performed
    
    :returns: output matrix in np.ndarray
    
    .. note::
    
        The input `xi_quad` and `wt_quad`, despite being numpy.ndarray,
        should be arrays of multi-precision `mpmath.mpfr` objs whose prec
        is at least the same as the calculation precision, otherwise the 
        multi-precision calculation is meaningless.
    """
    lambdify_modules = [{
        "jacobi": functools.partial(special.eval_jacobi_recur_mp, dps=n_dps, backend="mpmath"), 
        **symparser.v_functions_mpmath}, 
        "mpmath"
    ]
    f_A = sympy.lambdify([n_test, xi], operand_A.doit(), modules=lambdify_modules)
    f_B = sympy.lambdify([n_trial, xi], operand_B.doit(), modules=lambdify_modules)
    Ntest, Xi_test = np.meshgrid(nrange_A, xi_quad, indexing='ij')
    Phi_A = f_A(Ntest, Xi_test)
    Ntrial, Xi_trial = np.meshgrid(nrange_B, xi_quad, indexing='ij')
    Phi_B = f_B(Ntrial, Xi_trial)
    return (Phi_A*wt_quad) @ Phi_B.T


def quad_matrix_gmpy2(operand_A: sympy.Expr, operand_B: sympy.Expr, 
    nrange_A: List[int], nrange_B: List[int],
    xi_quad: np.ndarray, wt_quad: np.ndarray, 
    n_dps: int = 33) -> np.ndarray:
    """Compute quadrature matrix using gmpy2, 
    where :math:`w_k` is `wt_quad[k]`, :math:`\\xi_k` is `xi_quad[k]`,
    :math:`f_A` is `operand_A` and :math:`f_B` is `operand_B`.
    
    :param sympy.Expr operand_A: operand A
    :param sympy.Expr operand_B: operand B
    :param List[int] nrange_A: range of degrees where operand A is evaluated
    :param List[int] nrange_B: range of degrees where operand B is evaluated
    :param np.ndarray xi_quad: x coordinates where
        the operands are evaluated
    :param np.ndarray wt_quad: weights for summation
    :param int dps: decimal places to which the calculation is performed
    
    :returns: output matrix in np.ndarray
    
    .. note::
    
        The input `xi_quad` and `wt_quad`, despite being numpy.ndarray,
        should be arrays of multi-precision `gmpy2.mpf` objs whose prec
        is at least the same as the calculation precision, otherwise the 
        multi-precision calculation is meaningless.
    """
    lambdify_funcs = [{
        "jacobi": functools.partial(special.eval_jacobi_recur_mp, dps=n_dps, backend="gmpy2"), 
        **symparser.v_functions_gmpy2
    }]
    gmpy2_printer=symparser.Gmpy2Printer(settings={
        'fully_qualified_modules': False, 
        'inline': True, 
        'allow_unknown_functions': True, 
        'user_functions': {"jacobi": "jacobi", "sqrt": "sqrt"}}, prec=112)
    f_A = sympy.lambdify([n_test, xi], operand_A.doit(), 
        modules=lambdify_funcs, printer=gmpy2_printer)
    f_B = sympy.lambdify([n_trial, xi], operand_B.doit(), 
        modules=lambdify_funcs, printer=gmpy2_printer)
    Ntest, Xi_test = np.meshgrid(nrange_A, xi_quad, indexing='ij')
    Phi_A = f_A(Ntest, Xi_test)
    Ntrial, Xi_trial = np.meshgrid(nrange_B, xi_quad, indexing='ij')
    Phi_B = f_B(Ntrial, Xi_trial)
    return (Phi_A*wt_quad) @ Phi_B.T


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
        
        :param List[int] nrange_trial: range of trial functions, an array of int
            indices to be substituted into n_trial
        :param List[int] nrange_test: range of test functions, an array of int
            indices to be substituted into n_test
        """
        raise NotImplementedError


class InnerQuad_GaussJacobi(InnerQuad_Rule):
    """Quadrature of inner product following Gauss-Jacobi quadrature
    
    :param expansion.InnerProduct1D inner_prod: inner prod to be evaluated
    :param sympy.Symbol int_var: integration variable
    :param bool deduce: whether to automatically deduce the indices
    :param sympy.Expr alpha: alpha idx of Jacobi quadrature
    :param sympy.Expr beta: beta idx of Jacobi quadrature
    :param sympy.Expr powerN: total degree to be integrated
    """
    
    def __init__(self, inner_prod: xpd.InnerProduct1D, automatic: bool = False,
        alpha: sympy.Expr = -sympy.Rational(1, 2), 
        beta: sympy.Expr = -sympy.Rational(1, 2), 
        quadN: Optional[sympy.Expr] = None) -> None:
        """Initialization
        
        :param expansion.InnerProduct1D inner_prod: inner prod to be evaluated
        :param bool automatic: whether to automatically deduce the orders
            of Jacobi quadrature and the degree of polynomial to be integrated
        :param sympy.Expr alpha: alpha index of Jacobi quadrature. 
            Ignored when automatic is True, default to Chebyshev alpha = -1/2 
            when automatic deduction is turned off.
        :param sympy.Expr beta: beta index of Jacobi quadrature.
            Ignored when automatic is True, default to Chebyshev beta = -1/2 
            when automatic deduction is turned off.
        :param quadN: no. of quadrature points. Ignored when 
            automatic deduction is True and the quantity not explicitly given, 
            default to n_test + n_trial when automatic deduction turned off.
            When a valid quadN is given, the input will always be used.
        :type quadN: int or sympy.Expr
        """
        super().__init__(inner_prod)
        self.deduce = automatic
        if self.deduce:
            # powers_list = self.get_powers(self.int_var, 
            #     self.inner_prod.integrand().doit())
            powers_left = self.get_powers(self.int_var, 
                self.inner_prod._opd_A.doit())
            powers_right = self.get_powers(self.int_var, 
                (self.inner_prod._wt*self.inner_prod._opd_B).doit())
            if isinstance(powers_left[0], list):
                self.alpha_left = [powers[0] for powers in powers_left]
                self.beta_left = [powers[1] for powers in powers_left]
                powerN_left = [powers[2] for powers in powers_left]
            else:
                self.alpha_left = [powers_left[0]]
                self.beta_left = [powers_left[1]]
                powerN_left = [powers_left[2]]
            if isinstance(powers_right[0], list):
                self.alpha_right = [powers[0] for powers in powers_right]
                self.beta_right = [powers[1] for powers in powers_right]
                powerN_right = [powers[2] for powers in powers_right]
            else:
                self.alpha_right = [powers_right[0]]
                self.beta_right = [powers_right[1]]
                powerN_right = [powers_right[2]]
            # Calculate total degree for all terms in the integrand
            self.alpha = [a_left + a_right 
                for a_left in self.alpha_left for a_right in self.alpha_right]
            self.beta = [b_left + b_right 
                for b_left in self.beta_left for b_right in self.beta_right]
            self.powerN = [N_left + N_right 
                for N_left in powerN_left for N_right in powerN_right]
        else:
            self.alpha, self.beta = [alpha], [beta]
            self.alpha_left, self.alpha_right = [sympy.S.One], [alpha]
            self.beta_left, self.beta_right = [sympy.S.One], [beta]
            self.powerN = 2*quadN - 1
        if quadN is not None:
            self.powerN = 2*quadN - 1
        
    @classmethod
    def get_powers(cls, int_var: sympy.Symbol, expr: sympy.Expr, 
        **kwargs) -> np.ndarray:
        """Get the powers of p1=(1 - xi), p2=(1 + xi) and xi
        
        :param sympy.Symbol int_var: integration variable
        :param sympy.Expr expr: the expression where the powers are retrieved
        :param \**kwargs: whatever that needs to be passed to :func:`powers_of`
        
        For details, please refer to :func:`power_of`
        """
        p1 = sympy.Symbol(r"p_1", positive=True)
        p2 = sympy.Symbol(r"p_2", positive=True)
        replace_map = {1 - int_var: p1, int_var - 1: -p1, 
            sympy.Rational(1, 2) - int_var/2: p1/2, 
            int_var/2 - sympy.Rational(1, 2): -p1/2, 
            1 + int_var: p2, sympy.Rational(1, 2) + int_var/2: p2/2}
        expr = expr.xreplace(replace_map).expand()
        return symparser.powers_of(expr, p1, p2, int_var, **kwargs)
    
    def deduce_params(self, Ntrial: int, Ntest: int):
        """Determine the parameters of the quadrature
        
        This method is called to determine the values of the parameters
        during evaluation of Gram matrices as integration of the full integrand.
        
        :param int Ntrial: maximum value for n_trial
        :param int Ntest: maximum value for n_test
            we assume that the maximum degree of the function to be
            integrated will be reached at maximum n_trial and n_test
        :returns: alpha, beta, quadN
        """
        deduce_map = {n_trial: Ntrial, n_test: Ntest}
        # The integrand contains multiple terms, each term with some alpha, beta factors.
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
        if isinstance(self.powerN, list):
            powerN = [tmp + self.alpha[idx] + self.beta[idx] - alpha - beta
                for idx, tmp in enumerate(self.powerN)]
            powerN = np.array([tmp.subs(deduce_map) for tmp in powerN]).max()
        else:
            powerN = self.powerN.subs(deduce_map)
        quadN = int(powerN) // 2 + 1
        return alpha, beta, quadN
    
    def deduce_params_outer(self, Ntrial: int, Ntest: int):
        """Determine the parameters of the quadrature
        
        This method is called to determine the values of the parameters
        during evaluation of Gram matrices as integration of outer products.
        
        :param int Ntrial: maximum value for n_trial
        :param int Ntest: maximum value for n_test
            we assume that the maximum degree of the function to be
            integrated will be reached at maximum n_trial and n_test
        :returns: alpha, beta, quadN
        """
        deduce_map = {n_trial: Ntrial, n_test: Ntest}
        # Determine the smallest value of alpha and beta in each operand
        alpha_lmin = np.array([tmp.subs(deduce_map) for tmp in self.alpha_left]).min()
        alpha_rmin = np.array([tmp.subs(deduce_map) for tmp in self.alpha_right]).min()
        beta_lmin = np.array([tmp.subs(deduce_map) for tmp in self.beta_left]).min()
        beta_rmin = np.array([tmp.subs(deduce_map) for tmp in self.beta_right]).min()
        alpha_min = alpha_lmin + alpha_rmin
        beta_min = beta_lmin + beta_rmin
        # If alpha and beta of each term in the summation differ by a 
        # non-integer number, this means there will be endpoint singularities
        # that cannot be simultaneously integrated using one G-J quadrature
        for i_idx in range(len(self.alpha)):
            alpha_diff = self.alpha[i_idx].subs(deduce_map) - alpha_min
            beta_diff = self.beta[i_idx].subs(deduce_map) - beta_min
            if not alpha_diff.is_integer or not beta_diff.is_integer:
                warnings.warn("Incompatible singularities!"
                    "The quadrature cannot integrate exactly."
                    "Trying splitting the inner product instead.")
                break
        # alpha_list = np.array([tmp.subs(deduce_map) for tmp in self.alpha])
        # beta_list = np.array([tmp.subs(deduce_map) for tmp in self.beta])
        # pow_diff = [not term.is_integer for term in alpha_list - alpha_min] \
        #     + [not term.is_integer for term in beta_list - beta_min]
        # if np.array(pow_diff).sum() > 0:
        #     warnings.warn("Incompatible singularities!"
        #         "The quadrature cannot integrate exactly."
        #         "Trying splitting the inner product instead.")
        if isinstance(self.powerN, list):
            powerN = [tmp + self.alpha[idx] + self.beta[idx] - alpha_min - beta_min
                for idx, tmp in enumerate(self.powerN)]
            powerN = np.array([tmp.subs(deduce_map) for tmp in powerN]).max()
        else:
            powerN = self.powerN.subs(deduce_map)
        quadN = int(powerN) // 2 + 1
        return alpha_min, beta_min, quadN, alpha_lmin, beta_lmin
    
    def gramian(self, nrange_trial: List[int], nrange_test: List[int], 
        backend: str="sympy", int_opt: dict={}, output: str="sympy", out_opt: dict={}, 
        outer: bool=True, verbose: bool=True) -> Union[np.ndarray, np.matrix, sympy.Matrix]:
        """Compute Gram matrix, concrete realization for Gauss Jacobi quadrature
        
        This is the main interface for calculating the inner product matrix;
        it is a conglomerate of different methods with different options.
        
        :param List[int] nrange_trial: idx range for trial func, see InnerQuadRule.gramian
        :param List[int] nrange_test: idx range for test func, see InnerQuadRule.gramian
        :param str backend: which backend to use for integration.
            * "sympy": the evaluation will be done using sympy evalf
            * "scipy": the evaluation will be conducted using numpy/scipy funcs
                the precision will be limited to platform support for np.float
            * "mpmath": multi-precision evaluation with mpmath
            * "gmpy2": multi-precision evaluation with gmpy2
            
        :param dict int_opt: kwargs passed to integration function
        :param str output: which form of matrix to output.
            * "sympy": the output will be cast to a sympy.Matrix
            * "numpy": the output will be cast to a numpy.ndarray
            
        :param dict out_opt: kwargs passed to _output_form method
        :param bool outer: whether to use outer product formulation
        """
        if outer:
            alpha, beta, quadN, alpha_l, beta_l = self.deduce_params_outer(
                max(nrange_trial), max(nrange_test))
        else:
            alpha, beta, quadN = self.deduce_params(max(nrange_trial), max(nrange_test))
        
        # Throw warning of singularity if alpha or beta <= -1
        if alpha <= -1:
            alpha = 0
            quadN *= 2
            warnings.warn("Endpoint singularity detected! Check for integrability!")
        if beta <= -1:
            beta = 0
            quadN *= 2
            warnings.warn("Endpoint singularity detected! Check for integrability!")
        
        if verbose:
            print("Integrating with alpha={}, beta={}, N={}".format(alpha, beta, quadN))
        if backend == "sympy":
            if outer:
                M = self._quad_sympy_outer(nrange_trial, nrange_test, 
                    alpha, beta, quadN, alpha_l, beta_l, **int_opt)
            else:
                M = self._quad_sympy_integrand(nrange_trial, nrange_test, 
                    alpha, beta, quadN, **int_opt)
        elif backend == "scipy":
            if outer:
                M = self._quad_scipy_outer(nrange_trial, nrange_test,
                    alpha, beta, quadN, alpha_l, beta_l)
            else:
                M = self._quad_scipy_integrand(nrange_trial, nrange_test, 
                    alpha, beta, quadN)
        elif backend == "mpmath":
            M = self._quad_mpmath(nrange_trial, nrange_test, 
                alpha, beta, quadN, alpha_l, beta_l, **int_opt)
        elif backend == "gmpy2":
            M = self._quad_gmpy2(nrange_trial, nrange_test, 
                alpha, beta, quadN, alpha_l, beta_l, **int_opt)
        else:
            raise AttributeError
        return self.output_form(M, output=output, **out_opt)
    
    def _quad_sympy_integrand(self, nrange_trial: List[int], nrange_test: List[int], 
        alpha: Union[float, int, sympy.Expr], beta: Union[float, int, sympy.Expr], 
        quad_N: int, n_dps: int = 16) -> sympy.Matrix:
        """Concrete Gauss-Jacobi quad: sympy evaluation of integrand
        """
        xi_quad, wt_quad = qdsym.gauss_jacobi(quad_N, alpha, beta, n_dps)
        integrand = self.inner_prod.integrand()/(1 - xi)**alpha/(1 + xi)**beta
        
        M = list()
        for n_test_val in nrange_test:
            M_row = list()
            for n_trial_val in nrange_trial:
                int_tmp = integrand.subs({n_test: n_test_val, n_trial: n_trial_val}).doit()
                # quad_pts = [wt_quad[i]*int_tmp.subs({xi: xi_quad[i]}, n=precision) 
                #     for i in range(quad_N)]
                # The old version (line above) seems to completely ignore the precision?
                quad_pts = [wt_quad[i]*int_tmp.evalf(n_dps, subs={xi: xi_quad[i]})
                    for i in range(quad_N)]
                M_row.append(sum(quad_pts))
            M.append(M_row)
        return sympy.Matrix(M)
    
    def _quad_sympy_outer(self, nrange_trial: List[int], nrange_test: List[int], 
        alpha: Union[float, int, sympy.Expr], beta: Union[float, int, sympy.Expr], 
        quad_N: int, alpha_left: Union[float, int, sympy.Expr], 
        beta_left: Union[float, int, sympy.Expr], 
        n_dps: int = 16) -> sympy.Matrix:
        """Concrete Gauss-Jacobi quad: sympy evaluation of outer product
        """
        xi_quad, wt_quad = qdsym.gauss_jacobi(quad_N, alpha, beta, n_dps)
        opd_A = self.inner_prod._opd_A/(1 - xi)**alpha_left/(1 + xi)**beta_left
        opd_B = self.inner_prod._opd_B*self.inner_prod._wt/(1 - xi)**(alpha - alpha_left)/(1 + xi)**(beta - beta_left)
        return quad_matrix_sympy(opd_A, opd_B, nrange_test, nrange_trial, 
            xi_quad, wt_quad, n_dps=n_dps)
        # Phi_test = list()
        # for n_test_val in nrange_test:
        #     opd_tmp = opd_A.subs({n_test: n_test_val}).doit()
        #     Phi_test.append([
        #         # opd_tmp.subs({xi: xi_quad[i]}, n=precision)
        #         opd_tmp.evalf(n_dps, subs={xi: xi_quad[i]})
        #         for i in range(quad_N)
        #     ])
        # Phi_test = np.array(Phi_test, dtype=object)
        # Phi_trial = list()
        # for n_trial_val in nrange_trial:
        #     opd_tmp = opd_B.subs({n_trial: n_trial_val}).doit()
        #     Phi_trial.append([
        #         # opd_tmp.subs({xi: xi_quad[i]}, n=precision)
        #         opd_tmp.evalf(n_dps, subs={xi: xi_quad[i]})
        #         for i in range(quad_N)
        #     ])
        # Phi_trial = np.array(Phi_trial, dtype=object)
        # return sympy.Matrix(list((Phi_test*wt_quad) @ Phi_trial.T))
    
    def _quad_scipy_integrand(self, nrange_trial: List[int], nrange_test: List[int], 
        alpha: Union[float, int, sympy.Expr], beta: Union[float, int, sympy.Expr], 
        quad_N: int) -> np.ndarray:
        """Concrete Gauss-Jacobi quad: scipy evaluation of integrand
        """
        xi_quad, wt_quad = specfun.roots_jacobi(int(quad_N), float(alpha), float(beta))
        integrand = self.inner_prod.integrand()/(1 - xi)**alpha/(1 + xi)**beta
        int_fun = sympy.lambdify([n_test, n_trial, xi], integrand.doit(), 
            modules=["scipy", "numpy"])
        Ntest, Ntrial, Xi = np.meshgrid(nrange_test, nrange_trial, xi_quad, indexing='ij')
        return np.sum(int_fun(Ntest, Ntrial, Xi)*wt_quad, axis=-1)
    
    def _quad_scipy_outer(self, nrange_trial: List[int], nrange_test: List[int], 
        alpha: Union[float, int, sympy.Expr], beta: Union[float, int, sympy.Expr], quad_N: int,
        alpha_left: Union[float, int, sympy.Expr], beta_left: Union[float, int, sympy.Expr]) -> np.ndarray:
        """Concrete Gauss-Jacobi quad: scipy evaluation of outer product
        """
        xi_quad, wt_quad = specfun.roots_jacobi(int(quad_N), float(alpha), float(beta))
        opd_A = self.inner_prod._opd_A/(1 - xi)**alpha_left/(1 + xi)**beta_left
        opd_B = self.inner_prod._opd_B*self.inner_prod._wt/(1 - xi)**(alpha - alpha_left)/(1 + xi)**(beta - beta_left)
        return quad_matrix_scipy(opd_A, opd_B, nrange_test, nrange_trial, xi_quad, wt_quad)
        # opd_A = sympy.lambdify([n_test, xi], opd_A.doit(), modules=["scipy", "numpy"])
        # opd_B = sympy.lambdify([n_trial, xi], opd_B.doit(), modules=["scipy", "numpy"])
        # Ntest, Xi_test = np.meshgrid(nrange_test, xi_quad, indexing='ij')
        # Phi_test = opd_A(Ntest, Xi_test)
        # Ntrial, Xi_trial = np.meshgrid(nrange_trial, xi_quad, indexing='ij')
        # Phi_trial = opd_B(Ntrial, Xi_trial)
        # return (Phi_test*wt_quad) @ Phi_trial.T
    
    def _quad_mpmath(self, nrange_trial: List[int], nrange_test: List[int], 
        alpha: sympy.Expr, beta: sympy.Expr, quad_N: int, 
        alpha_left: sympy.Expr, beta_left: sympy.Expr, 
        n_dps: int = 33) -> np.ndarray:
        """Concrete Gauss-Jacobi quad: multi-prec evaluation with mpmath
        """
        opd_A = self.inner_prod._opd_A/(1 - xi)**alpha_left/(1 + xi)**beta_left
        opd_B = self.inner_prod._opd_B*self.inner_prod._wt/(1 - xi)**(alpha - alpha_left)/(1 + xi)**(beta - beta_left)
        with mp.workdps(n_dps):
            alpha_mp = mp.mpf(str(alpha.evalf(n_dps)))
            beta_mp = mp.mpf(str(beta.evalf(n_dps)))
        root_result = special.roots_jacobi_mp(int(quad_N), alpha_mp, beta_mp, n_dps=n_dps)
        xi_quad, wt_quad = root_result.xi, root_result.wt
        return quad_matrix_mpmath(opd_A, opd_B, nrange_test, nrange_trial, 
            xi_quad, wt_quad, n_dps=n_dps)
        
    def _quad_gmpy2(self, nrange_trial: List[int], nrange_test: List[int], 
        alpha: sympy.Expr, beta: sympy.Expr, quad_N: int, 
        alpha_left: sympy.Expr, beta_left: sympy.Expr, 
        n_dps: int = 33) -> np.ndarray:
        """Concrete Gauss-Jacobi quad: multi-prec evaluation with gmpy2
        """
        opd_A = self.inner_prod._opd_A/(1 - xi)**alpha_left/(1 + xi)**beta_left
        opd_B = self.inner_prod._opd_B*self.inner_prod._wt/(1 - xi)**(alpha - alpha_left)/(1 + xi)**(beta - beta_left)
        with mp.workdps(n_dps):
            alpha_mp = mp.mpf(str(alpha.evalf(n_dps)))
            beta_mp = mp.mpf(str(beta.evalf(n_dps)))
        root_result = special.roots_jacobi_mp(int(quad_N), alpha_mp, beta_mp, n_dps=n_dps)
        xi_quad = utils.to_gpmy2_f(root_result.xi, dps=n_dps) 
        wt_quad = utils.to_gpmy2_f(root_result.wt, dps=n_dps)
        return quad_matrix_gmpy2(opd_A, opd_B, nrange_test, nrange_trial, 
            xi_quad, wt_quad, n_dps=n_dps)
    
    def output_form(self, M_in: Union[np.ndarray, sympy.Matrix], 
        output: str = "sympy", **kwargs) -> Union[np.ndarray, sympy.Matrix]:
        """Cast output matrix to desired form and data types
        """
        if output == "sympy":
            return sympy.nsimplify(M_in, **kwargs)
        elif output == "numpy":
            if isinstance(M_in, sympy.Matrix):
                return np.array(M_in).astype(np.complex128)
            elif isinstance(M_in, np.ndarray):
                return M_in.astype(np.complex128)
        elif output == "gmpy2":
            # print(M_in)
            return utils.to_gpmy2_c(np.array(M_in), **kwargs)
        elif output == "none":
            return M_in
        else:
            raise AttributeError



class InnerProdQuad:
    """Quadrature of inner product
    class generator for all inner product quadratures in 1D
    
    Compared to the direct quadratures of the integral form,
    calculating quadratures in the notation of inner products
    "allows" one to drastically save of time of basis evaluation.
    When calculating the integral in the form of::
    
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



class LabeledBlockArray:
    """Block 1-D array with labels assigned to blocks
    
    LabeledBlockMatrix wraps around an array, so that blocks of it
    can be accessed using a single string
    
    :param _block_idx: dict, key=label(str) -> value=indices of the block (slice)
    :param _array: np.ndarray, the underlying matrix
    """
    
    def __init__(self, array: np.ndarray, 
        block_names: List[str], block_ranges: List[int]) -> None:
        """
        :param block_names: array-like, names of the blocks
        :param block_ranges: array of integers, the size of each block
        
        ..Example: for instance, 
            LabeledBlockArray(np.arange(10), ["A", "B", "C"], [3, 4, 3])
            will be interpreted in the following sense:
            [---A---|-----B------|---C---]
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        """
        assert len(block_names) == len(block_ranges)
        idx_sums = np.r_[0, np.cumsum(block_ranges)]
        self._block_idx = {name: slice(idx_sums[idx], idx_sums[idx+1])
            for idx, name in enumerate(block_names)}
        self._array = array
    
    def __getitem__(self, key):
        if isinstance(key, str):
            key = self._block_idx[key]
        return self._array.__getitem__(key)
    
    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = self._block_idx[key]
        self._array.__setitem__(key, value)



class LabeledBlockMatrix:
    """Block matrix with labels assigned to row & col blocks
    
    Unlike functions e.g. numpy.block, LabeledBlockMatrix assumes the
    matrix are segmented into block separated by fixed grid lines, i.e::
    
        AAA BB
        AAA BB
        CCC DD
    
    but not in the forms of::
    
        AAA BB
        AAA BB
        CC DDD
    
    or::
    
        AAA BB
        CCC BB
        CCC DD
        
    and so each block can be determined by one row and one column idx.
    
    :ivar dict _row_idx: key=label(str) -> value=row indices of the block (slice)
    :ivar dict _col_idx: key=label(str) -> value=col indices of the block (slice)
    :ivar np.ndarray _matrix: the underlying matrix
    """
    
    def __init__(self, matrix: np.ndarray, 
        row_names: List[str], row_ranges: List[int], 
        col_names: List[str], col_ranges: List[int]) -> None:
        """Initialization
        
        :param array-like row_names: names of the row blocks
        :param row_ranges: array of integers, the size of each block in 
            number of rows.
        :param array-like col_names: names of the col blocks
        :param col_ranges: array of integers, the size of each block in 
            number of cols.
        """
        assert matrix.shape == (sum(row_ranges), sum(col_ranges))
        assert len(row_names) == len(row_ranges)
        assert len(col_names) == len(col_ranges)
        idx_sums = np.r_[0, np.cumsum(row_ranges)]
        self._row_idx = {row_name: slice(idx_sums[idx], idx_sums[idx+1]) 
            for idx, row_name in enumerate(row_names)}
        idx_sums = np.r_[0, np.cumsum(col_ranges)]
        self._col_idx = {col_name: slice(idx_sums[idx], idx_sums[idx+1]) 
            for idx, col_name in enumerate(col_names)}
        self._matrix = matrix
    
    def __getitem__(self, key):
        if isinstance(key, str):
            key = self._row_idx[key]
        elif isinstance(key, tuple):
            key = (self._row_idx[key[0]] if isinstance(key[0], str) else key[0], 
                   self._col_idx[key[1]] if isinstance(key[1], str) else key[1])
        return self._matrix.__getitem__(key)
    
    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = self._row_idx[key]
        elif isinstance(key, tuple):
            key = (self._row_idx[key[0]] if isinstance(key[0], str) else key[0], 
                   self._col_idx[key[1]] if isinstance(key[1], str) else key[1])
        self._matrix.__setitem__(key, value)



class MatrixExpander:
    """Evaluation class for expanding system matrices 
    with InnerProduct1D elements into actual numerical matrices.
    
    :ivar expansion.SystemMatrix matrix: a system matrix with either zero
        or InnerProduct1D as elements.
    :ivar np.ndarray recipe: collection of QuadRecipe objects.
    :ivar n_trials: ranges of trial functions for the expansion
    :ivar n_tests: ranges of test functions for the expansion
    """
    
    def __init__(self, matrix: xpd.SystemMatrix, quad_recipes: np.ndarray,
        ranges_trial: List, ranges_test: List) -> None:
        """Initialization
        
        :param expansion.SystemMatrix matrix: a system matrix with either zero
            or InnerProduct1D as elements. Ideally in the future, this should
            also accept matrices containing elements written as a sum of inner
            products, for robustness.
        :param np.ndarray quad_recipes: collection of QuadRecipe objects.
        :param ranges_trial: ranges of trial functions for the expansion
        :param ranges_test: ranges of test functions for the expansion
        """
        assert matrix._matrix.shape == quad_recipes.shape
        assert matrix._matrix.shape == (len(ranges_test), len(ranges_trial))
        self.matrix = matrix
        self.recipe = quad_recipes
        self.n_trials = ranges_trial
        self.n_tests = ranges_test
    
    def expand(self, sparse=False, verbose=False):
        """Expand the matrix according to the recipes
        """
        if sparse:
            return self._expand_sparse(verbose=verbose)
        else:
            return self._expand_dense(verbose=verbose)
    
    def _expand_sparse(self, verbose=False) -> sparse.csr_array:
        """Form a sparse matrix during the expansion
        """
        raise NotImplementedError
        n_row = sum([len(nrange_test) for nrange_test in self.n_tests])
        n_col = sum([len(nrange_trial) for nrange_trial in self.n_trials])
        return sparse.csr_array((n_row, n_col))
    
    def _expand_dense(self, verbose=False) -> np.ndarray:
        """Form a dense matrix during the expansion
        """
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
                    M_row.append(np.asarray(M_tmp))
                else:
                    raise NotImplementedError
            M_list.append(M_row)
        return np.block(M_list)


def sparsify(array: np.ndarray, 
    clip_threshold: Union[float, np.float64] = np.finfo(np.float64).eps
    ) -> sparse.coo_array:
    """Create sparse array from dense array
    """
    sparse_array = array.copy()
    sparse_array[np.abs(sparse_array) <= clip_threshold] = 0.
    return sparse.coo_array(sparse_array)


def invert_block_diag(matrix: np.ndarray, block_seg: List[int]) -> np.ndarray:
    """Invert a block diagonal matrix
    
    The matrix is assumed to be square matrix,
    and the block_seg gives the segmentation of the blocks on the diagonal.
    
    Not yet implemented.
    """
    assert matrix.shape[0] == matrix.shape[1]
