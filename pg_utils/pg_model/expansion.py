# -*- coding: utf-8 -*-
"""
Define expansions of the fields.
The methods and classes in this file are mainly for expanding fields
in polar coordinates, in the unit disk (s in (0, 1), phi in (0, pi)).

These utilities depend on coordinate variables in the PG formulation,
but are not directly linked to PG variables.
As a result, classes and methods in this file will no longer be bound
to use with PG-formulation-specific classes such as CollectionPG,
but work with general LabeledCollection class and its instances.

This is mostly for flexible implementation.
For instance, in eigenvalue problems, the radial magnetic field at the
boundary and the corresponding evolution equation is often replaced
by B-fields in cylindrical coordinates at the boundary; Lorentz force
terms may be combined into one single dynamical variable, etc.
"""


from typing import Any, List, Optional, Union, Callable, TextIO
import sympy
from sympy import Rational, jacobi

from . import base
from .core import s, H_s
import numpy as np

import json



#: Integers denoting the indices of the radial and azimuthal bases
n, m = sympy.symbols("n, m", integer=True)

#: Index for the radial test functions
n_test = sympy.Symbol(r"\ell'", integer=True)
#: Index for the radial trial functions 
n_trial = sympy.Symbol(r"\ell", integer=True)

#: Angular frequency
omega = sympy.Symbol(r"\omega")

#: Integration variable for the radial direction
xi = sympy.Symbol(r"\xi")
#: :data:`xi` as a function of :data:`s`
xi_s = 2*s**2 - 1
#: :data:`s` as a function of :data:`xi`
s_xi = sympy.sqrt((xi + 1)/2)



class SpectralExpansion(base.LabeledCollection):
    """Base class for different kinds of spectral expansions
    
    This is intended as an abstract class, which should be overriden
    by its concrete child classes
    
    :param base.LabeledCollection fields: fields to be expanded
    :param Union[base.LabeledCollection, sympy.Expr] bases: bases to be used
    :param base.LabeledCollection coeffs: collection of the corresponding coeffs.
    """
    
    def __init__(self, fields: base.LabeledCollection, 
        bases: Union[base.LabeledCollection, sympy.Expr], 
        coeffs: base.LabeledCollection, 
        **relations: sympy.Expr) -> None:
        """Initialization
        
        :param base.LabeledCollection fields: fields to be expanded
        :param Union[base.LabeledCollection, sympy.Expr] bases: bases to be used
        :param base.LabeledCollection coeffs: collection of the corresponding coeffs.
        """
        super().__init__(fields._field_names, **relations)
        self.fields, self.bases, self.coeffs = fields, bases, coeffs
    


class FourierExpansions(SpectralExpansion):
    """Fourier expansion for fields
    
    This class can be used for reducing equations or expressions into the
    Fourier domain given specific arguments. The Fourier-domain eqs/
    exprs are then processed using `RadialExpansions`
    
    :param fields: fields to use the Fourier expansion
    :param bases: the current basis would be the complex Fourier basis
        Note: unlike RadialExpansions, the Fourier basis is assumed to
        be shared among all fields. From another view, this basis has
        the same form in the expansion of different fields.
    :param coeffs: Fourier coefficients
    :param ansatz: the Fourier ansatz
    """
        
    def __init__(self, argument: sympy.Expr, fields: base.LabeledCollection, 
        fourier_coeffs: base.LabeledCollection) -> None:
        """Initiate Fourier expansion
        
        :param sympy.Expr arguments: the complex argument of the Fourier basis
        :param base.LabeledCollection fields: fields to use the Fourier expansion on
        :param base.LabeledCollection fourier_coeffs: Fourier coefficients; 
            these should be Fourier coefficients of the fields specified 
            by `fields` expanded with argument specified by `argument`.
        
        .. note::
        
            If `fields` are functions of `s`, `p`, `z` and `t`, and `argument`
            takes the form ``omega*t + m*p``, then `fourier_coeffs` should
            be functions of `s` and `z`.
        """
        assert fields._field_names == fourier_coeffs._field_names
        fourier_basis = sympy.exp(sympy.I*argument)
        map_tmp = self._build_fourier_map(fields, fourier_basis, fourier_coeffs)
        super().__init__(fields, fourier_basis, fourier_coeffs, **map_tmp)
    
    @staticmethod
    def _build_fourier_map(fields: base.LabeledCollection, 
        fourier_basis: base.LabeledCollection, 
        fourier_coeffs: base.LabeledCollection):
        """Create mapping from fields to their Fourier-domain expressions
        """
        fourier_map = {
            fields._field_names[idx]: fourier_coeffs[idx]*fourier_basis
            for idx in range(fields.n_fields)
            if fourier_coeffs[idx] is not None}
        return fourier_map
    
    @staticmethod
    def to_fourier_domain(expr, ansatz, basis):
        """Convert an expression to Fourier domain
        
        :param sympy.Expr expr: expression to be converted
        :param dict ansatz: a substitution that maps fields to their Fourier ansatz
        :param basis: the basis used in the ansatz
        
        :returns: expression in the Fourier domain
        """
        expr_fourier = expr.subs(ansatz).doit()/basis
        return expr_fourier



class RadialExpansions(SpectralExpansion):
    """Radial expansions, a collection of radial expansions 
    for each field in the collection of dynamical variables.
    
    In many (even most) spectral expansions, 
    each field, or each component of the tensor are expanded using the 
    same bases, only with different coefficients.
    
    For PG applications, two complications arise:
    
    * As we expand the elements of tensors, it would be desirable to
      implement different forms of bases for each individual field, so
      that the regularity constraints are fulfilled;
    * we potentially have a mixture of quantities expressed in 
      cylindrical and spherical coordinates.
    
    The first point is tackled by using different bases and different
    relations, hence this class assumes a collection of basis.
    """
    
    def __init__(self, fields: base.LabeledCollection, 
        bases: base.LabeledCollection, 
        coeffs: base.LabeledCollection, 
        **relations: sympy.Expr) -> None:
        super().__init__(fields, bases, coeffs, **relations)



class RadialTestFunctions(base.LabeledCollection):
    """Collection of radial functions used as test functions for reducing
    differential equations into algebraic form.
    """
    
    def __init__(self, names: List[str], **test_functions: sympy.Expr) -> None:
        super().__init__(names, **test_functions)
        


class InnerProduct1D(sympy.Expr):
    """1-D (integration) inner product :math:`\\langle a, b \\rangle`
    
    :ivar sympy.Expr _opd_A: left operand
    :ivar sympy.Expr _opd_B: right operand
    :ivar sympy.Expr _wt: weight
    :ivar sympy.Symbol _int_var: integration variable
    :ivar List[sympy.Expr] _bound: integration bounds
    """
    
    def __init__(self, opd_A: sympy.Expr, opd_B: sympy.Expr, wt: sympy.Expr, 
        int_var: sympy.Symbol, lower: sympy.Expr, upper: sympy.Expr) -> None:
        """Initialization
        
        :param sympy.Expr opd_A: left operand
        :param sympy.Expr opd_B: right operand
        :param sympy.Expr wt: weight
        :param sympy.Symbol int_var: integration variable
        :param sympy.Expr lower: lower bound of integration
        :param sympy.Expr upper: upper bound of integration
        """
        self._opd_A = opd_A
        self._opd_B = opd_B
        self._wt = wt
        self._int_var = int_var
        self._bound = (lower, upper)
    
    def _latex(self, printer, *args):
        str_A = printer._print(self._opd_A, *args)
        str_B = printer._print(self._wt*self._opd_B, *args)
        str_var = printer._print(self._int_var, *args)
        return r"\left\langle %s \, , \, %s \right\rangle_{%s}" % (
            str_A, str_B, str_var)
    
    def doit(self, **hints):
        """Return un-evaluated integral form
        """
        if hints.get("integral", False):
            return self.integral_form()
    
    def integrand(self):
        """Get the explicit form of the integrand
        """
        return self._opd_A*self._opd_B*self._wt
    
    def integral_form(self):
        """Get the explicit integral form
        """
        return sympy.Integral(
            self._opd_A*self._opd_B*self._wt, 
            (self._int_var, self._bound[0], self._bound[1]))
    
    def change_variable(self, new_var: sympy.Symbol, 
        int_var_expr: sympy.Expr, inv_expr: sympy.Expr, 
        jac_positive: bool = True, merge: bool = False, simplify: bool = False) -> "InnerProduct1D":
        """Change the integration variable
        
        :param sympy.Symbol new_var: the new variable to be integrated over
        :param sympy.Expr int_var_expr: the current variable expressed in new variable
        :param sympy.Expr inv_expr: inverse expression of `int_var_expr`
            one needs to explicitly state the inverse expression
        :param bool jac_positive: whether to assume the Jacobian is positive,
            default is `True`. If False, the absolute value of the Jacobian will
            be taken before taking the integral
        :param bool merge: whether to merge the weight with the second operand.
            Default is `False`.
            Although not reasonable in the definition of inner products, it can be
            useful when the inner product is converted to integral form in the end.
        :param bool simplify: whether to simplify the operands
        
        :returns: `InnerProduct1D` object with changed integration variable
        """
        if jac_positive:
            jac = sympy.diff(int_var_expr, new_var).doit()
        else:
            jac = sympy.Abs(sympy.diff(int_var_expr, new_var).doit())
        opd_A = self._opd_A.subs({self._int_var: int_var_expr})
        opd_B = self._opd_B.subs({self._int_var: int_var_expr})
        wt = jac*self._wt.subs({self._int_var: int_var_expr})
        if merge:
            opd_B = wt*opd_B
            wt = sympy.S.One
        if simplify:
            opd_A = opd_A.simplify()
            opd_B = opd_B.simplify()
            wt = wt.simplify()
        new_inner_prod = InnerProduct1D(
            opd_A, opd_B, wt, new_var, 
            inv_expr.subs({self._int_var: self._bound[0]}).doit(),
            inv_expr.subs({self._int_var: self._bound[1]}).doit())
        return new_inner_prod
    
    def commute_factor_out(self, term: sympy.Expr, opd: int = 1) -> sympy.Expr:
        """Move a factor out of the inner product
        
        A manual operation to move a factor from an operand to outside the
        inner product. Realization of "linearity"
        
        :param sympy.Expr term: the factor to be moved out
        :param int opd: index for the operand. Either 0 or 1, default = 1.
            If `opd` = 0, the factor will be moved out of the first operand;
            if `opd` = 1, the factor will be moved out of the second operand
        """
        if opd == 0:
            return term*InnerProduct1D(self._opd_A/term, self._opd_B, self._wt, 
                self._int_var, self._bound[0], self._bound[1])
        elif opd == 1:
            return term*InnerProduct1D(self._opd_A, self._opd_B/term, self._wt, 
                self._int_var, self._bound[0], self._bound[1])
        else:
            raise AttributeError
    
    def commute_factor_in(self, term: sympy.Expr, opd: int = 1) -> "InnerProduct1D":
        """Move a factor into the inner product
        
        A manual operation to move a factor from outside the inner product
        to its operand. Realization of "linearity"
        
        :param sympy.Expr term: the factor to be moved in
        :param int opd: index for the operand. Either 0 or 1, default = 1.
            If `opd` = 0, the factor will be moved to the first operand;
            if `opd` = 1, the factor will be moved to the second operand
        """
        if opd == 0:
            return InnerProduct1D(term*self._opd_A, self._opd_B, self._wt, 
                self._int_var, self._bound[0], self._bound[1])
        elif opd == 1:
            return InnerProduct1D(self._opd_A, term*self._opd_B, self._wt, 
                self._int_var, self._bound[0], self._bound[1])
        else:
            raise AttributeError
    
    def split(self, opd: int = 1) -> sympy.Expr:
        """Split the inner product into sum of inner products.
        
        A manual operation of the distributive property of inner products to 
        split the inner product whose argument is a sum of several terms
        into the sum of several inner products.
        """
        if opd == 0 and isinstance(self._opd_A, sympy.Add):
            return sympy.Add(**[InnerProduct1D(
                arg, self._opd_B, self._wt, self._int_var, self._bound[0], self._bound[1])
                for arg in self._opd_A.args])
        elif opd == 1 and isinstance(self._opd_B, sympy.Add):
            return sympy.Add(**[InnerProduct1D(
                self._opd_A, arg, self._wt, self._int_var, self._bound[0], self._bound[1])
                for arg in self._opd_B.args])
        else:
            return self
    
    def serialize(self) -> dict:
        """Serialize inner product to a dictionary
        """
        return {"opd_A": sympy.srepr(self._opd_A), 
                "opd_B": sympy.srepr(self._opd_B), "wt": sympy.srepr(self._wt), 
                "int_var": sympy.srepr(self._int_var), 
                "lower": sympy.srepr(self._bound[0]), 
                "upper": sympy.srepr(self._bound[1])}



class InnerProductOp1D:
    """Inner product defined on 1-D function space :math:`\\langle \cdot, \cdot \\rangle`
    
    :ivar sympy.Symbol _int_var: integration variable
    :ivar sympy.Expr _wt: weight
    :ivar List[sympy.Expr] _bound: integration bound, 2-tuple
    :ivar _conj: which argument to perform the complex conjugation
    :vartype _conj: int or None
    """
    
    def __init__(self, int_var: sympy.Symbol, wt: sympy.Expr, 
        bound: List, conj: Optional[int] = None) -> None:
        """Initialization
        
        :param sympy.Symbol int_var: integration variable
        :param sympy.Expr wt: weight
        :param List[sympy.Expr] bound: integration bound, 2-tuple
        :param Optional[int] conj: which argument to perform the complex conjugation.
            Default to `None`.
            If `conj` = `None`, no complex conjugate will be taken;
            if `conj` = 0, the first operand will be taken conjugate of; 
            if `conj` = 1, the second operand will be taken conjugate of.
        """
        assert len(bound) == 2
        if isinstance(conj, int):
            assert conj in (0, 1)
        self._int_var = int_var
        self._wt = wt
        self._bound = bound
        self._conj = conj
    
    def __call__(self, opd_A: sympy.Expr, opd_B: sympy.Expr) -> InnerProduct1D:
        """Generate an InnerProduct1D from two operands
        """
        if self._conj is None:
            return InnerProduct1D(opd_A, opd_B, self._wt, 
                self._int_var, self._bound[0], self._bound[1])
        elif self._conj == 0:
            return InnerProduct1D(opd_A.conjugate(), opd_B, self._wt, 
                self._int_var, self._bound[0], self._bound[1])
        elif self._conj == 1:
            return InnerProduct1D(opd_A, opd_B.conjugate(), self._wt, 
                self._int_var, self._bound[0], self._bound[1])



class RadialInnerProducts(base.LabeledCollection):
    """Collection of inner products
    """
    
    def __init__(self, names: List[str], **inner_prod_op: InnerProductOp1D) -> None:
        super().__init__(names, **inner_prod_op)



class ExpansionRecipe:
    """The top-level class used for spectral expansion,
    which defines the recipe concerning radial and azimuthal expansions.
    
    :ivar str identifier: name of the expansion
    :ivar FourierExpansion fourier_xpd: defines the fourier expansion part
        this is considered shared among all fields
    :ivar RadialExpansion rad_xpd: defines the radial expansion
    :ivar RadialTestFunctions rad_test: test functions for each radial equation
    :ivar RadialInnerProducts inner_prod_op: inner products associated with each eq
    :ivar dict base_expr: explicit expressions of the bases for substitution
    :ivar dict test_expr: explicit expressions of the test functions for substitution
    """
    
    def __init__(self, 
        identifier: str,
        fourier_expand: FourierExpansions, 
        rad_expand: RadialExpansions, rad_test: RadialTestFunctions, 
        inner_prod_op: RadialInnerProducts, 
        base_expr: Optional[base.LabeledCollection] = None, 
        test_expr: Optional[base.LabeledCollection] = None) -> None:
        """Initialization
        
        :param FourierExpansion fourier_expand: defines the fourier expansion part
            which is considered shared among all fields
        :param RadialExpansion rad_expand: defines the radial expansion
        :param RadialTestFunctions rad_test: test functions for each radial equation
        :param RadialInnerProducts inner_prod_op: inner products associated with each eq
        :param Optional[base.LabeledCollection] base_expr: 
            explicit expression of the bases (if `rad_expand` uses placeholders)
        :param Optional[base.LabeledCollection] test_expr: 
            explicit expression of the test functions (if `rad_test` uses placeholders)
        """
        self.identifier = identifier
        self.fourier_xpd = fourier_expand
        self.rad_xpd = rad_expand
        self.rad_test = rad_test
        self.inner_prod_op = inner_prod_op
        if base_expr is not None:
            assert base_expr._field_names == self.rad_xpd.bases._field_names
            self.base_expr = base.map_collection(self.rad_xpd.bases, base_expr)
        else:
            self.base_expr = dict()
        if test_expr is not None:
            assert test_expr._field_names == self.rad_test._field_names
            self.test_expr = base.map_collection(self.rad_test, test_expr)
        else:
            self.test_expr = dict()



class SystemEquations(base.LabeledCollection):
    """The top-level class for system of equations to be solved
    The equations are always assumed to be written in the first-order
    form in time, i.e. LHS = d()/dt
    
    :ivar ExpansionRecipe expansion_recipe: expansion recipe used for the eq
    """
    
    def __init__(self, names: List[str], 
        expansion_recipe: ExpansionRecipe, **fields) -> None:
        """Initialization
        """
        self.recipe = expansion_recipe
        super().__init__(names, **fields)
    
    def as_empty(self):
        """Overriding :meth:`base.LabeledCollection.as_empty`
        """
        return SystemEquations(self._field_names, self.recipe)
    
    def copy(self):
        """Overriding :meth:`base.LabeledCollection.copy`
        """
        return SystemEquations(self._field_names, self.recipe, 
            **{self[fname] for fname in self._field_names})
    
    def to_fourier_domain(self, inplace: bool = False) -> "SystemEquations":
        """Convert expression into Fourier domain
        
        Uses the bases and coefficients defined in recipe.fourier_ansatz
        
        :param bool inplace: whether the operation is made in situ
        """
        f_map = base.map_collection(
            self.recipe.fourier_xpd.fields, 
            self.recipe.fourier_xpd)
        new_sys = self.apply(
            lambda eq: sympy.Eq(
                FourierExpansions.to_fourier_domain(eq.lhs, f_map, self.recipe.fourier_xpd.bases), 
                FourierExpansions.to_fourier_domain(eq.rhs, f_map, self.recipe.fourier_xpd.bases)), 
            inplace=inplace, metadata=False)
        return new_sys
    
    def to_radial(self, inplace: bool = False) -> "SystemEquations":
        """Introduce radial expansion to the equations.
                
        :param bool inplace: whether the operation is made in situ
        """
        f_map = base.map_collection(
            self.recipe.rad_xpd.fields, 
            self.recipe.rad_xpd)
        new_sys = self.apply(
            lambda eq: sympy.Eq(
                eq.lhs.subs(f_map).doit().expand(), 
                eq.rhs.subs(f_map).doit().expand()),
            inplace=inplace, metadata=False)
        return new_sys
        
    def to_inner_product(self, factor_lhs: sympy.Expr = sympy.S.One, 
        factor_rhs: sympy.Expr = sympy.S.One, 
        inplace: bool = False) -> "SystemEquations":
        """Collect the radial equations to inner product form
        
        :param sympy.Expr factor_lhs: allows the user to choose 
            which factor should be moved out of the inner product on LHS
        :param factor_rhs: ditto, but for RHS
        :param bool inplace: whether the operation is made in situ
        
        .. warning:: 
        
            Functions/fields that are not expanded with an coefficient
            (elements from `recipe.rad_xpd.coeffs`) will no longer be collected
            properly and may be lost in linear system formation!
        """
        def collect_inner_product(fname, eq):
            ip_od = self.recipe.inner_prod_op[fname]
            eq_test = self.recipe.rad_test[fname]
            old_lhs = (eq.lhs/factor_lhs).doit().expand()
            old_rhs = (eq.rhs/factor_rhs).doit().expand()
            new_lhs, new_rhs = sympy.S.Zero, sympy.S.Zero
            for coeff in self.recipe.rad_xpd.coeffs:
                if not old_lhs.coeff(coeff, 1).equals(sympy.S.Zero):
                    new_lhs += factor_lhs*coeff*ip_od(eq_test, old_lhs.coeff(coeff, 1))
                if not old_rhs.coeff(coeff, 1).equals(sympy.S.Zero):
                    new_rhs += factor_rhs*coeff*ip_od(eq_test, old_rhs.coeff(coeff, 1))
            return sympy.Eq(new_lhs, new_rhs)
        
        return self.apply(collect_inner_product, inplace=inplace, metadata=True)
    
    def collect_matrices(self, factor_lhs: sympy.Expr = sympy.S.One, 
        factor_rhs: sympy.Expr = sympy.S.One) -> List["SystemMatrix"]:
        """Collect the coefficient matrices of the equations
        
        :param sympy.Expr factor_lhs: sympy.Expr, allows the user to choose 
            which factor should be moved out of the inner product on LHS
        :param factor_rhs: ditto, but for RHS
        """
        exprs_m = base.LabeledCollection(self._field_names, 
            **{fname: self[fname].lhs/factor_lhs for fname in self._field_names})
        exprs_k = base.LabeledCollection(self._field_names,
            **{fname: self[fname].rhs/factor_rhs for fname in self._field_names})
        return (SystemMatrix(exprs_m, self.recipe.rad_xpd.coeffs), 
                SystemMatrix(exprs_k, self.recipe.rad_xpd.coeffs))



class SystemMatrix:
    """System matrix
    
    This is a wrapper for a matrix of symbolic expression (inner products)
    whose numerical values will be evaluated numerically.
    This is one of the final outcomes of the symbolic computation.
    
    :ivar List[str] _row_names: list of names corresponding to each row (block)
    :ivar dict _row_idx: dict that maps row name to their index
    :ivar List[str] _col_names: list of names corresponding to each col (block)
    :ivar dict _col_idx: dict that maps col names to their index
    :ivar array-like _matrix: a nested list representing the matrix
    """
    
    @staticmethod
    def build_matrix(exprs: base.LabeledCollection, 
        coeffs: base.LabeledCollection) -> List:
        """Build system matrix from a collection of expressions
        
        :param base.LabeledCollection exprs: collection of expressions
            each element is a `sympy.Expr` instance
        :param base.LabeledCollection coeffs: collection of coefficients
            each element is a `sympy.Symbol` instance
        
        .. warning::

            Whatever term in `exprs` that does not contain a coefficient
            included in the `coeffs` collection will be discarded in the
            process!
        """
        matrix = list()
        for expr in exprs:
            expr_row = list()
            for coeff in coeffs:
                expr_row.append(expr.expand().coeff(coeff, 1))
            matrix.append(expr_row)
        return np.array(matrix)
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialization
        
        There are currently two modes of initialization.
        
        * If the first argument is a :class:`base.LabeledCollection` instance,
          it will be interpreted as a collection of expressions; the second
          argument is then interpreted as the collection of coefficients.
          The signaure is `SystemMatrix(expressions, coefficients)`.
          The :class:`SystemMatrix` will be constructed by invoking the 
          :meth:`SystemMatrix.build_matrix` method
        * Otherwise, the arguments will be interpreted as the row names,
          column names and matrix of expressions, respectively.
          The signature is `SystemMatrix(row_names, col_names, matrix)`
        """
        if isinstance(args[0], base.LabeledCollection):
            # initiate from expression and coefficients
            exprs, coeffs = args[0], args[1]
            row_names = exprs._field_names
            col_names = coeffs._field_names
            matrix = self.build_matrix(exprs, coeffs)
        else:
            row_names = args[0]
            col_names = args[1]
            matrix = args[2]
            assert matrix.shape == (len(row_names), len(col_names))
        self._row_names = row_names
        self._row_idx = {fname: idx for idx, fname in enumerate(self._row_names)}
        self._col_names = col_names
        self._col_idx = {fname: idx for idx, fname in enumerate(self._col_names)}
        self._matrix = matrix
    
    def __getitem__(self, index: List):
        """Access by index (int or name)
        """
        assert len(index) == 2
        idx_int = (
            self._row_idx[index[0]] if isinstance(index[0], str) else index[0], 
            self._col_idx[index[1]] if isinstance(index[1], str) else index[1])
        return self._matrix[idx_int]
    
    def __setitem__(self, index: Union[List[int], List[str]], 
        value: sympy.Expr):
        """Set element by index (int or name)
        """
        assert len(index) == 2
        idx_int = (
            self._row_idx[index[0]] if isinstance(index[0], str) else index[0], 
            self._col_idx[index[1]] if isinstance(index[1], str) else index[1])
        self._matrix[idx_int] = value
    
    def block_sparsity(self):
        """Give the sparsity pattern of the symbolic matrix
        
        :returns: sparsity matrix.
            If an element of the `SystemMatrix` is zero or None, then 
            it will be marked as `False` in the output; otherwise it will
            be marked as `True`.
        :rtype: numpy.ndarray
        """
        return ~np.array([[self[ridx, cidx] == sympy.S.Zero or self[ridx, cidx] is None
                          for cidx in range(self._matrix.shape[1])]
                         for ridx in range(self._matrix.shape[0])], dtype=bool)
    
    def apply(self, fun: Callable, inplace: bool=False, 
        metadata: bool=False) -> "SystemMatrix":
        """Apply function to elements iteratively.
        
        :param Callable fun: function that processes the elements
        :param bool inplace: whether the change is applied *in situ*
        :param bool metadata: whether the name of the element is passed to `fun`.
        """
        sysm_out = self if inplace else SystemMatrix(
            self._row_names, self._col_names, np.zeros(self._matrix.shape, dtype=object))
        for i_row in range(self._matrix.shape[0]):
            for i_col in range(self._matrix.shape[1]):
                if metadata:
                    sysm_out[i_row, i_col] = fun(self[i_row, i_col], 
                        (self._row_names[i_row], self._col_names[i_col]))
                else:
                    sysm_out[i_row, i_col] = fun(self[i_row, i_col])
        return sysm_out
    
    @staticmethod
    def serialize_element(element: Union[InnerProduct1D, sympy.Expr]) -> Any:
        """Serialize an element
        """
        if isinstance(element, InnerProduct1D):
            return element.serialize()
        elif isinstance(element, sympy.Expr):
            return sympy.srepr(element)
        else:
            raise TypeError
    
    def serialize(self) -> List[List]:
        """Serialize object
        """
        matrix_array = [
            [self.serialize_element(element) for element in row] 
            for row in self._matrix
        ]
        matrix_array = [self._row_names, self._col_names] + matrix_array
        return matrix_array
    
    def save_json(self, fp: TextIO) -> None:
        """Save to json file
        """
        json.dump(self.serialize(), fp, indent=4)
    
    @staticmethod
    def load_serialized_element(element: Union[dict, str]) -> sympy.Expr:
        if isinstance(element, dict):
            return InnerProduct1D(
                sympy.parse_expr(element["opd_A"]), 
                sympy.parse_expr(element["opd_B"]), 
                sympy.parse_expr(element["wt"]), 
                sympy.parse_expr(element["int_var"]), 
                sympy.parse_expr(element["lower"]), 
                sympy.parse_expr(element["upper"]))
        elif isinstance(element, str):
            return sympy.parse_expr(element)
    
    @staticmethod
    def deserialize(matrix_obj: List[List]) -> "SystemMatrix":
        """Construct system matrix from object
        """
        row_names = matrix_obj[0]
        col_names = matrix_obj[1]
        matrix_array = matrix_obj[2:]
        matrix_array = [
            [SystemMatrix.load_serialized_element(element) for element in row]
            for row in matrix_array
        ]
        return SystemMatrix(row_names, col_names, np.array(matrix_array))
    
    @staticmethod
    def load_json(fp: TextIO) -> "SystemMatrix":
        """Load from json file
        """
        matrix_array = json.load(fp)
        return SystemMatrix.deserialize(matrix_array)


# =================== Auxiliary functions =====================


def placeholder_collection(names, notation: str, *vars) -> base.LabeledCollection:
    """Build collection of placeholder symbolic functions
    """
    placeholder = base.LabeledCollection(names,
        **{fname: sympy.Symbol(r"%s_%d" % (notation, i_f))(*vars)
            for i_f, fname in enumerate(names)})
    return placeholder


def orth_pref_jacobi(pow_H: Union[sympy.Expr, int], 
    pow_s: Union[sympy.Expr, int]) -> sympy.Expr:
    """Form the basis given on the power of s and H
    This set of bases is guaranteed to form an orthogonal bases
    in an approriate Hilbert space.
    
    :param pow_H: power in H
    :param pow_s: power in s
    :returns: desired form of basis
    """
    return (H_s**pow_H*s**pow_s)*jacobi(n, pow_H, pow_s - Rational(1, 2), xi_s)


pgvar_s = base.CollectionPG(
    # Stream function
    Psi = sympy.Function(r"\Psi^{m}")(s),
    # Integrated magnetic moments
    Mss = sympy.Function(r"\overline{M_{ss}}^{m}")(s),
    Mpp = sympy.Function(r"\overline{M_{\phi\phi}}^{m}")(s),
    Msp = sympy.Function(r"\overline{M_{s\phi}}^{m}")(s),
    Msz = sympy.Function(r"\widetilde{M_{sz}}^{m}")(s),
    Mpz = sympy.Function(r"\widetilde{M_{\phi z}}^{m}")(s),
    zMss = sympy.Function(r"\widetilde{zM_{ss}}^{m}")(s),
    zMpp = sympy.Function(r"\widetilde{zM_{\phi\phi}}^{m}")(s),
    zMsp = sympy.Function(r"\widetilde{zM_{\phi s}}^{m}")(s),
    # Magnetic fields in equatorial plane
    Bs_e = sympy.Function(r"B_{s}^{em}")(s),
    Bp_e = sympy.Function(r"B_{\phi}^{em}")(s),
    Bz_e = sympy.Function(r"B_{z}^{em}")(s),
    dBs_dz_e = sympy.Function(r"B_{s, z}^{em}")(s),
    dBp_dz_e = sympy.Function(r"B_{\phi, z}^{em}")(s),
    # Magnetic fields at the boundary
    # Note here Br_b is not in 2-D disk, hence there is no 
    # radial (in cylindrical coordinates) function for it
    Bs_p = sympy.Function(r"B_s^{+m}")(s),
    Bp_p = sympy.Function(r"B_\phi^{+m}")(s),
    Bz_p = sympy.Function(r"B_z^{+m}")(s),
    Bs_m = sympy.Function(r"B_s^{-m}")(s),
    Bp_m = sympy.Function(r"B_\phi^{-m}")(s),
    Bz_m = sympy.Function(r"B_z^{-m}")(s)
)
"""Radial placeholder functions of PG fields in 2-D disk.

These are the Fourier coefficients for use in combination with
:data:`core.pgvar` or :data:`core.pgvar_ptb`, with ``omega*t+p*z`` the Fourier argument
"""

cgvar_s = base.CollectionConjugate(
    # Stream function, unchanged
    Psi = pgvar_s.Psi,
    # Conjugate variables for magnetic moments
    M_1 = sympy.Function(r"\overline{M_1}^m")(s),
    M_p = sympy.Function(r"\overline{M_+}^m")(s),
    M_m = sympy.Function(r"\overline{M_-}^m")(s),
    M_zp = sympy.Function(r"\widetilde{M_{z+}}^m")(s),
    M_zm = sympy.Function(r"\widetilde{M_{z-}}^m")(s),
    zM_1 = sympy.Function(r"\widetilde{zM_1}^m")(s),
    zM_p = sympy.Function(r"\widetilde{zM_+}^m")(s),
    zM_m = sympy.Function(r"\widetilde{zM_-}^m")(s),
    # Conjugate variables for magnetic fields in equatorial plane
    B_ep = sympy.Function(r"B_{+}^{em}")(s),
    B_em = sympy.Function(r"B_{-}^{em}")(s),
    Bz_e = pgvar_s.Bz_e,
    dB_dz_ep = sympy.Function(r"B_{+, z}^{em}")(s),
    dB_dz_em = sympy.Function(r"B_{-, z}^{em}")(s),
    # Magnetic field at the boundary
    Br_b = pgvar_s.Br_b,
    B_pp = sympy.Function(r"B_+^{+m}")(s),
    B_pm = sympy.Function(r"B_-^{+m}")(s),
    Bz_p = pgvar_s.Bz_p,
    B_mp = sympy.Function(r"B_+^{-m}")(s),
    B_mm = sympy.Function(r"B_-^{-m}")(s),
    Bz_m = pgvar_s.Bz_m
)
"""Radial placeholder functions for conjugate variables in 2-D disk.

These are the Fourier coefficients for use in combination with
:data:`core.cgvar` or :data:`core.cgvar_ptb`, with ``omega*t+p*z`` the Fourier argument
"""


reduced_var_s = base.LabeledCollection(
    ["Psi", "F_ext"],
    Psi = pgvar_s.Psi,
    F_ext = sympy.Function(r"F_\mathrm{ext}^m")(s)
)
"""Radial placeholder functions for reduced system in 2-D disk.

These are the Fourier coefficients for use in combination with
:data:`core.reduced_var`, with ``omega*t+p*z`` the Fourier argument
"""