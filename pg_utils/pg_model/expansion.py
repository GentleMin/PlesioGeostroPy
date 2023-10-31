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

from . import base
from .core import s
import numpy as np

import json



# Integers denoting the radial and azimuthal basis
n, m = sympy.symbols("n, m", integer=True)

# Index for the orders of the radial test and trial functions
n_test = sympy.Symbol(r"\ell'", integer=True)
n_trial = sympy.Symbol(r"\ell", integer=True)

# Angular frequency
omega = sympy.Symbol(r"\omega")

# Integration variable for the radial direction and transforms
xi = sympy.Symbol(r"\xi")
xi_s = 2*s**2 - 1
s_xi = sympy.sqrt((xi + 1)/2)



class SpectralExpansion(base.LabeledCollection):
    """Base class for different kinds of spectral expansions
    
    :param fields: collection to be expanded
    :param bases: collection of bases to be used
    :param coeffs: collection of the corresponding coeffs.
    """
    
    def __init__(self, fields: base.LabeledCollection, 
        bases: Union[base.LabeledCollection, sympy.Expr], 
        coeffs: base.LabeledCollection, 
        **relations: sympy.Expr) -> None:
        """Initialization
        """
        super().__init__(fields._field_names, **relations)
        self.fields, self.bases, self.coeffs = fields, bases, coeffs
    


class FourierExpansions(SpectralExpansion):
    """Fourier expansion for fields
    This class is used for reducing equations or expressions into the
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
        
        :param arguments: the complex argument of the Fourier basis
        :param fields: fields to use the Fourier expansion on
        :param fourier_coeffs: Fourier coefficients; these should be
            Fourier coefficients of the fields specified by `fields`
            expanded with argument specified by `argument`.
        .. Example:
            if `fields` are functions of s, p, z and t, and `argument`
            takes the form omega*t + m*p, then fourier_coeffs should
            be functions of s and z.
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
    - As we expand the elements of tensors, it would be desirable to
    implement different forms of bases for each individual field, so
    that the regularity constraints are fulfilled;
    - we potentially have a mixture of quantities expressed in 
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
    """Radial functions that are used as test functions for reducing
    differential equations into algebraic form.
    """
    
    def __init__(self, names: List[str], **test_functions: sympy.Expr) -> None:
        super().__init__(names, **test_functions)
        


class InnerProduct1D(sympy.Expr):
    """1-D (integration) inner product
    
    :param _opd_A: left operand
    :param _opd_B: right operand
    :param _wt: weight
    :param _int_var: integration variable
    :param _bound: integration bounds
    :param _print_mid: whether to print weight in the middle
        if False, the weight will be merged with the second argument
    """
    
    def __init__(self, opd_A: sympy.Expr, opd_B: sympy.Expr, wt: sympy.Expr, 
        int_var: sympy.Symbol, lower: sympy.Expr, upper: sympy.Expr) -> None:
        """Initialization
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
        
        :param new_var: the new variable to be integrated over
        :param int_var_expr: the current variable expressed in new variable
        :param inv_expr: one needs to explicitly state the inverse expression
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
        """Apply the distributive property of inner products to 
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
        return {"opd_A": sympy.srepr(self._opd_A), 
                "opd_B": sympy.srepr(self._opd_B), "wt": sympy.srepr(self._wt), 
                "int_var": sympy.srepr(self._int_var), 
                "lower": sympy.srepr(self._bound[0]), 
                "upper": sympy.srepr(self._bound[1])}



class InnerProductOp1D:
    """Inner product defined on 1-D function space
    
    :param _int_var: integration variable
    :param _wt: weight
    :param _bound: integration bound, 2-tuple
    :param _conj: which argument to perform the complex conjugation
    """
    
    def __init__(self, int_var: sympy.Symbol, wt: sympy.Expr, 
        bound: List, conj: Optional[int] = None) -> None:
        """Initialization
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
    
    :param activated: bool array indicating activated equations
    :param fourier_ansatz: defines the fourier expansion part
        this is considered shared among all fields
    :param rad_expansion: defines the radial expansion
    :param rad_test: test functions for each radial equation
    :param inner_products: inner products associated with each eq
    """
    
    def __init__(self, fourier_expand: FourierExpansions, 
        rad_expand: RadialExpansions, rad_test: RadialTestFunctions, 
        inner_prod_op: RadialInnerProducts, 
        base_expr: Optional[base.LabeledCollection] = None, 
        test_expr: Optional[base.LabeledCollection] = None) -> None:
        """Initialization
        """
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
    
    :param recipe: ExpansionRecipe, collection of expansion rules
    :param \**fields: equations as attributes.
    """
    
    def __init__(self, names: List[str], 
        expansion_recipe: ExpansionRecipe, **fields) -> None:
        """Initialization
        """
        self.recipe = expansion_recipe
        super().__init__(names, **fields)
    
    def as_empty(self):
        """Construct empty object with the same set of fields and recipe.
        Overriden method.
        """
        return SystemEquations(self._field_names, self.recipe)
    
    def copy(self):
        """Deep copy method, overriden due to a new attribute, recipe,
        to be copied (copying of recipe is however shallow).
        """
        return SystemEquations(self._field_names, self.recipe, 
            **{self[fname] for fname in self._field_names})
    
    def to_fourier_domain(self, inplace: bool = False) -> "SystemEquations":
        """Convert expression into Fourier domain
        using the bases and coefficients defined in recipe.fourier_ansatz
        
        :param inplace: bool, whether the operation is made in situ
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
        Note: functions/fields that are not expanded with an coefficient
            (elements from recipe.rad_xpd.coeffs) will no longer be collected
            properly and may be lost in linear system formation!
        
        :param inplace: bool, whether the operation is made in situ
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
        
        :param factor_lhs: sympy.Expr, allows the user to choose 
            which factor should be moved out of the inner product on LHS
        :param factor_rhs: ditto, but for RHS
        :param inplace: bool, whether the operation is made in situ
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
        
        :param factor_lhs: sympy.Expr, allows the user to choose 
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
    """
    
    @staticmethod
    def build_matrix(exprs: base.LabeledCollection, 
        coeffs: base.LabeledCollection) -> List:
        matrix = list()
        for expr in exprs:
            expr_row = list()
            for coeff in coeffs:
                expr_row.append(expr.expand().coeff(coeff, 1))
            matrix.append(expr_row)
        return np.array(matrix)
    
    def __init__(self, *args, **kwargs) -> None:
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
        return ~np.array([[self[ridx, cidx] == sympy.S.Zero
                          for cidx in range(self._matrix.shape[1])]
                         for ridx in range(self._matrix.shape[0])], dtype=bool)
    
    def apply(self, fun: Callable, inplace: bool=False, 
        metadata: bool=False) -> "SystemMatrix":
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
        if isinstance(element, InnerProduct1D):
            return element.serialize()
        elif isinstance(element, sympy.Expr):
            return sympy.srepr(element)
        else:
            raise TypeError
    
    def save_json(self, fp: TextIO) -> None:
        """Save to json file
        """
        matrix_array = [
            [self.serialize_element(element) for element in row] 
            for row in self._matrix
        ]
        matrix_array = [self._row_names, self._col_names] + matrix_array
        json.dump(matrix_array, fp, indent=4)
    
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
    def load_json(fp: TextIO) -> "SystemMatrix":
        """Load from json file
        """
        matrix_array = json.load(fp)
        row_names = matrix_array[0]
        col_names = matrix_array[1]
        matrix_array = matrix_array[2:]
        matrix_array = [
            [SystemMatrix.load_serialized_element(element) for element in row]
            for row in matrix_array
        ]
        return SystemMatrix(row_names, col_names, np.array(matrix_array))



def placeholder_collection(names, notation: str, *vars) -> base.LabeledCollection:
    """Build collection of placeholder symbolic functions
    """
    placeholder = base.LabeledCollection(names,
        **{fname: sympy.Symbol(r"%s_%d" % (notation, i_f))(*vars)
            for i_f, fname in enumerate(names)})
    return placeholder


"""Radial placeholder functions of PG fields in 2-D disk.
These are the Fourier coefficients for using in combination with
core.pgvar or core.pgvar_lin, with omega*t+p*z the Fourier argument
"""
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
    Bs_e = sympy.Function(r"B_{es}^{m}")(s),
    Bp_e = sympy.Function(r"B_{e\phi}^{m}")(s),
    Bz_e = sympy.Function(r"B_{ez}^{m}")(s),
    dBs_dz_e = sympy.Function(r"B_{es, z}^{m}")(s),
    dBp_dz_e = sympy.Function(r"B_{e\phi, z}^{m}")(s),
    # Magnetic fields at the boundary
    # Note here Br_b is not in 2-D disk, hence there is no 
    # radial (in cylindrical coordinates) function for it
    Bs_p = sympy.Function(r"B_s^{m+}")(s),
    Bp_p = sympy.Function(r"B_\phi^{m+}")(s),
    Bz_p = sympy.Function(r"B_z^{m+}")(s),
    Bs_m = sympy.Function(r"B_s^{m-}")(s),
    Bp_m = sympy.Function(r"B_\phi^{m-}")(s),
    Bz_m = sympy.Function(r"B_z^{m-}")(s)
)

