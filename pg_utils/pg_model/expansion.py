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


from typing import Any, List, Optional, Union
import sympy

from . import base
from .core import s



# Integers denoting the radial and azimuthal basis
n, m = sympy.symbols("n, m", integer=True)

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
    
    def integral_form(self):
        """Get the explicit integral form
        """
        return sympy.Integral(
            self._opd_A*self._opd_B*self._wt, 
            (self._int_var, self._bound[0], self._bound[1]))
    
    def change_variable(self, new_var: sympy.Symbol, 
        int_var_expr: sympy.Expr, inv_expr: sympy.Expr):
        """Change the integration variable
        
        :param new_var: the new variable to be integrated over
        :param int_var_expr: the current variable expressed in new variable
        :param inv_expr: one needs to explicitly state the inverse expression
        """
        jac = sympy.Abs(sympy.diff(int_var_expr, new_var).doit())
        self._opd_A = self._opd_A.subs({self._int_var: int_var_expr})
        self._opd_B = self._opd_B.subs({self._int_var: int_var_expr})
        self._wt = jac*self._wt.subs({self._int_var: int_var_expr})
        self._bound = (
            inv_expr.subs({self._int_var: self._bound[0]}).doit(),
            inv_expr.subs({self._int_var: self._bound[1]}).doit())
        self._int_var = new_var



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
    """This is the top-level class used for spectral expansion,
    which defines the recipe concerning radial and azimuthal expansions.
    
    :param equations: set of PG equations
    :param activated: bool array indicating activated equations
    :param fourier_ansatz: defines the fourier expansion part
        this is considered shared among all fields
    :param rad_expansion: defines the radial expansion
    :param rad_test: test functions for each radial equation
    :param inner_products: inner products associated with each eq
    """
    
    def __init__(self, equations: base.LabeledCollection, 
        fourier_expand: FourierExpansions, rad_expand: RadialExpansions, 
        rad_test: RadialTestFunctions, inner_prod_op: RadialInnerProducts) -> None:
        """Initialization
        """
        self.equations = equations
        self.fourier_ansatz = fourier_expand
        self.rad_expansion = rad_expand
        self.rad_test = rad_test
        self.inner_prod_op = inner_prod_op



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

