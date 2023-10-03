# -*- coding: utf-8 -*-

"""
Define expansions of the PG fields in the unit circle
"""


from typing import Any, List, Optional
import sympy
from . import base, core
from .core import s, p, t, H, H_s



# Integers denoting the radial and azimuthal basis
n, m = sympy.symbols("n, m", integer=True)

# Angular frequency
omega = sympy.Symbol(r"\omega")

# Integration variable for the radial direction
xi = sympy.Symbol(r"\xi")
# xi_s = 



class FourierExpansion:
    """Fourier expansion for PG variables in 2D disk
    This class is used for reducing equations or expressions into the
    Fourier domain given specific arguments. The Fourier-domain eqs/
    exprs are then processed using `RadialExpansion`
    
    :param _fourier_basis: complex Fourier basis
    :param _fourier_coeffs: Fourier coefficients
    :param _fields: fields to use the Fourier expansion
    :param _fourier_map
    """
        
    def __init__(self, argument: sympy.Expr, fields: base.CollectionPG, 
        fourier_coeffs: base.CollectionPG) -> None:
        """Initiate Fourier expansion
        
        :param arguments: the complex argument of the Fourier basis
        :param fields: PG fields to use the Fourier expansion on
        :param fourier_coeffs: Fourier coefficients; these should be
            Fourier coefficients of the fields specified by `fields`
            expanded with argument specified by `argument`.
        .. Example:
            if `fields` are functions of s, p, z and t, and `argument`
            takes the form omega*t + m*p, then fourier_coeffs should
            be functions of s and z.
        """
        self._fourier_basis = sympy.exp(sympy.I*argument)
        self._fourier_coeffs = fourier_coeffs
        self._fields = fields
        self._fourier_map = self._build_fourier_map()
    
    def _build_fourier_map(self):
        """Create mapping from fields to their Fourier-domain expressions
        """
        fourier_map = {
            self._fields[idx]: self._fourier_coeffs[idx]*self._fourier_basis
            for idx in range(self._fourier_coeffs.n_fields)
            if self._fourier_coeffs[idx] is not None}
        return fourier_map
    
    def to_fourier_domain(self, expr):
        """Convert an expression to Fourier domain
        """
        expr_fourier = expr.subs(self._fourier_map).doit()/self._fourier_basis
        return expr_fourier



class RadialBases:
    """Set of radial bases
    
    This class is especially used for the case when
    the bases do not coincide with the PG variables
    i.e. when PG variables are combinations of these
    bases, rather than each PG variable is expanded
    into a unique basis
    
    :param _bases: list of radial bases to be used
    :param _coeffs: list of coefficients corresponding
        to the elements in _bases. Coefficients need 
        to be unique. Coefficients and bases
        and used in pair to extract the proper inner
        products and compute the matrix elements.
    """
    
    def __init__(self, bases: List[Any], coeffs: List[sympy.Symbol]) -> None:
        """There is not necessarily special meaning of the bases
        and therefore it is only a wrapper class for two lists
        """
        # Forced check of uniqueness and correspondence
        assert len(coeffs) == len(set(coeffs))
        assert len(bases) == len(coeffs)
        self._bases = bases
        self._coeffs = coeffs
    
    def __getitem__(self, __key: int):
        return self._coeffs[__key], self._bases[__key]
    
    def __setitem__(self, __key: int, __val: Any):
        self._bases[__key] == __val



class RadialExpansionPG(base.CollectionPG):
    """Radial expansion for PG variables
    """
    
    def __init__(self, rad_basis: RadialBases) -> None:
        super().__init__()
        self.rad_basis = rad_basis



class RadialTestPG(base.CollectionPG):
    """Radial functions that are used as test functions for reducing
    PG equations into algebraic form.
    """
    
    def __init__(self, **fields) -> None:
        super().__init__(**fields)



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
        self._print_mid = True
    
    def _latex(self, printer, *args):
        if self._print_mid:
            str_A = printer._print(self._opd_A, *args)
            str_B = printer._print(self._opd_B, *args)
            str_wt = printer._print(self._wt, *args)
            str_var = printer._print(self._int_var, *args)
            return r"\left\langle %s \, \big|\, %s \,\big|\, %s \right\rangle_{%s}" % (
                str_A, str_wt, str_B, str_var)
        else:
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



class InnerProductPG(base.CollectionPG):
    """Collection of inner products
    """
    
    def __init__(self, **fields) -> None:
        super().__init__(**fields)



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
    
    def __init__(self, 
        equations: base.CollectionPG, activated: List[bool],
        fourier_ansatz: FourierExpansion, rad_expansion: RadialExpansionPG, 
        rad_test: RadialTestPG, inner_products: InnerProductPG) -> None:
        """Initialization
        """
        self.equations = equations
        self.activated = activated
        self.fourier_ansatz = fourier_ansatz
        self.rad_expansion = rad_expansion
        self.rad_test = rad_test
        self.inner_products = inner_products



"""Radial placeholder functions of PG fields in 2-D disk.
These are the Fourier coefficients for using in combination with
core.pgvar or core.pgvar_lin, with omega*t+p*z the Fourier argument
"""
pgvar_s = base.CollectionPG(
    # Vorticity
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

