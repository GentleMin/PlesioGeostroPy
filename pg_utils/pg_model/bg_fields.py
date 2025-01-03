# -*- coding: utf-8 -*-
"""Background fields for linearization
"""

from typing import List

from sympy import *
from sympy.functions.special import spherical_harmonics as SH
from pg_utils.sympy_supp import vector_calculus_3d as v3d
from pg_utils.pg_model import *
from pg_utils.pg_model import core


class BackgroundFieldMHD:
    """Abstract base class for MHD background fields
    
    :ivar Vector3D U0_val: background velocity field
    :ivar Vector3D B0_val: background magnetic field
    :ivar List params: list of variable parameters
    """
    
    def __init__(self, velocity: v3d.Vector3D, 
        magnetic: v3d.Vector3D, params: List = []) -> None:
        """
        """
        self.U0_val = velocity
        self.B0_val = magnetic
        self.params = params
        
    def __mul__(self, scalar: Expr) -> 'BackgroundFieldMHD':
        """Scalar product, left multiplication to scalar
        """
        prod = BackgroundFieldMHD(
            tuple(scalar*comp for comp in self.U0_val),
            tuple(scalar*comp for comp in self.B0_val),
            params=list(set(self.params) | (scalar.atoms(Symbol) - {s, p, z}))
        )
        return prod
    
    def __rmul__(self, scalar: Expr) -> 'BackgroundFieldMHD':
        """Scalar product, right multiplication to scalar
        """
        return self.__mul__(scalar)
    
    def __add__(self, bg_field: 'BackgroundFieldMHD') -> 'BackgroundFieldMHD':
        """Field addition
        """
        summation = BackgroundFieldMHD(
            tuple(self.U0_val[i_comp] + bg_field.U0_val[i_comp] for i_comp in range(3)),
            tuple(self.B0_val[i_comp] + bg_field.B0_val[i_comp] for i_comp in range(3)),
            params=list(set(self.params) | set(bg_field.params))
        )
        return summation
    
    def subs(self, sub_params, inplace=False) -> 'BackgroundFieldMHD':
        """Parameter substitution
        """
        o_subs = self
        if not inplace:
            o_subs = BackgroundFieldMHD(self.U0_val, self.B0_val, self.params)
            
        o_subs.U0_val = self.U0_val.subs(sub_params)
        o_subs.B0_val = self.B0_val.subs(sub_params)
        return o_subs


class BackgroundHydro(BackgroundFieldMHD):
    """Purely static, hydrodynamic background field
    
    .. math:: \\mathbf{U}_0 = \\mathbf{0}, \\quad \\mathbf{B}_0 = \\mathbf{0}
    """
    
    def __init__(self) -> None:
        super().__init__(
            v3d.Vector3D([S.Zero, S.Zero, S.Zero], core.cyl), 
            v3d.Vector3D([S.Zero, S.Zero, S.Zero], core.cyl)
        )


class BackgroundMalkus(BackgroundFieldMHD):
    """Malkus background field
    
    .. math:: \\mathbf{U}_0 = \\mathbf{0}, \\quad \\mathbf{B}_0 = s \\hat{\\phi}
    """
    
    def __init__(self) -> None:
        super().__init__(
            v3d.Vector3D([S.Zero, S.Zero, S.Zero], core.cyl), 
            v3d.Vector3D([S.Zero, s, S.Zero], core.cyl)
        )


class BackgroundToroidalQuadrupole(BackgroundFieldMHD):
    """Toroidal quadrupolar background field (T1)
    
    .. math:: 

        \\mathbf{U}_0 = \\mathbf{0},
        
        \\mathbf{B}_0 = \\gamma s (1 - s^2 - z^2) \\hat{\\phi}.
    """
    
    def __init__(self) -> None:
        cf_gamma = Symbol(r"\gamma")
        super().__init__(
            v3d.Vector3D([S.Zero, S.Zero, S.Zero], core.cyl), 
            v3d.Vector3D([S.Zero, cf_gamma*s*(1 - s**2 - z**2), S.Zero], core.cyl),
            params=[cf_gamma,]
        )
        self.params_ref = [[3*sqrt(Integer(3))/2],]


class Background_T1(BackgroundFieldMHD):
    """Toroidal quadrupolar background field (T1) expressed in normalised SH
    """
    
    def __init__(self) -> None:
        cf_gamma = Symbol(r'\gamma')
        T1 = r*(1 - r**2)
        T1 = (T1*r*SH.Ynm(1, 0, theta, p).expand(func=True), S.Zero, S.Zero)
        B_T1_sph = core.sph.curl(T1)
        B_T1_sph = tuple(comp.simplify() for comp in B_T1_sph)
        B_T1_cyl = [
            cf_gamma*(B_T1_sph[0]*sin(theta) + B_T1_sph[1]*cos(theta)).subs(core.coordmap_s2c).simplify(),
            cf_gamma*B_T1_sph[2].subs(core.coordmap_s2c).simplify(),
            cf_gamma*(B_T1_sph[0]*cos(theta) - B_T1_sph[1]*sin(theta)).subs(core.coordmap_s2c).simplify()
        ]
        super().__init__(
            v3d.Vector3D([S.Zero, S.Zero, S.Zero], core.cyl), 
            v3d.Vector3D(B_T1_cyl, core.cyl), 
            params=[cf_gamma]
        )
        self.params_ref = [
            [3*sqrt(pi),]
        ]
        

class Background_T2(BackgroundFieldMHD):
    """Toroidal l=2 (T2) expressed in normalised SH
    """
    
    def __init__(self) -> None:
        cf_gamma = Symbol(r'\gamma')
        T2 = r**2*(1 - r**2)
        T2 = (T2*r*SH.Ynm(2, 0, theta, p).expand(func=True), S.Zero, S.Zero)
        B_sph = core.sph.curl(T2)
        B_sph = tuple(comp.simplify() for comp in B_sph)
        B_cyl = [
            cf_gamma*(B_sph[0]*sin(theta) + B_sph[1]*cos(theta)).subs(core.coordmap_s2c).simplify(),
            cf_gamma*B_sph[2].subs(core.coordmap_s2c).simplify(),
            cf_gamma*(B_sph[0]*cos(theta) - B_sph[1]*sin(theta)).subs(core.coordmap_s2c).simplify()
        ]
        super().__init__(
            v3d.Vector3D([S.Zero, S.Zero, S.Zero], core.cyl), 
            v3d.Vector3D(B_cyl, core.cyl), 
            params=[cf_gamma]
        )
        self.params_ref = [
            [Rational(16, 3)*sqrt(pi/5),]
        ]


class BackgroundPoloidalDipole(BackgroundFieldMHD):
    """Poloidal dipolar background field (S1)
    
    .. math:: 

        \\mathbf{U}_0 = \\mathbf{0},
        
        \\mathbf{B}_0 = -6sz \\hat{s} - 2 (5 - 6s^2 - 3z^2) \\hat{z}.
    """
    
    def __init__(self) -> None:
        super().__init__(
            v3d.Vector3D([S.Zero, S.Zero, S.Zero], core.cyl), 
            v3d.Vector3D([-6*s*z, S.Zero, -2*(5 - 6*s**2 - 3*z**2)], core.cyl)
        )


class BackgroundPoloidalDipoleTunable(BackgroundFieldMHD):
    """Poloidal dipolar background field (S1)
    
    .. math:: 

        \\mathbf{U}_0 = \\mathbf{0},
        
        \\mathbf{B}_0 = \\gamma [-6sz \\hat{s} - 2 (5 - 6s^2 - 3z^2) \\hat{z}].
    """
    
    def __init__(self) -> None:
        cf_gamma = Symbol(r'\gamma')
        super().__init__(
            v3d.Vector3D([S.Zero, S.Zero, S.Zero], core.cyl), 
            v3d.Vector3D([-cf_gamma*3*s*z/5, S.Zero, cf_gamma*((6*s**2 + 3*z**2)/5 - 1)], core.cyl),
            params=[cf_gamma,]
        )
        self.params_ref = [[S.One],]


class Background_S1(BackgroundFieldMHD):
    """Poloidal dipolar background field (S1) expressed in SH    
    """
    def __init__(self) -> None:
        cf_gamma = Symbol(r'\gamma')
        S1 = r*(5 - 3*r**2)
        S1 = (S1*r*SH.Ynm(1, 0, theta, p).expand(func=True), S.Zero, S.Zero)
        B_S1_sph = core.sph.curl(core.sph.curl(S1))
        B_S1_sph = tuple(comp.simplify() for comp in B_S1_sph)
        B_S1_cyl = [
            cf_gamma*(B_S1_sph[0]*sin(theta) + B_S1_sph[1]*cos(theta)).subs(core.coordmap_s2c).simplify(),
            cf_gamma*B_S1_sph[2].subs(core.coordmap_s2c).simplify(),
            cf_gamma*(B_S1_sph[0]*cos(theta) - B_S1_sph[1]*sin(theta)).subs(core.coordmap_s2c).simplify()
        ]
        super().__init__(
            v3d.Vector3D([S.Zero, S.Zero, S.Zero], core.cyl), 
            v3d.Vector3D(B_S1_cyl, core.cyl), 
            params=[cf_gamma]
        )
        self.params_ref = [
            [sqrt(pi/3)/5,]
        ]        


class Background_S2(BackgroundFieldMHD):
    """Poloidal quadrupolar background field (S2)
    
    .. math::
    
        \\mathbf{U}_0 = \\mathbf{0},
        
        \\mathbf{B}_0 = \\gamma \\nabla\\times\\nabla\\times [r^2(157 - 296r^2 + 143r^4) Y_2^0 \\mathbf{r}]
                      = \\gamma [-s(157 - 296s^2 - 888z^2 + 715z^4 + 143s^4 + 858s^2z^2) \\hat{s}]
    """
    def __init__(self) -> None:
        cf_gamma = Symbol(r'\gamma')
        S2 = r**2*(157 - 296*r**2 + 143*r**4)
        S2 = (S2*r*SH.Ynm(2, 0, theta, p).expand(func=True), S.Zero, S.Zero)
        B_S2_sph = core.sph.curl(core.sph.curl(S2))
        B_S2_sph = tuple(comp.simplify() for comp in B_S2_sph)
        B_S2_cyl = [
            cf_gamma*(B_S2_sph[0]*sin(theta) + B_S2_sph[1]*cos(theta)).subs(core.coordmap_s2c).simplify(),
            cf_gamma*B_S2_sph[2].subs(core.coordmap_s2c).simplify(),
            cf_gamma*(B_S2_sph[0]*cos(theta) - B_S2_sph[1]*sin(theta)).subs(core.coordmap_s2c).simplify()
        ]
        super().__init__(
            v3d.Vector3D([S.Zero, S.Zero, S.Zero], core.cyl), 
            v3d.Vector3D(B_S2_cyl, core.cyl), 
            params=[cf_gamma]
        )
        self.params_ref = [
            [Rational(5, 14)*sqrt(Rational(3, 182)),]
        ]


class Background_S_l2_n2(BackgroundFieldMHD):
    """Poloidal field, L=2, N=2
    """
    def __init__(self) -> None:
        cf_gamma = Symbol(r'\gamma')
        S2 = r**2*(5*r**2 - 7)
        S2 = (S2*r*SH.Ynm(2, 0, theta, p).expand(func=True), S.Zero, S.Zero)
        B_S2_sph = core.sph.curl(core.sph.curl(S2))
        B_S2_sph = tuple(comp.simplify() for comp in B_S2_sph)
        B_S2_cyl = [
            cf_gamma*(B_S2_sph[0]*sin(theta) + B_S2_sph[1]*cos(theta)).subs(core.coordmap_s2c).simplify(),
            cf_gamma*B_S2_sph[2].subs(core.coordmap_s2c).simplify(),
            cf_gamma*(B_S2_sph[0]*cos(theta) - B_S2_sph[1]*sin(theta)).subs(core.coordmap_s2c).simplify()
        ]
        super().__init__(
            v3d.Vector3D([S.Zero, S.Zero, S.Zero], core.cyl), 
            v3d.Vector3D(B_S2_cyl, core.cyl), 
            params=[cf_gamma]
        )
        self.params_ref = [
            [Rational(1, 4)*sqrt(Rational(3, 26)),]
        ]


class Background_T1S1(BackgroundFieldMHD):
    """Mixed T1-S1 background field
    """
    def __init__(self) -> None:
        c_T1, c_S1 = symbols('C_{T1}, C_{S1}')
        B_T1 = Background_T1()
        B_S1 = Background_S1()
        B_T1S1 = [
            B_T1.B0_val[0].subs({B_T1.params[0]: c_T1}) + B_S1.B0_val[0].subs({B_S1.params[0]: c_S1}),
            B_T1.B0_val[1].subs({B_T1.params[0]: c_T1}) + B_S1.B0_val[1].subs({B_S1.params[0]: c_S1}),
            B_T1.B0_val[2].subs({B_T1.params[0]: c_T1}) + B_S1.B0_val[2].subs({B_S1.params[0]: c_S1}),
        ]
        super().__init__(
            v3d.Vector3D([S.Zero, S.Zero, S.Zero], core.cyl), 
            v3d.Vector3D(B_T1S1, core.cyl), 
            params=[c_T1, c_S1]
        )
        self.params_ref = [
            [Rational(3, 8)*sqrt(70), sqrt(Rational(7, 69)*pi)/2]
        ]
