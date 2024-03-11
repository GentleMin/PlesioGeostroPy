# -*- coding: utf-8 -*-
"""Background fields for linearization
"""

from typing import List

from sympy import *
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


class BackgroundHydro(BackgroundFieldMHD):
    """Purely static, hydrodynamic background field
    
    .. math:: \\mathbf{U}_0 = \\mathbf{0}, \\quad \\mathbf{B}_0 = \\mathbf{0}
    """
    
    def __init__(self) -> None:
        super().__init__(
            v3d.Vector3D((S.Zero, S.Zero, S.Zero), core.cyl), 
            v3d.Vector3D((S.Zero, S.Zero, S.Zero), core.cyl)
        )


class BackgroundMalkus(BackgroundFieldMHD):
    """Malkus background field
    
    .. math:: \\mathbf{U}_0 = \\mathbf{0}, \\quad \\mathbf{B}_0 = s \\hat{\\phi}
    """
    
    def __init__(self) -> None:
        super().__init__(
            v3d.Vector3D((S.Zero, S.Zero, S.Zero), core.cyl), 
            v3d.Vector3D((S.Zero, s, S.Zero), core.cyl)
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
            v3d.Vector3D((S.Zero, S.Zero, S.Zero), core.cyl), 
            v3d.Vector3D((S.Zero, cf_gamma*s*(1 - s**2 - z**2), S.Zero), core.cyl),
            params=[cf_gamma,]
        )


class BackgroundPoloidalDipole(BackgroundFieldMHD):
    """Poloidal dipolar background field (S1)
    
    .. math:: 

        \\mathbf{U}_0 = \\mathbf{0},
        
        \\mathbf{B}_0 = -6sz \\hat{s} - 2 (5 - 6s^2 - 3z^2) \\hat{z}.
    """
    
    def __init__(self) -> None:
        super().__init__(
            v3d.Vector3D((S.Zero, S.Zero, S.Zero), core.cyl), 
            v3d.Vector3D((-6*s*z, S.Zero, -2*(5 - 6*s**2 - 3*z**2)), core.cyl)
        )
