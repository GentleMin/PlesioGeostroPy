# -*- coding: utf-8 -*-
"""Background fields for linearization
"""

from typing import List

from sympy import *
from pg_utils.sympy_supp import vector_calculus_3d as v3d
from pg_utils.pg_model import *
from pg_utils.pg_model import core


class BackgroundFieldMHD:
    """
    """
    
    def __init__(self, velocity: v3d.Vector3D, 
        magnetic: v3d.Vector3D, params: List = []) -> None:
        """
        """
        self.U0_val = velocity
        self.B0_val = magnetic
        self.params = params


class BackgroundHydro(BackgroundFieldMHD):
    
    def __init__(self) -> None:
        super().__init__(
            v3d.Vector3D((S.Zero, S.Zero, S.Zero), core.cyl), 
            v3d.Vector3D((S.Zero, S.Zero, S.Zero), core.cyl)
        )


class BackgroundMalkus(BackgroundFieldMHD):
    
    def __init__(self) -> None:
        super().__init__(
            v3d.Vector3D((S.Zero, S.Zero, S.Zero), core.cyl), 
            v3d.Vector3D((S.Zero, s, S.Zero), core.cyl)
        )


class BackgroundToroidalQuadrupole(BackgroundFieldMHD):
    
    def __init__(self) -> None:
        cf_gamma = Symbol(r"\gamma")
        super().__init__(
            v3d.Vector3D((S.Zero, S.Zero, S.Zero), core.cyl), 
            v3d.Vector3D((S.Zero, cf_gamma*s*(1 - s**2 - z**2), S.Zero), core.cyl),
            params=[cf_gamma,]
        )


class BackgroundPoloidalDipole(BackgroundFieldMHD):
    
    def __init__(self) -> None:
        super().__init__(
            v3d.Vector3D((S.Zero, S.Zero, S.Zero), core.cyl), 
            v3d.Vector3D((-6*s*z, S.Zero, -2*(5 - 6*s**2 - 3*z**2)), core.cyl)
        )
