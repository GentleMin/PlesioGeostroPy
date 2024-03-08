# -*- coding: utf-8 -*-
"""Background setup for a toroidal quadrupolar field

.. math::

    B_s = 0
    B_{\\phi} = \\gamma s (1 - s^2 - z^2) \\ 
    B_z = 0
"""

from sympy import *
from pg_utils.sympy_supp import vector_calculus_3d as v3d
from pg_utils.pg_model import *
from pg_utils.pg_model import core

#: coefficient of the field
cf_gamma = Symbol(r"\gamma")

U0_val = v3d.Vector3D((S.Zero, S.Zero, S.Zero), core.cyl)
"""Background velocity field: zero velocity
"""

B0_val = v3d.Vector3D((S.Zero, cf_gamma*s*(1 - s**2 - z**2), S.Zero), core.cyl)
"""Background magnetic field: a toroidal quadrupolar field
"""

