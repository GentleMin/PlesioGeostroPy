# -*- coding: utf-8 -*-
"""Background setup for a toroidal quadrupolar field

.. math::

    B_s = -6sz \\
    B_{\\phi} = 0 \\ 
    B_z = -2 (5 - 6s^2 - 3z^2)
"""

from sympy import *
from pg_utils.sympy_supp import vector_calculus_3d as v3d
from pg_utils.pg_model import *
from pg_utils.pg_model import core

#: Background velocity
U0_val = v3d.Vector3D((S.Zero, S.Zero, S.Zero), core.cyl)

#: Background magnetic field
B0_val = v3d.Vector3D((-6*s*z, S.Zero, -2*(5 - 6*s**2 - 3*z**2)), core.cyl)


