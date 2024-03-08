# -*- coding: utf-8 -*-
"""Background field setup for the purely hydrodynamic eigenmodes
"""


from sympy import *
from pg_utils.sympy_supp import vector_calculus_3d as v3d
from pg_utils.pg_model import *
from pg_utils.pg_model import core

#: Background velocity field = 0
U0_val = v3d.Vector3D((S.Zero, S.Zero, S.Zero), core.cyl)

#: Background magnetic field (Malkus field)
B0_val = v3d.Vector3D((S.Zero, S.Zero, S.Zero), core.cyl)
