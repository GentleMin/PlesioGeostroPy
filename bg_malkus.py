# -*- coding: utf-8 -*-
"""Background field setup for the Malkus field
"""


from sympy import *
from pg_utils.sympy_supp import vector_calculus_3d as v3d
from pg_utils.pg_model import *
from pg_utils.pg_model import core


U0_val = v3d.Vector3D((S.Zero, S.Zero, S.Zero), core.cyl)

B0_val = v3d.Vector3D((S.Zero, s, S.Zero), core.cyl)
