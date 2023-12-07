# -*- coding: utf-8 -*-
"""
Symbolic realization of the PG model and the spectral recipes

This module includes the following sub-modules:

* :mod:`~pg_utils.pg_model.base` base classes
* :mod:`~pg_utils.pg_model.core` core variables for the PG model
* :mod:`~pg_utils.pg_model.equations` equations for the PG model
* :mod:`~pg_utils.pg_model.forcing` all sorts of forcings to be inserted into equations
* :mod:`~pg_utils.pg_model.params` controlling parameters of the system
* :mod:`~pg_utils.pg_model.expansion` base classes for expansion recipes

"""


from .core import x, y, z, s, p, z, r, theta, t, H, H_s
