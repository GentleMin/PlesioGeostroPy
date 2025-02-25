# -*- coding: utf-8 -*-
"""
Controlling parameters of the system
"""


import sympy


Le = sympy.Symbol('\mathrm{Le}')
"""Lehnert number :math:`\\mathrm{Le} = \\frac{B}{\sqrt{\\rho \mu_0}\Omega L}` """

Lu = sympy.Symbol('\mathrm{Lu}')
"""Lundquist number :math:`\\mathrm{Lu} = \\frac{\\nu}{\eta}` """

Ek = sympy.Symbol('E')
"""Ekman number :math:`E = \\frac{\\nu}{\\Omega L^2}` """

Em = sympy.Symbol('E_\eta')
"""Magnetic Ekman number :math:`E_\\eta = \\frac{\\eta}{\\Omega L^2}` """

Pr = sympy.Symbol('\mathrm{Pr}')
"""Prandtl number :math:`\mathrm{Pr} = \\frac{\\nu}{\\kappa}` """

Pm = sympy.Symbol('\mathrm{Pm}')
"""Magnetic Prandtl number :math:`\mathrm{Pm} = \\frac{\\nu}{\\eta}` """

El = sympy.Symbol('\Lambda')
"""Elsasser number :math:`\Lambda = \\frac{B^2}{\\rho \\mu_0 \\eta \\Omega}` """
