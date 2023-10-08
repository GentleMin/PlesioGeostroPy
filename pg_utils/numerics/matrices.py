# -*- coding: utf-8 -*-
"""
Symbolic manipulation and numerical computations 
of the coefficient matrices (mass and stiffness matrices)

The bridge between the symbolic expressions and numerical computations
"""


import sympy
from ..pg_model import core
from ..pg_model import expansion as xpd


class InnerProdQuad:
    """Quadrature of inner product
    Base class for all inner product quadratures in 1D
    
    Compared to the direct quadratures of the integral form,
    calculating quadratures in the notation of inner products
    allows one to drastically save of time of basis evaluation.
    When calculating the integral in the form of
        Integral(w(x)*Phi1(l, x)*Phi2(n, x), (x, -1, 1))
    directly calculating the integral using K-point quadrature
    for 0 <= l,n <= N would need KN^2 evaluations of both Phi1
    and Phi2; however, Phi1 and Phi2 in fact only need to be 
    evaluated KN times. This reduces the complexity of evaluation
    from O(KN^2) to O(KN).
    """
    
    def __init__(self) -> None:
        pass
