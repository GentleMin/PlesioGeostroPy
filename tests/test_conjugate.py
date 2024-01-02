# -*- coding: utf-8 -*-
"""For testing equations concerning the conjugate quantities
"""

from sympy import diff, I
from pg_utils.pg_model.core import *
from pg_utils.pg_model import equations as pgeq


# For simplicity of notations
Us, Up, Uz = U_vec.s, U_vec.p, U_vec.z


class TestConjugateEquations:
    
    @staticmethod
    def u_dot_grad_h(var):
        return U_vec[0]*diff(var, s) + U_vec[1]*diff(var, p)/s
    
    def test_M_1(self):
        assert pgeq.eqs_cg.M_1.lhs.equals(diff(cgvar.M_1, t))
        rhs_tmp = pgeq.eqs_cg.M_1.rhs.subs({H: H_s}).doit().subs({H_s: H})
        rhs_cmp = - self.u_dot_grad_h(cgvar.M_1) \
            + (diff(Us, s) - s*Us/2/H**2)*(cgvar.M_p + cgvar.M_m) \
            - I/2*(s*diff(Up/s, s) + diff(Us, p)/s)*(cgvar.M_p - cgvar.M_m)
        rhs_cmp = rhs_cmp.subs({H: H_s}).doit().subs({H_s: H})
        assert rhs_tmp.equals(rhs_cmp)
    
    def test_M_p(self):
        assert pgeq.eqs_cg.M_p.lhs.equals(diff(cgvar.M_p, t))
        rhs_tmp = pgeq.eqs_cg.M_p.rhs.subs({H: H_s}).doit().subs({H_s: H})
        rhs_cmp = - self.u_dot_grad_h(cgvar.M_p) \
            + cgvar.M_1*(2*diff(Us, s) - s*Us/H**2) \
            + I*cgvar.M_1*(s*diff(Up/s, s) + diff(Us, p)/s) \
            + I*cgvar.M_p*(s*diff(Up/s, s) - diff(Us, p)/s) 
        rhs_cmp = rhs_cmp.subs({H: H_s}).doit().subs({H_s: H})
        assert rhs_tmp.equals(rhs_cmp)
    
    def test_M_m(self):
        assert pgeq.eqs_cg.M_m.lhs.equals(diff(cgvar.M_m, t))
        rhs_tmp = pgeq.eqs_cg.M_m.rhs.subs({H: H_s}).doit().subs({H_s: H})
        rhs_cmp = - self.u_dot_grad_h(cgvar.M_m) \
            + cgvar.M_1*(2*diff(Us, s) - s*Us/H**2) \
            - I*cgvar.M_1*(s*diff(Up/s, s) + diff(Us, p)/s) \
            - I*cgvar.M_m*(s*diff(Up/s, s) - diff(Us, p)/s) 
        rhs_cmp = rhs_cmp.subs({H: H_s}).doit().subs({H_s: H})
        assert rhs_tmp.equals(rhs_cmp)
    
    def test_M_zp(self):
        assert pgeq.eqs_cg.M_zp.lhs.equals(diff(cgvar.M_zp, t))
        rhs_tmp = pgeq.eqs_cg.M_zp.rhs.subs({H: H_s}).doit().subs({H_s: H})
        rhs_cmp = - self.u_dot_grad_h(cgvar.M_zp) \
            + cgvar.M_zp/2*(3*diff(Uz, z) - I/s*diff(Us, p) + I*s*diff(Up/s, s)) \
            + cgvar.M_zm/2*(diff(Uz, z) + 2*diff(Us, s) + I/s*diff(Us, p) + I*s*diff(Up/s, s)) \
            - cgvar.zM_1/sympy.sqrt(2)*(diff(s*Us/H**2, s) + I/H**2*diff(Us, p)) \
            + cgvar.zM_p/sympy.sqrt(2)*(-diff(s*Us/H**2, s) + I/H**2*diff(Us, p))
        rhs_cmp = rhs_cmp.subs({H: H_s}).doit().subs({H_s: H})
        assert rhs_tmp.equals(rhs_cmp)
    
    def test_M_zm(self):
        assert pgeq.eqs_cg.M_zm.lhs.equals(diff(cgvar.M_zm, t))
        rhs_tmp = pgeq.eqs_cg.M_zm.rhs.subs({H: H_s}).doit().subs({H_s: H})
        rhs_cmp = - self.u_dot_grad_h(cgvar.M_zm) \
            + cgvar.M_zm/2*(3*diff(Uz, z) + I/s*diff(Us, p) - I*s*diff(Up/s, s)) \
            + cgvar.M_zp/2*(diff(Uz, z) + 2*diff(Us, s) - I/s*diff(Us, p) - I*s*diff(Up/s, s)) \
            - cgvar.zM_1/sympy.sqrt(2)*(diff(s*Us/H**2, s) - I/H**2*diff(Us, p)) \
            + cgvar.zM_m/sympy.sqrt(2)*(-diff(s*Us/H**2, s) - I/H**2*diff(Us, p))
        rhs_cmp = rhs_cmp.subs({H: H_s}).doit().subs({H_s: H})
        assert rhs_tmp.equals(rhs_cmp)

    def test_zM_1(self):
        assert pgeq.eqs_cg.zM_1.lhs.equals(diff(cgvar.zM_1, t))
        rhs_tmp = pgeq.eqs_cg.zM_1.rhs.subs({H: H_s}).doit().subs({H_s: H})
        rhs_cmp = - self.u_dot_grad_h(cgvar.zM_1) + diff(Uz, z)*cgvar.zM_1 \
            + (diff(Us, s) + diff(Uz, z)/2)*(cgvar.zM_p + cgvar.zM_m) \
            - I/2*(s*diff(Up/s, s) + diff(Us, p)/s)*(cgvar.zM_p - cgvar.zM_m)
        rhs_cmp = rhs_cmp.subs({H: H_s}).doit().subs({H_s: H})
        assert rhs_tmp.equals(rhs_cmp)
    
    def test_zM_p(self):
        assert pgeq.eqs_cg.zM_p.lhs.equals(diff(cgvar.zM_p, t))
        rhs_tmp = pgeq.eqs_cg.zM_p.rhs.subs({H: H_s}).doit().subs({H_s: H})
        rhs_cmp = - self.u_dot_grad_h(cgvar.zM_p) + diff(Uz, z)*cgvar.zM_p \
            + I*cgvar.zM_p*(s*diff(Up/s, s) - diff(Us, p)/s) \
            + cgvar.zM_1*(2*diff(Us, s) + diff(Uz, z) + I*s*diff(Up/s, s) + I/s*diff(Us, p))
        rhs_cmp = rhs_cmp.subs({H: H_s}).doit().subs({H_s: H})
        assert rhs_tmp.equals(rhs_cmp)
    
    def test_zM_m(self):
        assert pgeq.eqs_cg.zM_m.lhs.equals(diff(cgvar.zM_m, t))
        rhs_tmp = pgeq.eqs_cg.zM_m.rhs.subs({H: H_s}).doit().subs({H_s: H})
        rhs_cmp = - self.u_dot_grad_h(cgvar.zM_m) + diff(Uz, z)*cgvar.zM_m \
            - I*cgvar.zM_m*(s*diff(Up/s, s) - diff(Us, p)/s) \
            + cgvar.zM_1*(2*diff(Us, s) + diff(Uz, z) - I*s*diff(Up/s, s) - I/s*diff(Us, p))
        rhs_cmp = rhs_cmp.subs({H: H_s}).doit().subs({H_s: H})
        assert rhs_tmp.equals(rhs_cmp)
    
    def test_B_ep(self):
        assert pgeq.eqs_cg.B_ep.lhs.equals(diff(cgvar.B_ep, t))
        rhs_tmp = pgeq.eqs_cg.B_ep.rhs.subs({H: H_s}).doit().subs({H_s: H})
        rhs_cmp = - self.u_dot_grad_h(cgvar.B_ep) \
            + cgvar.B_ep/2*(diff(Us + I*Up, s) - I/s*diff(Us + I*Up, p) + (Us - I*Up)/s) \
            + cgvar.B_em/2*(diff(Us + I*Up, s) + I/s*diff(Us + I*Up, p) - (Us + I*Up)/s)
        rhs_cmp = rhs_cmp.subs({H: H_s}).doit().subs({H_s: H})
        assert rhs_tmp.equals(rhs_cmp)

    def test_B_em(self):
        assert pgeq.eqs_cg.B_em.lhs.equals(diff(cgvar.B_em, t))
        rhs_tmp = pgeq.eqs_cg.B_em.rhs.subs({H: H_s}).doit().subs({H_s: H})
        rhs_cmp = - self.u_dot_grad_h(cgvar.B_em) \
            + cgvar.B_ep/2*(diff(Us - I*Up, s) - I/s*diff(Us - I*Up, p) - (Us - I*Up)/s) \
            + cgvar.B_em/2*(diff(Us - I*Up, s) + I/s*diff(Us - I*Up, p) + (Us + I*Up)/s)
        rhs_cmp = rhs_cmp.subs({H: H_s}).doit().subs({H_s: H})
        assert rhs_tmp.equals(rhs_cmp)

    def test_dB_dz_ep(self):
        assert pgeq.eqs_cg.dB_dz_ep.lhs.equals(diff(cgvar.dB_dz_ep, t))
        rhs_tmp = pgeq.eqs_cg.dB_dz_ep.rhs.subs({H: H_s}).doit().subs({H_s: H})
        rhs_cmp = - self.u_dot_grad_h(cgvar.dB_dz_ep) - diff(Uz, z)*cgvar.dB_dz_ep \
            + cgvar.dB_dz_ep/2*(diff(Us + I*Up, s) - I/s*diff(Us + I*Up, p) + (Us - I*Up)/s) \
            + cgvar.dB_dz_em/2*(diff(Us + I*Up, s) + I/s*diff(Us + I*Up, p) - (Us + I*Up)/s)
        rhs_cmp = rhs_cmp.subs({H: H_s}).doit().subs({H_s: H})
        assert rhs_tmp.equals(rhs_cmp)

    def test_dB_dz_em(self):
        assert pgeq.eqs_cg.dB_dz_em.lhs.equals(diff(cgvar.dB_dz_em, t))
        rhs_tmp = pgeq.eqs_cg.dB_dz_em.rhs.subs({H: H_s}).doit().subs({H_s: H})
        rhs_cmp = - self.u_dot_grad_h(cgvar.dB_dz_em) - diff(Uz, z)*cgvar.dB_dz_em \
            + cgvar.dB_dz_ep/2*(diff(Us - I*Up, s) - I/s*diff(Us - I*Up, p) - (Us - I*Up)/s) \
            + cgvar.dB_dz_em/2*(diff(Us - I*Up, s) + I/s*diff(Us - I*Up, p) + (Us + I*Up)/s)
        rhs_cmp = rhs_cmp.subs({H: H_s}).doit().subs({H_s: H})
        assert rhs_tmp.equals(rhs_cmp)
