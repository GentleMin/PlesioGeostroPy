# -*- coding: utf-8 -*-
"""Testing if the modularized 
"""

from sympy import parse_expr
import pytest
from pg_utils.pg_model import base


class TestConjugateEquations:
    
    @pytest.fixture(autouse=True)
    def load_equations(self):
        eq_base_path = "./out/symbolic/eqs_cg_lin.json"
        eq_comp_path = "./out/symbolic/eqs_cg_lin_simp.json"

        with open(eq_base_path, 'r') as fread:
            self.eq_base = base.LabeledCollection.load_json(fread, parser=parse_expr)
        with open(eq_comp_path, 'r') as fread:
            self.eq_comp = base.LabeledCollection.load_json(fread, parser=parse_expr)
    
    def test_Psi(self):
        fname = "Psi"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_M_1(self):
        fname = "M_1"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_M_p(self):
        fname = "M_p"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
        
    def test_M_m(self):
        fname = "M_m"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_M_zp(self):
        fname = "M_zp"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)

    def test_M_zm(self):
        fname = "M_zm"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)

    def test_zM_1(self):
        fname = "zM_1"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_zM_p(self):
        fname = "zM_p"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
        
    def test_zM_m(self):
        fname = "zM_m"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)

    def test_B_ep(self):
        fname = "B_ep"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_B_em(self):
        fname = "B_em"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
        
    def test_Bz_e(self):
        fname = "Bz_e"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
        
    def test_dB_dz_ep(self):
        fname = "dB_dz_ep"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)

    def test_dB_dz_em(self):
        fname = "dB_dz_em"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_Br_b(self):
        fname = "Br_b"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_B_pp(self):
        fname = "B_pp"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_B_pm(self):
        fname = "B_pm"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
        
    def test_Bz_p(self):
        fname = "Bz_p"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_B_mp(self):
        fname = "B_mp"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_B_mm(self):
        fname = "B_mm"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
        
    def test_Bz_m(self):
        fname = "Bz_m"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
