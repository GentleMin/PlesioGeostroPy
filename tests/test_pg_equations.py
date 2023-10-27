# -*- coding: utf-8 -*-
"""Testing if the modularized 
"""

from sympy import parse_expr
import pytest
from pg_utils.pg_model import base


class TestPGEquations:
    
    @pytest.fixture(autouse=True)
    def load_equations(self):
        eq_base_path = "./out/symbolic/eqs_pg_lin.json"
        eq_comp_path = "./out/symbolic/eqs_pg_lin_simp.json"

        with open(eq_base_path, 'r') as fread:
            self.eq_base = base.LabeledCollection.load_json(fread, parser=parse_expr)
        with open(eq_comp_path, 'r') as fread:
            self.eq_comp = base.LabeledCollection.load_json(fread, parser=parse_expr)
    
    def test_Psi(self):
        fname = "Psi"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_Mss(self):
        fname = "Mss"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_Mpp(self):
        fname = "Mpp"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
        
    def test_Msp(self):
        fname = "Msp"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_Msz(self):
        fname = "Msz"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)

    def test_Mpz(self):
        fname = "Mpz"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)

    def test_zMss(self):
        fname = "zMss"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_zMpp(self):
        fname = "zMpp"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
        
    def test_zMsp(self):
        fname = "zMsp"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)

    def test_Bs_e(self):
        fname = "Bs_e"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_Bp_e(self):
        fname = "Bp_e"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
        
    def test_Bz_e(self):
        fname = "Bz_e"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
        
    def test_dBs_dz_e(self):
        fname = "dBs_dz_e"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)

    def test_dBp_dz_e(self):
        fname = "dBp_dz_e"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_Br_b(self):
        fname = "Br_b"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_Bs_p(self):
        fname = "Bs_p"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_Bp_p(self):
        fname = "Bp_p"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
        
    def test_Bz_p(self):
        fname = "Bz_p"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_Bs_m(self):
        fname = "Bs_m"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
    
    def test_Bp_m(self):
        fname = "Bp_m"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
        
    def test_Bz_m(self):
        fname = "Bz_m"
        assert self.eq_base[fname].lhs.equals(self.eq_comp[fname].lhs)
        assert self.eq_base[fname].rhs.equals(self.eq_comp[fname].rhs)
