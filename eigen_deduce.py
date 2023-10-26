# -*- coding: utf-8 -*-
"""Equation deduction of the eigenvalue problem.
Jingtao Min @ ETH Zurich 2023

This script is part of the routine to solve an eigenvalue problem
in the framework of PG model.
When in doubt, it is still recommended to use an interactive session
such as IPython notebook (recommended), IDLE or IPython Console to
interactively verify the correctness of the equations/elements.
"""


import os
from sympy import *
import pg_utils.sympy_supp.vector_calculus_3d as v3d

from pg_utils.pg_model import *
from pg_utils.pg_model import base, core, params, forcing
from pg_utils.pg_model import base_utils as pgutils
from pg_utils.pg_model import equations as pgeq

import bg_malkus as bg_cfg


def apply_components(eqs: base.LabeledCollection, *args: str, 
    timescale: str ="Alfven", verbose: int=0) -> base.LabeledCollection:
    """Add components of the forcing in the PG system
    
    :param eqs: original set of equations
    :param *args: strings of components to be added.
        The following are implemented (case-insensitive):
        - 'Lorentz': Lorentz force terms; assumes vorticity equation exists
    :param timescale: type of characteristic time scale for nondimensionalization.
        Choice of timescale only influences the vorticity equation
        The following are implemented (case-insensitive):
        - 'Alfven': use Alfven time scales T = B/sqrt(rho_0*mu_0) (default)
        - 'Spin': use inverse spin rate T = 1/Omega
    :param verbose: verbosity level
    :returns: set of equations with forcing added
    """
    
    if verbose > 0:
        print("========== Assembling components... ==========")
    
    # Convert component names to lower case and remove duplicate
    components = {arg.lower() for arg in args}
    
    # Initiate new copy
    eqs_new = eqs.copy()
    fe_p = fs_sym = fp_sym = fz_asym = S.Zero
    
    # Collect components
    for comp in components:
        if verbose > 1:
            print("Collecting %s..." % (components,))
        if comp == "lorentz":
            assert "Psi" in eqs._field_names and eqs.Psi is not None
            fe_p += forcing.Le_p
            fs_sym += forcing.Ls_sym
            fp_sym += forcing.Lp_sym
            fz_asym += forcing.Lz_asym
        else:
            raise TypeError
    
    # Treatment of vorticity equation
    if "Psi" in eqs._field_names and eqs.Psi is not None:
        if verbose > 1:
            print("Adding body forces to vorticity equation...")
        # Separate Coriolis and body force
        term_coriolis = term_body_force = S.Zero
        for term in eqs.Psi.rhs.expand().args:
            funcs = term.atoms(Function)
            if pgeq.fe_p in funcs or pgeq.fs_sym in funcs \
                or pgeq.fp_sym in funcs or pgeq.fz_asym in funcs:
                term_body_force += term
            else:
                term_coriolis += term
        # Choose prefactors based on time scale
        if verbose > 0:
            print("Using %s time scale in vorticity equation" % (timescale,))
        if timescale.lower() == "alfven":
            prefactor_f, prefactor_coriolis = S.One, 1/params.Le
        elif timescale.lower() == "spin":
            prefactor_f, prefactor_coriolis = params.Le**2, S.One
        # Assemble terms
        eqs_new.Psi = Eq(
            eqs.Psi.lhs, 
            prefactor_coriolis*term_coriolis 
            + prefactor_f*term_body_force.subs({
                pgeq.fe_p: fe_p, pgeq.fs_sym: fs_sym, 
                pgeq.fp_sym: fp_sym, pgeq.fz_asym: fz_asym}).expand())
    
    return eqs_new


def apply_bg_eqwise(fname: str, eq: Eq, bg_map: dict, verbose: int = 0) -> Eq:
    """Apply background field equation-wise
    
    :param fname: name of the field / equation
    :param eq: a sympy equation
    :param bg_map: background dictionary
    """
    if verbose > 0:
        print("Applying background field to %s equation..." % (fname,))
    # Treatment of vorticity equation: do not try to "simplify" the RHS of vorticity
    # equation; it often leads to unnecessary rationalization
    if fname == "Psi":
        new_lhs = eq.lhs.subs(bg_map).subs({H: H_s}).doit().subs({H_s: H}).expand()
        new_rhs = eq.rhs.subs(bg_map).subs({H: H_s}).doit().subs({H_s: H}).expand()
    # For other equations: try to simplify for visual simplicity
    # !!!!! ==================================================== Note ===========
    # If the code is not used interactively, perhaps all simplify can be skipped?
    else:
        new_lhs = eq.lhs.subs(bg_map).subs({H: H_s}).doit().simplify()
        new_rhs = eq.rhs.subs(bg_map).subs({H: H_s}).doit().simplify()
        # Take z to +H or -H at the boundaries or to 0 at the equatorial plane
        if fname in base.CollectionPG.pg_field_names[-6:-3]:
            new_rhs = new_rhs.subs({z: +H}).doit().simplify()
        elif fname in base.CollectionPG.pg_field_names[-3:]:
            new_rhs = new_rhs.subs({z: -H}).doit().simplify()
        elif fname in base.CollectionPG.pg_field_names[-11:-6]:
            new_rhs = new_rhs.subs({z: S.Zero}).doit().simplify()
        new_lhs = new_lhs.subs({H_s: H}).expand()
        new_rhs = new_rhs.subs({H_s: H}).expand()
    return Eq(new_lhs, new_rhs)


def apply_bg_field(eqs: base.LabeledCollection, U0_val: v3d.Vector3D, 
    B0_val: v3d.Vector3D, verbose: int = 0) -> base.LabeledCollection:
    """Apply background field to obtain explicit equations
    
    :param eqs: original set of equations
    :param U0_val: expression for background velocity field 
        in vector components, in cylindrical coordinates
    :param B0_val: expression for background magnetic field 
        in vector components, in cylindrical coordinates
    :returns: new set of equations
    """
    if verbose > 0:
        print("========== Applying background field ==========")
    pg0_val = pgutils.assemble_background(B0=B0_val)
    bg_sub = {u_comp: U0_val[i_c] for i_c, u_comp in enumerate(core.U0_vec)}
    bg_sub.update({b_comp: B0_val[i_c] for i_c, b_comp in enumerate(core.B0_vec)})
    bg_sub.update({pg_comp: pg0_val[i_c] for i_c, pg_comp in enumerate(core.pgvar_bg)})
    eqs_new = eqs.apply(
        lambda fname, eq: apply_bg_eqwise(fname, eq, bg_sub, verbose=verbose - 1), 
        inplace=False, metadata=True)
    return eqs_new


if __name__ == "__main__":
    eqs = apply_components(pgeq.eqs_pg_lin, "Lorentz", timescale="Alfven", verbose=2)
    eqs.Psi = Eq(eqs.Psi.lhs, eqs.Psi.rhs.subs(forcing.force_explicit_lin))
    eqs = apply_bg_field(eqs, bg_cfg.U0_val, bg_cfg.B0_val, verbose=2)
    with open("./out/symbolic/Malkus/eqs_ptb_deduce.json", 'x') as fwrite:
        eqs.save_json(fwrite, serializer=srepr)
    
