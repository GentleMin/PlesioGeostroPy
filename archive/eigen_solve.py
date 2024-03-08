# -*- coding: utf-8 -*-
"""Equation deduction of the eigenvalue problem.
Jingtao Min @ ETH Zurich 2023

This script is the routine to solve an eigenvalue problem
in the framework of PG model.
When in doubt, it is still recommended to use an interactive session
such as IPython notebook (recommended), IDLE or IPython Console to
interactively verify the correctness of the equations/elements.
"""

PRECOMPUTE_EQN = True
VARIABLE_MODE = "CG"

assert VARIABLE_MODE in ("PG", "CG", "Psi-F")

import os, h5py, json
from typing import List, Any, Optional, Union
from sympy import S, I, oo, Integer, Rational, Function, \
    Add, diff, Eq, srepr, parse_expr, sqrt
import numpy as np
import pg_utils.sympy_supp.vector_calculus_3d as v3d

from pg_utils.pg_model import *
from pg_utils.pg_model import base, core, params, forcing
from pg_utils.pg_model import base_utils as pgutils
from pg_utils.pg_model import expansion as xpd
from pg_utils.pg_model.expansion import omega, n, m, xi, n_test, n_trial

from pg_utils.numerics import matrices as nmatrix
from pg_utils.numerics import io as num_io

from scipy.sparse import coo_array

# Background field setup
import bg_hydrodynamic as bg_cfg

# Radial expansion setup
# from pg_utils.pg_model import expand_daria_mm_malkus as xpd_cfg
# from pg_utils.pg_model import expand_daria_thesis as xpd_cfg
from pg_utils.pg_model import expand_conjugate as xpd_cfg
# from pg_utils.pg_model import expand_stream_force as xpd_cfg

# Physical variables
PHYS_PARAMS = {
    # bg_cfg.cf_gamma: 3*np.sqrt(3)/2,
    m: Integer(3), 
    params.Le: Rational(1, 10000), 
    params.Lu: oo
}

if not PRECOMPUTE_EQN:
    from pg_utils.pg_model import equations as pgeq



"""Equation derivation and deduction utilities"""


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
            if core.fe_p in funcs or core.fs_sym in funcs \
                or core.fp_sym in funcs or core.fz_asym in funcs:
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
                core.fe_p: fe_p, core.fs_sym: fs_sym, 
                core.fp_sym: fp_sym, core.fz_asym: fz_asym}).expand())
    
    return eqs_new


def apply_bg_eqwise(fname: str, eq: Eq, bg_map: dict, mode: str = "PG",
    verbose: int = 0) -> Eq:
    """Apply background field equation-wise
    
    :param fname: name of the field / equation
    :param eq: a sympy equation
    :param bg_map: background dictionary
    """
    if mode.lower() == "pg":
        fnames = base.CollectionPG.pg_field_names
    elif mode.lower() == "cg":
        fnames = base.CollectionConjugate.cg_field_names
    else:
        raise TypeError
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
        if fname in fnames[-6:-3]:
            new_rhs = new_rhs.subs({z: +H}).doit().simplify()
        elif fname in fnames[-3:]:
            new_rhs = new_rhs.subs({z: -H}).doit().simplify()
        elif fname in fnames[-11:-6]:
            new_rhs = new_rhs.subs({z: S.Zero}).doit().simplify()
        new_lhs = new_lhs.subs({H_s: H}).expand()
        new_rhs = new_rhs.subs({H_s: H}).expand()
    return Eq(new_lhs, new_rhs)


def apply_bg_field(eqs: base.LabeledCollection, U0_val: v3d.Vector3D, 
    B0_val: v3d.Vector3D, mode: str="PG", verbose: int = 0) -> base.LabeledCollection:
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
    # Building the background field map
    f0_val = pgutils.assemble_background(B0=B0_val, mode=mode)
    bg_sub = {u_comp: U0_val[i_c] for i_c, u_comp in enumerate(core.U0_vec)}
    bg_sub.update({b_comp: B0_val[i_c] for i_c, b_comp in enumerate(core.B0_vec)})
    if mode.lower() == "pg":
        bg_sub.update({comp: f0_val[i_c] for i_c, comp in enumerate(core.pgvar_bg)})
    elif mode.lower() == "cg":
        bg_sub.update({comp: f0_val[i_c] for i_c, comp in enumerate(core.cgvar)})
    eqs_new = eqs.apply(
        lambda fname, eq: apply_bg_eqwise(fname, eq, bg_sub, mode=mode, verbose=verbose-1), 
        inplace=False, metadata=True)
    return eqs_new



"""Dimensional reduction utilities"""


def reduce_eqsys_to_force_form(eqsys_old: base.LabeledCollection, 
    verbose: int = 0) -> base.LabeledCollection:
    """Reduce a system of equations to streamfunction-force formulation,
    thus drastically reducing the dimensionality of the dynamical system
    """
    if verbose > 0:
        print("========== Reducing dimension of dynamical system... ==========")
    assert "Psi" in eqsys_old._field_names and eqsys_old.Psi is not None
    
    # Extract dynamical variables other than Psi
    dynamic_vars = tuple(
        tuple(eqsys_old[fname].lhs.atoms(Function))[0] 
        for fname in eqsys_old._field_names if fname != "Psi"
    )
    
    # Process the vorticity equation
    if verbose > 1:
        print("Extracting body forces...")
    psi_term = f_term = S.Zero
    for term in eqsys_old.Psi.rhs.expand().args:
        is_dynamic_var = False
        for func in term.atoms(Function):
            if func in dynamic_vars:
                is_dynamic_var = True
                f_term += term
                break
        if not is_dynamic_var:
            psi_term += term
    
    # Form evolution equation for external force
    if verbose > 1:
        print("Forming dynamical system of Psi and F...")
    dynamic_subs = {eqsys_old[fname].lhs: eqsys_old[fname].rhs
        for fname in eqsys_old._field_names if fname != "Psi"}
    f_term = diff(f_term, t).doit().subs(dynamic_subs).doit()
    f_term = Add(*[term.subs({H: H_s}).doit().subs({H_s: H}).expand()
        for term in f_term.args])
    eqsys_new = base.LabeledCollection(
        ["Psi", "F_ext"], 
        Psi = Eq(eqsys_old.Psi.lhs, psi_term + core.reduced_var.F_ext),
        F_ext = Eq(diff(core.reduced_var.F_ext, t), f_term)
    )
    return eqsys_new

    

"""Matrix collection utilities"""


def process_matrix_element(element: Any, map_trial: dict, map_test: dict) -> Any:
    """Process matrix elements to desired form
    
    :param element: original element
    :param map_trial: dictionary to substitute trial functions into the expr
    :param map_test: dictionary to substitute test functions into the expr
    :return element: new element
    """
    if element is None or element == S.Zero or element == 0:
        return S.Zero
    elif isinstance(element, xpd.InnerProduct1D):
        element = element.subs(map_trial).subs(map_test)
        element = element.subs({H_s: H, H_s**2: H**2}).expand().subs({H: H_s})
        return element.change_variable(xi, xpd.s_xi, xpd.xi_s, 
            jac_positive=True, merge=True, simplify=False)
    else:
        raise TypeError


def collect_eigen_matrices(eqs: base.LabeledCollection, 
    xpd_recipe: xpd.ExpansionRecipe, inplace: bool = False, 
    verbose: int = 0) -> List[xpd.SystemMatrix]:
    """Collect matrix elements
    
    :param eqs: LabeledCollection containing all equations
    :param xpd_recipe: expansion recipe
    :param inplace: whether allow modifying the eqs inplace
    :returns: Mass matrix, stiffness matrix
    """
    if verbose > 0:
        print("========== Converting eqn to matrices... ==========")
    # Collect matrices
    eqs_sys = xpd.SystemEquations(eqs._field_names, xpd_recipe, 
        **{fname: eqs[fname] for fname in eqs._field_names})
    if verbose > 2:
        print("Converting equations to Fourier domain...")
    eqs_sys = eqs_sys.to_fourier_domain(inplace=inplace)
    if verbose > 2:
        print("Converting equations to radial coordinates...")
    eqs_sys = eqs_sys.to_radial(inplace=inplace)
    if verbose > 2:
        print("Converting equations to inner products...")
    eqs_sys = eqs_sys.to_inner_product(factor_lhs=I*omega, inplace=inplace)
    M_expr, K_expr = eqs_sys.collect_matrices(factor_lhs=I*omega)
    # Convert matrix elements to desired form
    if verbose > 1:
        print("Collecting and converting mass matrix M elements...")
    M_expr = M_expr.apply(
        lambda x: process_matrix_element(x, xpd_recipe.base_expr, xpd_recipe.test_expr),
        inplace=True, metadata=False
    )
    if verbose > 1:
        print("Collecting and converting stiffness matrix K elements...")
    K_expr = K_expr.apply(
        lambda x: process_matrix_element(x, xpd_recipe.base_expr, xpd_recipe.test_expr),
        inplace=True, metadata=False
    )
    return M_expr, K_expr



"""Top-level functions
Top level functions are mainly functions wrapped with input/output
utilities to further simplify the procedure
"""


def routine_eqn_formation(read_from: str = "default", components: List = ["Lorentz"],
    timescale: str = "Alfven", save_to: Optional[str]=None) -> base.LabeledCollection:
    """Top-level routine function
    Stage 1: equation derivation
    """
    # Input
    if PRECOMPUTE_EQN:
        if read_from == "default":
            if VARIABLE_MODE == "PG":
                with open("./out/symbolic/eqs_pg_lin.json", 'r') as fread:
                    eqs = base.CollectionPG.load_json(fread, parser=parse_expr)
            elif VARIABLE_MODE == "CG":
                with open("./out/symbolic/eqs_cg_lin.json", 'r') as fread:
                    eqs = base.CollectionConjugate.load_json(fread, parser=parse_expr)
        else:
            with open(read_from, 'r') as fread:
                eqs = base.LabeledCollection.load_json(fread, parser=parse_expr)
    else:
        if VARIABLE_MODE == "PG":
            eqs = pgeq.eqs_pg_lin
        elif VARIABLE_MODE == "CG":
            eqs = pgeq.eqs_cg_lin
    # Body
    eqs = apply_components(eqs, *components, timescale=timescale, verbose=2)
    if VARIABLE_MODE == "PG":
        eqs.Psi = Eq(eqs.Psi.lhs, eqs.Psi.rhs.subs(forcing.force_explicit_lin))
    elif VARIABLE_MODE == "CG":
        eqs.Psi = Eq(eqs.Psi.lhs, eqs.Psi.rhs.subs(forcing.force_explicit_lin_cg))
    eqs = apply_bg_field(eqs, bg_cfg.U0_val, bg_cfg.B0_val, 
        mode=VARIABLE_MODE, verbose=2)
    # Output
    if save_to is not None:
        fdir = os.path.dirname(save_to)
        os.makedirs(fdir, exist_ok=True)
        with open(save_to, 'x') as fwrite:
            eqs.save_json(fwrite, serializer=srepr)
    return eqs


def routine_dim_reduction(read_from: Union[str, base.LabeledCollection], 
    save_to: Optional[str]=None) -> base.LabeledCollection:
    """Top-level routine function
    Stage 1b: dimension reduction of the equations
    """
    # Input
    if isinstance(read_from, str):
        with open(read_from, 'r') as fread:
            eqs = base.LabeledCollection.load_json(fread, parser=parse_expr)
    elif isinstance(read_from, base.LabeledCollection):
        eqs = read_from
    else:
        raise TypeError
    # Body
    eqs_new = reduce_eqsys_to_force_form(eqs, verbose=5)
    # Output
    if save_to is not None:
        fdir = os.path.dirname(save_to)
        os.makedirs(fdir, exist_ok=True)
        with open(save_to, 'x') as fwrite:
            eqs_new.save_json(fwrite, serializer=srepr)
    return eqs_new


def routine_matrix_collection(read_from: Union[str, base.LabeledCollection], 
    save_to: Optional[str]=None) -> List[xpd.SystemMatrix]:
    """Top-level routine function
    Stage 2: matrix collection
    """
    # Input
    if isinstance(read_from, str):
        with open(read_from, 'r') as fread:
            eqs = base.LabeledCollection.load_json(fread, parser=parse_expr)
    elif isinstance(read_from, base.LabeledCollection):
        eqs = read_from
    else:
        raise TypeError
    # Body
    M_expr, K_expr = collect_eigen_matrices(eqs, 
        xpd_cfg.recipe, inplace=False, verbose=5)
    # Output
    if save_to is not None:
        fdir = os.path.dirname(save_to)
        os.makedirs(fdir, exist_ok=True)
        # if save_to[-5:] == '.json':
        #     save_to = save_to[:-5]
        # with open(save_to + "_M.json", 'x') as fwrite:
        #     M_expr.save_json(fwrite)
        # with open(save_to + "_K.json", 'x') as fwrite:
        #     K_expr.save_json(fwrite)
        serialized_obj = {
            "xpd": xpd_cfg.identifier,
            "M": M_expr.serialize(),
            "K": K_expr.serialize()
        }
        with open(save_to, 'x') as fwrite:
            json.dump(serialized_obj, fwrite, indent=4)
    return M_expr, K_expr


def routine_matrix_calculation(read_from: Union[str, List[xpd.SystemMatrix]], 
    Ntrunc: int = 5, xpd_recipe: xpd.ExpansionRecipe = xpd_cfg.recipe,
    save_to: Optional[str] = None, outer: bool = True, 
    verbose: int = 0) -> List[np.ndarray]:
    """Top-level routine function
    Stage 3: Routine calculation of matrix elements
    """
    
    # Input
    if isinstance(read_from, str):
        # if read_from[-5:] == '.json':
        #     read_from = read_from[:-5]
        # with open(read_from + '_M.json', 'r') as fread:
        #     M_expr = xpd.SystemMatrix.load_json(fread)
        # with open(read_from + '_K.json', 'r') as fread:
        #     K_expr = xpd.SystemMatrix.load_json(fread)
        with open(read_from, 'r') as fread:
            matrix_obj = json.load(fread)
        M_expr = xpd.SystemMatrix.deserialize(matrix_obj["M"])
        K_expr = xpd.SystemMatrix.deserialize(matrix_obj["K"])
    else:
        M_expr = read_from[0]
        K_expr = read_from[1]
    
    if verbose > 0:
        print("========== Expanding matrices... ==========")
    
    if verbose > 1:
        print("Pre-processing elements...")
    # Pre-processing of elements
    M_expr.apply(lambda ele: ele.subs(PHYS_PARAMS), inplace=True, metadata=False)
    K_expr.apply(lambda ele: ele.subs(PHYS_PARAMS), inplace=True, metadata=False)
    
    if verbose > 1:
        print("Configuring expansions and quadratures...")    
    # Configure expansions
    fnames = xpd_recipe.rad_xpd.fields._field_names
    cnames = xpd_recipe.rad_xpd.bases._field_names
    if VARIABLE_MODE in ("PG", "CG"):
        ranges_trial = [
            np.arange(2*Ntrunc + 1) if 'M' in cname else np.arange(Ntrunc + 1)
            for cname in cnames]
        ranges_test = [
            np.arange(2*Ntrunc + 1) if 'M' in fname else np.arange(Ntrunc + 1)
            for fname in fnames]
    elif VARIABLE_MODE == "Psi-F":
        ranges_trial = [np.arange(Ntrunc + 1), np.arange(Ntrunc + 1)]
        ranges_test = [np.arange(Ntrunc + 1), np.arange(Ntrunc + 1)]
    
    # Configure quadrature
    quad_recipe_list = np.array([
        [nmatrix.QuadRecipe(
            init_opt={"automatic": True, "quadN": None},
            gram_opt={"backend": "scipy", "output": "numpy", "outer": outer}
        ) for ele in row] for row in M_expr._matrix
    ])
    
    if verbose > 1:
        print("Computing quadratures of elements...")
    # Computation
    M_val = nmatrix.MatrixExpander(
        M_expr, quad_recipe_list, ranges_trial, ranges_test).expand(verbose=verbose > 1)
    K_val = nmatrix.MatrixExpander(
        K_expr, quad_recipe_list, ranges_trial, ranges_test).expand(verbose=verbose > 1)
    
    # Output
    if save_to is not None:
        fdir = os.path.dirname(save_to)
        os.makedirs(fdir, exist_ok=True)
        with h5py.File(save_to, 'x') as fwrite:
            str_type = h5py.string_dtype(encoding="utf-8")
            fwrite.attrs["xpd"] = xpd_cfg.identifier
            fwrite.attrs["azm"] = int(PHYS_PARAMS[m])
            fwrite.attrs["Le"] = float(PHYS_PARAMS[params.Le])
            fwrite.attrs["Lu"] = float(PHYS_PARAMS[params.Lu]) \
                if not PHYS_PARAMS[params.Lu].equals(oo) else "+inf"
            gp = fwrite.create_group("rows")
            gp.create_dataset("names", data=fnames, dtype=str_type)
            gp.create_dataset("ranges", 
                data=np.array([len(nrange) for nrange in ranges_test]))
            gp = fwrite.create_group("cols")
            gp.create_dataset("names", data=cnames, dtype=str_type)
            gp.create_dataset("ranges", 
                data=np.array([len(nrange) for nrange in ranges_trial]))            
            # fwrite.create_dataset("M", data=M_val)
            # fwrite.create_dataset("K", data=K_val)
            matrix_gp = fwrite.create_group("M")
            num_io.sparse_coo_save_to_group(coo_array(M_val), matrix_gp)
            matrix_gp = fwrite.create_group("K")
            num_io.sparse_coo_save_to_group(coo_array(K_val), matrix_gp)
    return M_val, K_val


def routine_eigen_compute(read_from: Union[str, List[np.ndarray]], 
    save_to: Optional[str], verbose: int = 0) -> List[np.ndarray]:
    """Top-level routine function
    Stage 4: Computing eigenvalues and eigenvectors from Matrices
    """
    # Input
    if isinstance(read_from, str):
        with h5py.File(read_from, 'r') as fread:
            identifier = fread.attrs["xpd"]
            m_val = fread.attrs["azm"]
            Le = fread.attrs["Le"]
            Lu = fread.attrs["Lu"]
            fnames = list(fread["rows"]["names"].asstr()[()])
            frange = fread["rows"]["ranges"][()]
            cnames = list(fread["cols"]["names"].asstr()[()])
            crange = fread["cols"]["ranges"][()]
            M_val = num_io.matrix_load_from_group(fread["M"]).todense()
            K_val = num_io.matrix_load_from_group(fread["K"]).todense()
    else:
        M_val = read_from[0]
        K_val = read_from[1]
    
    if verbose > 0:
        print("========== Calculating eigenvalues... ==========")
    
    # Convert to ordinary eigenvalue problem and solve
    assert np.linalg.cond(M_val) < 1e+8
    A_val = np.linalg.inv(M_val) @ K_val
    eig_val, eig_vec = np.linalg.eig(A_val)
    
    # Sorting
    eig_sort = np.argsort(-np.abs(eig_val))
    eig_val = eig_val[eig_sort]
    eig_vec = eig_vec[:, eig_sort]
    
    # Output
    if save_to is not None:
        fdir = os.path.dirname(save_to)
        os.makedirs(fdir, exist_ok=True)
        with h5py.File(save_to, 'x') as fwrite:
            str_type = h5py.string_dtype(encoding="utf-8")
            fwrite.attrs["xpd"] = identifier
            fwrite.attrs["azm"] = m_val
            fwrite.attrs["Le"] = Le
            fwrite.attrs["Lu"] = Lu
            gp = fwrite.create_group("bases")
            gp.create_dataset("names", data=cnames, dtype=str_type)
            gp.create_dataset("ranges", data=np.asarray(crange))            
            fwrite.create_dataset("eigval", data=eig_val)
            fwrite.create_dataset("eigvec", data=eig_vec)
    return eig_val, eig_vec



if __name__ == "__main__":
    
    # fname_pgeq = "./out/symbolic/eqs_pg_lin.json"
    output_dir = "./out/cases/Hydrodynamic/Recipe_Conjugate/"
    # output_dir = "./out/cases/Malkus/Conjugate_recipe"
    # output_dir = "./out/cases/Toroidal_quadrupolar/Recipe_Daria/"
    # output_dir = "./out/tmp/"
    prefix = ''
    suffix = ''
    
    """Stage 1: equation formation, apply background"""
    # fname_eqn = os.path.join(os.path.dirname(os.path.dirname(output_dir)), 
    #     prefix + "Eqs_ptb" + suffix + ".json")
    # fname_eqn = os.path.join(os.path.dirname(os.path.dirname(output_dir)), 
    #     prefix + "Eqs_conjugate_ptb" + suffix + ".json")
    # routine_eqn_formation(read_from="default", timescale="spin", save_to=fname_eqn)
    
    """Stage 1: dimension reduction"""
    # fname_eqn = "./out/cases/Malkus/Eqs_ptb.json"
    # fname_eqn_reduced = os.path.join(os.path.dirname(os.path.dirname(output_dir)), 
    #     prefix + "Eqs_reduced" + suffix + ".json")
    # routine_dim_reduction(fname_eqn, save_to=fname_eqn_reduced)
    
    """Stage 2: matrix extraction"""
    # Choose equations
    # with open(fname_eqn, 'r') as fread:
    #     eqs = base.LabeledCollection.load_json(fread, parser=parse_expr)
    # eqs_solve = eqs
    # solve_idx = np.full(21, False)
    # solve_idx[:14] = True
    # eqs_solve = eqs.generate_collection(solve_idx)
    fname_matrix_expr = os.path.join(output_dir, 
        prefix + "Matrix_expr" + suffix + ".json")
    # routine_matrix_collection(eqs_solve, save_to=fname_matrix_expr)
    
    """Stage 3: compute matrices"""
    Ntrunc = 100
    fname_matrix_val = os.path.join(output_dir, 
        prefix + "Matrix_eval_N{:d}".format(Ntrunc) + suffix + ".h5")
    routine_matrix_calculation(fname_matrix_expr, Ntrunc=Ntrunc, 
        xpd_recipe=xpd_cfg.recipe, save_to=fname_matrix_val, verbose=5)
    
    """Stage 4: compute eigenvalues"""
    fname_eig_result = os.path.join(output_dir, 
        prefix + "Eigen_N{:d}".format(Ntrunc) + suffix + ".h5")
    routine_eigen_compute(fname_matrix_val, save_to=fname_eig_result, verbose=5)
    
