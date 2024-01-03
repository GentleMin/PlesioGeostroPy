# -*- coding: utf-8 -*-
"""Solving the eigenvalue problem.

This file contains the code for solving an eigenvalue problem
in the framework of PG model.
"""

import os, h5py, json, warnings
from typing import List, Any, Optional, Union, Tuple
from sympy import S, I, Function, Add, diff, Eq, srepr, parse_expr

from .pg_model import *
from .pg_model import base, core, params, forcing, bg_fields
from .pg_model import base_utils as pgutils
from .pg_model import expansion as xpd
from .pg_model.expansion import omega, xi, m

from .numerics import matrices as nmatrix
from .numerics import io as num_io

import numpy as np
from scipy.sparse import coo_array


"""Equation derivation and deduction utilities"""


def assemble_forcing(eqs: base.LabeledCollection, *args: str, 
    timescale: str ="Alfven", verbose: int=0) -> Tuple[base.LabeledCollection, List]:
    """Assemble components of the forcing in the PG system
    
    :param base.LabeledCollection eqs: original set of equations
    :param str *args: strings of components to be added.
        The following are implemented (case-insensitive):
        * 'Lorentz': Lorentz force terms; assumes vorticity equation exists
        
    :param str timescale: type of char. time scale for nondimensionalization.
        Choice of timescale only influences the vorticity equation
        The following are implemented (case-insensitive):
        * 'Alfven': use Alfven time scales T = B/sqrt(rho_0*mu_0) (default)
        * 'Spin': use inverse spin rate T = 1/Omega
    
    :param int verbose: verbosity level
    :returns: set of equations with forcing added, and the unknown variables
    """
    
    if verbose > 0:
        print("========== Assembling components... ==========")
    
    par_list = list()
    
    # Convert component names to lower case and remove duplicate
    components = {arg.lower() for arg in args}
    
    # Initiate new copy
    eqs_new = eqs.copy()
    fe_p = fs_sym = fp_sym = fz_asym = S.Zero

    # Choose prefactors based on time scale
    if verbose > 0:
        print("Using %s time scale in vorticity equation" % (timescale,))
    if timescale.lower() == "alfven":
        if "lorentz" not in components:
            warnings.warn("No Lorentz force in the system! "
                "Alfven timescale may be undefined.")
        prefactor_lorentz, prefactor_coriolis = S.One, 1/params.Le
    elif timescale.lower() == "spin":
        prefactor_lorentz, prefactor_coriolis = params.Le**2, S.One
    
    # Collect components
    for comp in components:
        if verbose > 1:
            print("Collecting %s..." % (components,))
        if comp == "lorentz":
            assert "Psi" in eqs._field_names and eqs.Psi is not None
            fe_p += prefactor_lorentz*forcing.Le_p
            fs_sym += prefactor_lorentz*forcing.Ls_sym
            fp_sym += prefactor_lorentz*forcing.Lp_sym
            fz_asym += prefactor_lorentz*forcing.Lz_asym
            par_list.append(params.Le)
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
        # Assemble terms
        eqs_new.Psi = Eq(
            eqs.Psi.lhs, 
            prefactor_coriolis*term_coriolis 
            + term_body_force.subs({
                core.fe_p: fe_p, core.fs_sym: fs_sym, 
                core.fp_sym: fp_sym, core.fz_asym: fz_asym}).expand())
    
    return eqs_new, par_list


def apply_bg_to_eq(fname: str, eq: Eq, bg_map: dict, mode: str = "PG",
    verbose: int = 0) -> Eq:
    """Apply background field equation-wise
    
    :param str fname: name of the field / equation
    :param eq: a sympy equation
    :param dict bg_map: background dictionary
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
        new_lhs = eq.lhs.subs(bg_map).subs({H: H_s}).doit().subs({H_s: H}).expand()
        new_rhs = eq.rhs.subs(bg_map).subs({H: H_s}).doit().subs({H_s: H}).expand()
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


def apply_bg_to_set(eqs: base.LabeledCollection, bg: bg_fields.BackgroundFieldMHD, 
    mode: str="PG", verbose: int = 0) -> Tuple[base.LabeledCollection, List]:
    """Apply background field to a set of equations
    
    :param base.LabeledCollection eqs: original set of equations
    :param bg_fields.BackgroundFieldMHD bg: background field 
    :returns: new set of equations, and the unknown params in background field
    """
    if verbose > 0:
        print("========== Applying background field ==========")
    # Building the background field map
    f0_val = pgutils.assemble_background(B0=bg.B0_val, mode=mode)
    bg_sub = {u_comp: bg.U0_val[i_c] for i_c, u_comp in enumerate(core.U0_vec)}
    bg_sub.update({b_comp: bg.B0_val[i_c] for i_c, b_comp in enumerate(core.B0_vec)})
    if mode.lower() == "pg":
        bg_sub.update({comp: f0_val[i_c] for i_c, comp in enumerate(core.pgvar_bg)})
    elif mode.lower() == "cg":
        bg_sub.update({comp: f0_val[i_c] for i_c, comp in enumerate(core.cgvar_bg)})
    eqs_new = eqs.apply(
        lambda fname, eq: apply_bg_to_eq(fname, eq, bg_sub, mode=mode, verbose=verbose-1), 
        inplace=False, metadata=True)
    return eqs_new, bg.params


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
    :param dict map_trial: dictionary to substitute trial functions into the expr
    :param dict map_test: dictionary to substitute test functions into the expr
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


def equations_to_matrices(eqs: base.LabeledCollection, 
    xpd_recipe: xpd.ExpansionRecipe, inplace: bool = False, 
    verbose: int = 0) -> List[xpd.SystemMatrix]:
    """Collect matrix elements
    
    :param base.LabeledCollection eqs: collection of all equations
    :param expansion.ExpansionRecipe xpd_recipe: expansion recipe
    :param bool inplace: whether to modify the eqs inplace
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


INPUT_MODES = {
    "pg": "pg",
    "cg": "cg",
    "reduced": "pg"
}
EQS_FILES = {
    "pg": "./out/symbolic/eqs_pg_lin.json",
    "cg": "./out/symbolic/eqs_cg_lin.json",
}
FORCING_TERMS = {
    "pg": forcing.force_explicit_lin,
    "cg": forcing.force_explicit_lin_cg,
}


def form_equations(
    eq_mode: str = "pg", 
    components: List = ["Lorentz"],
    timescale: str = "Alfven", 
    bg: bg_fields.BackgroundFieldMHD = bg_fields.BackgroundHydro(),
    save_to: Optional[str] = None, 
    verbose: int = 0) -> Tuple[base.LabeledCollection, List]:
    """Top-level function: form set of equations
    
    :param str eq_mode: mode of equation, "pg", "cg" or "reduced"
    :param List components: list of forcing ingredients to be added to the system.
        See arguments for :func:`assemble_forcing` for choices
    :param str timescale: characteristic timescale for nondimensionalization.
        See arguments for :func:`assemble_forcing` for choices
    :param bg_fields.BackgroundFieldMHD bg: a background field,
    :param Optional[str] save_to: output json file name, 
        if None (default), no file will be written.
    :param int verbose: verbosity level, default to 0.
    
    :returns: set of equations and a list of unknown parameters.
    """
    i_mode = INPUT_MODES[eq_mode.lower()]
    
    # Input
    with open(EQS_FILES[i_mode], 'r') as fread:
        eqs = base.LabeledCollection.load_json(fread, parser=parse_expr)
    
    # Assemble forcing
    eqs, par_list_nd = assemble_forcing(eqs, 
        *components, timescale=timescale, verbose=verbose-1)
    eqs.Psi = Eq(eqs.Psi.lhs, eqs.Psi.rhs.subs(FORCING_TERMS[i_mode]))
    eqs, par_list_bg = apply_bg_to_set(eqs, bg, mode=i_mode, verbose=verbose-1)
    par_list = par_list_nd + par_list_bg
    
    # reduce dimensions if applies
    if eq_mode.lower() == "reduced":
        eqs = reduce_eqsys_to_force_form(eqs, verbose=verbose-1)
    
    # Output
    if save_to is not None:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        with open(save_to, 'x') as fwrite:
            json.dump(
                {
                    "params": [srepr(par) for par in par_list],
                    "equations": eqs.serialize(serializer=srepr)
                }, 
                fwrite, indent=4)
        if verbose > 0:
            print("Results saved to {:s}".format(save_to))
    return eqs, par_list


def reduce_dimensions(
    read_from: Union[str, Tuple[base.LabeledCollection, List]], 
    save_to: Optional[str] = None, 
    verbose: int = 0) -> base.LabeledCollection:
    """Top-level function: reduce dimension of the system
    
    :param Union[str, Tuple[base.LabeledCollection, List]] read_from: 
        file name to be loaded as the starting set of equations.
    :param Optional[str] save_to: output json file name, 
        if None (default), no file will be written.
    :param int verbose: verbosity level, default to 0.
    
    :returns: a set of reduced dimensional equations
    
    ..warning:: The current :func:`form_equations` function already
        contains functionality to reduce system into low dimensional
        forms. As a result, the plan is to phase out this function.
    """
    # Input
    if isinstance(read_from, str):
        with open(read_from, 'r') as fread:
            load_array = json.load(fread)
            eqs = base.LabeledCollection.deserialize(
                load_array["equations"], parser=parse_expr)
            par_list = [parse_expr(par) for par in load_array["params"]]
    elif isinstance(read_from, tuple):
        assert len(read_from) == 2 and isinstance(read_from[0], base.LabeledCollection)
        eqs = read_from[0]
        par_list = read_from[1]
    else:
        raise TypeError
    
    # Body
    eqs_new = reduce_eqsys_to_force_form(eqs, verbose=verbose-1)
    
    # Output
    if save_to is not None:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        with open(save_to, 'x') as fwrite:
            json.dump(
                {
                    "params": [srepr(par) for par in par_list],
                    "equations": eqs_new.serialize(serializer=srepr)
                }, 
                fwrite, indent=4)
    return eqs_new, par_list


def collect_matrix_elements(
    read_from: Union[str, Tuple[base.LabeledCollection, List]], 
    manual_select: Optional[List],
    expansion_recipe: xpd.ExpansionRecipe,
    save_to: Optional[str]=None, 
    verbose: int = 0) -> Tuple[xpd.SystemMatrix, xpd.SystemMatrix, List]:
    """Top-level function: collect matrix elements in symbolic forms
    
    :param Union[str, Tuple[base.LabeledCollection, List]] read_from: 
        the starting set of equations.
        * If `read_from` is `str`, then this will be interpreted as a
            json file name to be loaded as equations;
        * if `read_from` is a 2-tuple of `base.LabeledCollection` and 
            List, these will be understood as the set of equations and
            a list of required unknown parameters.
    :param Optional[List] manual_select: if specified, a boolean array
        indicating which equations to be used.
    :param xpd.ExpansionRecipe expansion_recipe: spectral expansion
    :param Optional[str] save_to: output json file name, 
        if None (default), no file will be written.
    :param int verbose: verbosity level, default to 0.
    
    :returns: Mass matrix (expression), stiffness matrix (expression)
        and a list of required unknown parameters
    """
    # Input
    if isinstance(read_from, str):
        with open(read_from, 'r') as fread:
            load_array = json.load(fread)
            eqs = base.LabeledCollection.deserialize(
                load_array["equations"], parser=parse_expr)
            par_list = [parse_expr(par) for par in load_array["params"]]
    elif isinstance(read_from, tuple):
        assert len(read_from) == 2 and isinstance(read_from[0], base.LabeledCollection)
        eqs = read_from[0]
        par_list = read_from[1]
    else:
        raise TypeError
    if manual_select is not None:
        eqs = eqs.generate_collection(manual_select)
    
    # These matrices only make sense at specified azimuthal wavenumber
    par_list.append(m)
    
    # Body
    M_expr, K_expr = equations_to_matrices(eqs, 
        expansion_recipe, inplace=False, verbose=5)
    
    # Output
    if save_to is not None:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        serialized_obj = {
            "xpd": expansion_recipe.identifier,
            "params": [srepr(par) for par in par_list],
            "M": M_expr.serialize(),
            "K": K_expr.serialize()
        }
        with open(save_to, 'x') as fwrite:
            json.dump(serialized_obj, fwrite, indent=4)
        if verbose > 0:
            print("Results saved to {:s}".format(save_to))
    return M_expr, K_expr, par_list


def compute_matrix_numerics(
    read_from: Union[str, Tuple[xpd.SystemMatrix, xpd.SystemMatrix, List]], 
    xpd_recipe: xpd.ExpansionRecipe,
    Ntrunc: int, 
    par_val: dict,
    jacobi_rule_opt: dict = {"automatic": True, "quadN": None},
    quadrature_opt: dict = {"backend": "scipy", "output": "numpy", "outer": True},
    save_to: Optional[str] = None, 
    verbose: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Top-level function: computation of matrix elements
    
    :param Union[str, Tuple[expansion.SystemMatrix, expansion.SystemMatrix, List]] read_from: 
        the symbolic matrices to be computed.
        * If `read_from` is `str`, then this will be interpreted as a
            json file name to be loaded as symbolic matrices;
        * if `read_from` is a 3-tuple of two `expansion.SystemMatrix`s and 
            List, these will be understood as the mass matrix, the stiffness
            matrix and a list of required unknown parameters.
    :param expansion.ExpansionRecipe expansion_recipe: spectral expansion
    :param int Ntrunc: truncation degree (for the vorticity / magnetic field),
    :param dict par_val: the values to be used for unknown parameters,
    :param dict jacobi_rule_opt: options for Gauss-Jacobi quadrature, 
        to be passed to :class:`~pg_utils.numerics.matrices.InnerQuad_GaussJacobi`
    :param dict quadrature_opt: options for forming the inner product matrix, 
        to be passed to :class:`~pg_utils.numerics.matrices.InnerQuad_GaussJacobi.gramian`
    :param Optional[str] save_to: output json file name, 
        if None (default), no file will be written.
    :param int verbose: verbosity level, default to 0.
    
    :returns: the numerical mass and stiffness matrices.
    """
    # Input
    if isinstance(read_from, str):
        with open(read_from, 'r') as fread:
            matrix_obj = json.load(fread)
        M_expr = xpd.SystemMatrix.deserialize(matrix_obj["M"])
        K_expr = xpd.SystemMatrix.deserialize(matrix_obj["K"])
        par_list = [parse_expr(par) for par in matrix_obj["params"]]
    else:
        M_expr = read_from[0]
        K_expr = read_from[1]
        par_list = read_from[2]
    
    # Safeguard: values for all unknown variables should be correctly provided
    assert set(par_list) == set(par_val.keys())
    
    if verbose > 0:
        print("========== Expanding matrices... ==========")
    
    if verbose > 1:
        print("Pre-processing elements...")
    # Pre-processing of elements
    M_expr.apply(lambda ele: ele.subs(par_val), inplace=True, metadata=False)
    K_expr.apply(lambda ele: ele.subs(par_val), inplace=True, metadata=False)
    
    if verbose > 1:
        print("Configuring expansions and quadratures...")    
    # Configure expansions
    fnames = xpd_recipe.rad_xpd.fields._field_names
    cnames = xpd_recipe.rad_xpd.bases._field_names
    ranges_trial = [np.arange(2*Ntrunc + 1) 
        if 'M' in cname else np.arange(Ntrunc + 1) for cname in cnames]
    ranges_test = [np.arange(2*Ntrunc + 1) 
        if 'M' in fname else np.arange(Ntrunc + 1) for fname in fnames]
    
    # Configure quadrature
    quad_recipe_list = np.array([
        [nmatrix.QuadRecipe(
            init_opt=jacobi_rule_opt,
            gram_opt=quadrature_opt
        ) for ele in row] for row in M_expr._matrix
    ])
    
    if verbose > 1:
        print("Computing quadratures of elements...")
    # Computation
    M_val = nmatrix.MatrixExpander(M_expr, 
        quad_recipe_list, ranges_trial, ranges_test).expand(verbose=verbose > 1)
    K_val = nmatrix.MatrixExpander(K_expr, 
        quad_recipe_list, ranges_trial, ranges_test).expand(verbose=verbose > 1)
    
    # Output
    if save_to is not None:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        with h5py.File(save_to, 'x') as fwrite:
            str_type = h5py.string_dtype(encoding="utf-8")
            fwrite.attrs["xpd"] = xpd_recipe.identifier
            for par, val in par_val.items():
                fwrite.attrs[srepr(par)] = float(val)
            gp = fwrite.create_group("rows")
            gp.create_dataset("names", data=fnames, dtype=str_type)
            gp.create_dataset("ranges", 
                data=np.array([len(nrange) for nrange in ranges_test]))
            gp = fwrite.create_group("cols")
            gp.create_dataset("names", data=cnames, dtype=str_type)
            gp.create_dataset("ranges", 
                data=np.array([len(nrange) for nrange in ranges_trial]))
            matrix_gp = fwrite.create_group("M")
            num_io.sparse_coo_save_to_group(coo_array(M_val), matrix_gp)
            matrix_gp = fwrite.create_group("K")
            num_io.sparse_coo_save_to_group(coo_array(K_val), matrix_gp)
        if verbose > 0:
            print("Results saved to {:s}".format(save_to))
    return M_val, K_val


def compute_eigen(
    read_from: Union[str, Tuple[np.ndarray, np.ndarray]], 
    save_to: Optional[str], 
    verbose: int = 0) -> List[np.ndarray]:
    """Top-level routine function
    Stage 4: Computing eigenvalues and eigenvectors from Matrices
    """
    # Input
    if isinstance(read_from, str):
        with h5py.File(read_from, 'r') as fread:
            par_dict = {par: val for par, val in fread.attrs.items()}
            # identifier = fread.attrs["xpd"]
            # m_val = fread.attrs["azm"]
            # Le = fread.attrs["Le"]
            # Lu = fread.attrs["Lu"]
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
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        with h5py.File(save_to, 'x') as fwrite:
            str_type = h5py.string_dtype(encoding="utf-8")
            for par, val in par_dict.items():
                fwrite.attrs[par] = val
            gp = fwrite.create_group("bases")
            gp.create_dataset("names", data=cnames, dtype=str_type)
            gp.create_dataset("ranges", data=np.asarray(crange))            
            fwrite.create_dataset("eigval", data=eig_val)
            fwrite.create_dataset("eigvec", data=eig_vec)
        if verbose > 0:
            print("Results saved to {:s}".format(save_to))
    return eig_val, eig_vec
