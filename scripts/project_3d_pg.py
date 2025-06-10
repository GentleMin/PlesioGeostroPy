# -*- coding: utf-8 -*-
"""Project 3D quantities to PG
"""


import os, sys, h5py
import numpy as np
import sympy as sym
sys.path.append(os.getcwd())

from pg_utils import tools
from pg_utils.pg_model import expand_conjugate_ext, xpd_conj_sym, xpd_conj_antisym
from pg_utils.pg_model import expansion as xpd
from pg_utils.processing import postproc as pproc
from pg_utils.processing import pg_3d
symmetry_xpd = {
    'symmetric': xpd_conj_sym,
    'anti-symmetric': xpd_conj_antisym,
    'none': expand_conjugate_ext
}

from typing import List, Optional


def load_pgevec_template_cstm(fname: str):
    """Load PG eigenvector file as template
    """
    with h5py.File(fname, 'r') as fread:
        xpd_identifier = fread.attrs['xpd']
        m_key = 'azm' if 'azm' in fread.attrs.keys() else sym.srepr(xpd.m)
        m_val = int(fread.attrs[m_key])
        cnames = list(fread['bases']['names'].asstr()[()])
        cranges = fread['bases']['ranges'][()]
    
    return m_val, cnames, cranges, xpd_identifier


def load_3devec_cstm(fname: str, k: int = 0, i_res: Optional[int] = None):
    """Load 3D eigenvector file
    """
    if i_res is None:
        eig_vals_3d = list()
        eig_vecs_3d = list()
        
        with h5py.File(fname, 'r') as fread:
            for i_gps in range(len(fread.keys())):
                eig_vals_3d.append(fread[f'eigenmode_target{i_gps}']['eigenvals'][k])
                eig_vecs_3d.append(fread[f'eigenmode_target{i_gps}']['eigenvecs'][:, k])
        
        eig_vals_3d = np.array(eig_vals_3d)
        eig_vecs_3d = np.stack(eig_vecs_3d, axis=1)
        return eig_vals_3d, eig_vecs_3d
    else:
        with h5py.File(fname, 'r') as fread:
            gp_res = fread[f'Resolution_{i_res}']
            eig_vals_3d = gp_res['eigenvals'][()]
            eig_vecs_3d = gp_res['eigenvecs'][()]
            L, N, m = gp_res.attrs['L'], gp_res.attrs['N'], gp_res.attrs['m']
        return eig_vals_3d, eig_vecs_3d, L, N, m


def project_3d_pg_cstm(
    evecs_3d: np.ndarray, bg_3d: List, 
    spec_ctx: pg_3d.ResolutionContext3D_cstM, 
    spec_ctx_bg: pg_3d.ResolutionContext3D_cstM, 
    spec_recipe: xpd.ExpansionRecipe, 
    cnames: List, cranges: List, timer: tools.ProcTimer, normalize: bool = False, **kwargs
):
    """Project list of eigenvectors to PG eigenvectors
    """
    B_bg = sum([pg_3d.VectorFieldSingleM.from_SH_mode(spec_ctx_bg.N, spec_ctx_bg.L, mode_SH) for mode_SH in bg_3d])
    col_ranges = dict(zip(cnames, cranges))
    evec_can_list = list()
    
    for i_eig in range(evecs_3d.shape[1]):
        tools.print_heading(f"Projecting {i_eig+1}/{evecs_3d.shape[1]} eigenvalues...", prefix='\n', lines='over', char='-')
        usp, bsp = pg_3d.eigmode_3d_from_eigvec(evecs_3d[:, i_eig], spec_ctx, normalize='none')
        evec_can = pg_3d.project_3d_to_conj(usp, bsp, B_bg, spec_ctx, spec_recipe, col_ranges, timer=timer, **kwargs)
        evec_can = evec_can._array
        if normalize:
            evec_can /= np.linalg.norm(evec_can)
        evec_can_list.append(evec_can)
        timer.flag(loginfo=f"{i_eig+1}/{evecs_3d.shape[1]} eigenvalues processed.", print_str=True, mode='0+')
    
    evec_can_list = np.stack(evec_can_list, axis=1)
    return evec_can_list


def save_eig_vecs(fname: str, eig_vals: np.ndarray, eig_vecs: np.ndarray, cnames: List, cranges: List, par_dict: dict):
    """Save eigenvalues and eigenvectors
    """
    with h5py.File(fname, 'x') as fwrite:
        str_type = h5py.string_dtype(encoding="utf-8")
        for par, val in par_dict.items():
            fwrite.attrs[par] = val
        gp = fwrite.create_group("bases")
        gp.create_dataset("names", data=cnames, dtype=str_type)
        gp.create_dataset("ranges", data=np.asarray(cranges))
        fwrite.create_dataset("eigval", data=eig_vals.astype(np.complex128))
        fwrite.create_dataset("eigvec", data=eig_vecs.astype(np.complex128))


dir_3d_model = os.path.join('../Modes_3D/MCmodes/runs', 'QGP-SL2N2_Le1e-4_Lu2e+4_m3')
dir_pg_model = os.path.join('./out/eigen', 'S_L2_N1', 'Canonical')
fname_3d_eigenval = os.path.join(dir_3d_model, 'eigenmodes_traced.h5')
fname_pg_template = os.path.join(dir_pg_model, 'eigen_ideal_m3_Le1e-4_N120_p113.h5')
fname_proj_output = os.path.join(dir_pg_model, 'eigen_f3dsym_Lu2e+4_Le1e-4_3DL75N66-PGN120.h5')
i_res = 1

# SL2N1
B_bg = [pg_3d.SphericalHarmonicMode("pol", 2, 0, "1/4 Sqrt[3/26] r^2(5r^2 - 7)")]
bg_symmetry = 'Symmetric'
xpd_cfg = symmetry_xpd[bg_symmetry.lower()]


if __name__ == '__main__':
    
    tools.print_heading(f"3-D to PG Projection", prefix='\n', suffix='\n', lines='both', char='=')
    timer = tools.ProcTimer(start=True)
    
    if i_res is None:
        eig_vals_3d, eig_vecs_3d = load_3devec_cstm(fname_3d_eigenval)
        spec_ctx = pg_3d.ResolutionContext3D_cstM(N=63, L=63, m_val=3)
        spec_ctx_bg = pg_3d.ResolutionContext3D_cstM(N=10, L=10, m_val=0)
    else:
        eig_vals_3d, eig_vecs_3d, Lmax, Nmax, m = load_3devec_cstm(fname_3d_eigenval, i_res=i_res)
        spec_ctx = pg_3d.ResolutionContext3D_cstM(N=Nmax, L=Lmax, m_val=m)
        spec_ctx_bg = pg_3d.ResolutionContext3D_cstM(N=10, L=10, m_val=0)
    
    timer.flag(loginfo="Eigenvector for 3D calculation loaded", print_str=True, mode='0+')
    
    m_val, cnames, cranges, xpd_identifier = load_pgevec_template_cstm(fname_pg_template)
    timer.flag(loginfo="Eigenvector template loaded.", print_str=True, mode='0+')
    
    tools.print_heading(f"Projecting 3-D solutions to PG under {xpd_cfg.identifier} spectral expansion.", prefix='\n', lines='over', char='=')
    eig_vecs_proj = project_3d_pg_cstm(
        eig_vecs_3d, B_bg, spec_ctx, spec_ctx_bg, xpd_cfg.recipe, cnames, cranges, 
        timer, normalize=True, psi_estimator='lsq_zmean', verbose=True
    )
    
    par_dict = {'spectral-expansion': xpd_cfg.identifier, sym.srepr(xpd.m): spec_ctx.m_val}
    save_eig_vecs(fname_proj_output, eig_vals_3d, eig_vecs_proj, cnames, cranges, par_dict)
    timer.flag(loginfo=f"Results saved to {fname_proj_output}", print_str=True, mode='0+')
    