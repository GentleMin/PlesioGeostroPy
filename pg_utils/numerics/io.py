# -*- coding: utf-8 -*-
"""Input and output of numerical matrices
"""


import h5py
import numpy as np
from scipy.sparse import coo_array, coo_matrix

from typing import Union, Optional


def matrices_save_h5(fwrite: h5py.Group, f_attrs: dict, 
    rows: Optional[dict] = None, cols: Optional[dict] = None, 
    **kwargs) -> None:
    """Save matrices to h5py file/group"""
    # Save attributes
    for attr_name in f_attrs:
        fwrite.attrs[attr_name] = f_attrs[attr_name]
    # Save rows and columns, if not None
    str_type = h5py.string_dtype(encoding="utf-8")
    if rows is not None:
        gp = fwrite.create_group("rows")
        gp.create_dataset("names", data=rows["names"], dtype=str_type)
        gp.create_dataset("ranges", data=rows["ranges"])
    if cols is not None:
        gp = fwrite.create_group("cols")
        gp.create_dataset("names", data=cols["names"], dtype=str_type)
        gp.create_dataset("ranges", data=cols["ranges"])
    # Save the arrays/matrices
    for name, array in kwargs.items():
        if isinstance(array, np.ndarray):
            fwrite.create_dataset(name, data=array)
        elif isinstance(array, coo_array) or isinstance(array, coo_matrix):
            sp_gp = fwrite.create_group(name)
            sparse_coo_save_to_group(array, sp_gp)


def matrices_load_h5(fread: h5py.Group):
    """Load matrices to h5py file/group
    """
    # Load attributes
    f_attrs = dict()
    for attr_name in fread.attrs.keys():
        f_attrs[attr_name] = fread.attrs[attr_name]
    # Load rows and columns, if they exist
    if "rows" in fread.keys():
        rows = {
            "names": list(fread["rows"]["names"].asstr()[()]),
            "ranges": fread["rows"]["ranges"][()]
        }
    else:
        rows = None
    if "cols" in fread.keys():
        cols = {
            "names": list(fread["cols"]["names"].asstr()[()]),
            "ranges": fread["cols"]["ranges"][()]
        }
    else:
        cols = None
    # Load matrices
    matrices = dict()
    for gp_name in fread.keys():
        if gp_name in ("rows", "cols"):
            continue
        matrices[gp_name] = matrix_load_from_group(fread[gp_name])
    return f_attrs, rows, cols, matrices


def matrix_load_from_group(sp_gp: Union[h5py.Group, h5py.Dataset]
    ) -> Union[np.ndarray, coo_matrix]:
    """Load matrix from h5py group
    """
    # If stored in sparse format: treat as group and construct as sparse
    if "sparse" in sp_gp.attrs.keys():
        if sp_gp.attrs["sparse"] == "coo":
            return sparse_coo_load_from_group(sp_gp)
        else:
            raise AttributeError
    # If not sparse, treat as dataset and read in the dense matrix
    return sp_gp[()]


def sparse_coo_save_to_group(sp_matrix: Union[coo_array, coo_matrix], 
    sp_gp: h5py.Group) -> None:
    """Save sparse matrix in coordinate format to h5py group
    """
    sp_gp.attrs["sparse"] = "coo"
    sp_gp.attrs["nrows"] = sp_matrix.shape[0]
    sp_gp.attrs["ncols"] = sp_matrix.shape[1]
    sp_gp.create_dataset("data", data=sp_matrix.data)
    sp_gp.create_dataset("row", data=sp_matrix.row)
    sp_gp.create_dataset("col", data=sp_matrix.col)
    
    
def sparse_coo_load_from_group(sp_gp: h5py.Group) -> coo_matrix:
    """Load sparse matrix in coordinate format from h5py group
    """
    shape = (sp_gp.attrs["nrows"], sp_gp.attrs["ncols"])
    row = sp_gp["row"][()]
    col = sp_gp["col"][()]
    data = sp_gp["data"][()]
    return coo_array((data, (row, col)), shape=shape)
