# -*- coding: utf-8 -*-
"""Input and output of numerical matrices
"""


import h5py, json
import numpy as np
from scipy.sparse import coo_array, coo_matrix

from typing import Union, Tuple, Optional, Callable, Literal, List, Any
from . import utils


def matrices_save_h5(fwrite: h5py.Group, matrix: Union[np.ndarray, coo_array, coo_matrix], 
    f_attrs: dict = {}, rows: Optional[dict] = None, cols: Optional[dict] = None) -> None:
    """Save matrices to h5py file/group. This is the main interface
    for saving matrices as h5py object.
    
    :param h5py.Group fwrite: an h5py group object for writing
    :param dict f_attrs: attributes of the group
    :param Optional[dict] rows: blocks of rows.
        Default is None, then no row info will be stored.
        If not None, the dict will be interpreted as a dict with
        keys "names" and "ranges", which specifies the names of 
        the row blocks and ranges of the row blocks.
    :param Optional[dict] cols: blocks of cols, see rows.
    :param \**kwargs: key-value pairs of matrix name - matrix val
    """
    # Save matrix
    if isinstance(matrix, np.ndarray):
        fwrite.create_dataset("Matrix", data=matrix)
    elif isinstance(matrix, coo_array) or isinstance(matrix, coo_matrix):
        sp_gp = fwrite.create_group("Matrix")
        sparse_coo_save_to_group(matrix, sp_gp)
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


def matrices_load_h5(fread: h5py.Group
    ) -> Tuple[Union[np.ndarray, coo_array], dict, Optional[dict], Optional[dict]]:
    """Load matrices to h5py file/group
    """
    # Load matrix
    matrix = matrix_load_from_group(fread["Matrix"])
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
    # Load matrix
    return matrix, f_attrs, rows, cols


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
    
    
def sparse_coo_load_from_group(sp_gp: h5py.Group) -> coo_array:
    """Load sparse matrix in coordinate format from h5py group
    """
    shape = (sp_gp.attrs["nrows"], sp_gp.attrs["ncols"])
    row = sp_gp["row"][()]
    col = sp_gp["col"][()]
    data = sp_gp["data"][()]
    return coo_array((data, (row, col)), shape=shape)


def serialize_coo(sp_array: Union[coo_array, coo_matrix],
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None, 
    format: Literal["json", "pickle"] = "pickle") -> dict:
    """Retrieve serializable objs of COOrdinate format sparse array
    
    :param scipy.sparse.coo_array sp_array: sparse array to serialize
    :param Optional[Callable[[np.ndarray], np.ndarray]] transform: transform
        the sparse data elements. Default is None, no transformation
    :param Literal["json", "pickle"] format: serialization format;
        if pickle, the objects will be left as it is;
        if json, the numpy arrays will be unparsed into lists, 
        and data are preferrably converted to string using transform
    """
    coo_dict = {
        "row": sp_array.row, 
        "col": sp_array.col,
        "data": sp_array.data,
        "shape": sp_array.shape
    }
    # Convert data if necessary
    if transform is not None:
        coo_dict["data"] = transform(coo_dict["data"])
    # For serializing to json: unparse all numpy arrays to list
    if format == "json":
        coo_dict["row"] = coo_dict["row"].tolist()
        coo_dict["col"] = coo_dict["col"].tolist()
        coo_dict["data"] = list(map(str, coo_dict["data"]))
    
    return coo_dict


def parse_coo(serialized_obj: dict, 
    transform: Optional[Callable[[Any], Any]] = None) -> coo_array:
    """
    """
    assert ("row" in serialized_obj and "col" in serialized_obj 
        and "data" in serialized_obj and "shape" in serialized_obj)
    # Convert data if necessary
    data = serialized_obj["data"]
    if transform is not None:
        data = transform(data)
    return coo_array(
        (data, (np.asarray(serialized_obj["row"]), np.asarray(serialized_obj["col"]))), 
        shape=serialized_obj["shape"]
    )


class CompactArrayJSONEncoder(json.JSONEncoder):
    """A Json encoder that puts long arrays on one line
    
    This encoder is adopted from the encoder written by
    `Jannismain <https://gist.github.com/jannismain/e96666ca4f059c3e5bc28abb711b5c92>`_
    for compressing small containers on single lines.
    """
    
    CONTAINER_TYPES = (list, tuple)
    SINGLELINE_THRESHOLD = 10
    
    def __init__(self, *args, **kwargs):
        if kwargs.get("indent") is None:
            kwargs["indent"] = 4
        super().__init__(*args, **kwargs)
        self.indentation_level = 0
    
    def encode(self, o) -> str:
        """Encode JSON"""
        if isinstance(o, self.CONTAINER_TYPES):
            return self._encode_list(o)
        elif isinstance(o, dict):
            return self._encode_object(o)
        return json.dumps(
            o,
            skipkeys=self.skipkeys,
            ensure_ascii=self.ensure_ascii,
            check_circular=self.check_circular,
            allow_nan=self.allow_nan,
            sort_keys=self.sort_keys,
            indent=self.indent,
            separators=(self.item_separator, self.key_separator),
            default=self.default if hasattr(self, "default") else None,
        )
        
    def _encode_list(self, o) -> str:
        if self._if_single_line(o):
            return '[' + ','.join(self.encode(el) for el in o) + ']'
        self.indentation_level += 1
        output = [self.indent_str + self.encode(el) for el in o]
        self.indentation_level -= 1
        return '[\n' + ',\n'.join(output) + '\n' + self.indent_str + ']'
    
    def _encode_object(self, o) -> str:
        if not o:
            return "{}"
        self.indentation_level += 1
        output = [
            f"{self.indent_str}{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()
        ]
        self.indentation_level -= 1
        return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"
    
    def _if_single_line(self, o):
        return self._primitives_only(o) and len(o) >= self.SINGLELINE_THRESHOLD
    
    def _primitives_only(self, o: List):
        return not any(isinstance(el, self.CONTAINER_TYPES) for el in o)
    
    @property
    def indent_str(self) -> str:
        if isinstance(self.indent, int):
            return " "*(self.indentation_level*self.indent)
        elif isinstance(self.indent, str):
            return self.indentation_level*self.indent
        else:
            raise ValueError(
                f"indent must either be of type int or str (is: {type(self.indent)})"
            )
