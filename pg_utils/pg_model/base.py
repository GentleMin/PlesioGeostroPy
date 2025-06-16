# -*- coding: utf-8 -*-
"""
Base classes and collections
"""


from typing import Any, Callable, List, TextIO
import json


class LabeledCollection:
    """Abstract base class for collections to be used in PG model
    
    LabeledCollection is a base class that defines following behaviours
    
    * indexing by integer; in other words, it is sorted;
    * indexing by string; in other words, it is labeled;
    * the elements can be accessed as attributes.
    
    In addition, LabeledCollection supports the following operations
    
    * iterator: the object can be traversed as an iterator;
    * subcollection: a subset can be extracted from it.
    
    """
    
    def __init__(self, names: List[str], **fields) -> None:
        """Initialization
        
        :param List[str] names: names of the fields, 
            list of names to be used as field attributes
        :param \**fields: keyword arguments to be initiated as attributes;
            the keys must be part of the `names`, 
            otherwise the key-value pair is ignored;
            fields in `names` not in `fields` will be initiated as None;
            keys in `fields` not in `names` raises an error.
        """
        # Allow attribute assignment
        self._enable_attribute_addition()
        # Generate fields and values
        self._field_names = names
        self._validate_field_entries(**fields)
        for field_key in names:
            if field_key in fields:
                setattr(self, field_key, fields[field_key])
            else:
                setattr(self, field_key, None)
        self.n_fields = len(self._field_names)
        # Iterator mode
        self._iter_name = False
        self._iter_filter = False
        self.n_iter = 0
    
    def _validate_field_entries(self, **fields) -> None:
        """Validate field keyword arguments
        manually raise a TypeError if keyword in `fields` not in `names`
        """
        for field_key in fields:
            if field_key not in self._field_names:
                raise KeyError('%s not a valid field name!' % (field_key,))
  
    @property
    def iter_name(self):
        """iter_name: bool
        determines whether the string of the field is also returned.
        """
        return self._iter_name
    
    @iter_name.setter
    def iter_name(self, option):
        if not isinstance(option, bool):
            raise TypeError
        self._iter_name = option
    
    @property
    def iter_filter(self):
        """iter_filter: bool
        determines whether None fields are skipped.
        """
        return self._iter_filter
    
    @iter_filter.setter
    def iter_filter(self, option):
        if not isinstance(option, bool):
            raise TypeError
        self._iter_filter = option

    def __setattr__(self, __name: str, __value: Any) -> None:
        """Attribute setter
        Overridden to prevent attribute addition when _is_locked is True
        """
        if hasattr(self, "_is_locked"):
            if self._is_locked and not hasattr(self, __name):
                raise AttributeError
        super().__setattr__(__name, __value)
    
    def _enable_attribute_addition(self) -> None:
        super().__setattr__("_is_locked", False)
    
    def _disable_attribute_addition(self) -> None:
        self._is_locked = True
    
    def __getitem__(self, __key):
        """Indexed access
        
        :param __key: int, slice or str. Index/key of the item to be accessed
        :returns: the indexed or sliced items
        """
        if isinstance(__key, int):
            return self._getitem_by_idx(__key)
        elif isinstance(__key, slice):
            return self._getitem_by_slice(__key)
        elif isinstance(__key, str):
            return self._getitem_by_name(__key)
        else:
            raise TypeError
    
    def _getitem_by_idx(self, __idx):
        """Accessing item based on integer index.
        """
        name = self._field_names[__idx]
        return self._getitem_by_name(name)
    
    def _getitem_by_slice(self, __slice):
        """Accessing items based on slice.
        """
        name_list = self._field_names[__slice]
        item_list = [self._getitem_by_name(name) for name in name_list]
        return item_list
    
    def _getitem_by_name(self, __name):
        """Accessing item based on string.
        """
        if __name in self._field_names:
            return getattr(self, __name)
        else:
            raise KeyError
    
    def __setitem__(self, __key, __val):
        """Indexed assignment
        
        :param __key: int or str, index/key of the field to be assigned
        :param __val: Any, value to be assigned to the field
        """
        if isinstance(__key, int):
            self._setitem_by_idx(__key, __val)
        elif isinstance(__key, str):
            self._setitem_by_name(__key, __val)
        else:
            return TypeError
    
    def _setitem_by_idx(self, __idx, __val):
        """Assigning item based on index
        """
        name = self._field_names[__idx]
        self._setitem_by_name(name, __val)
    
    def _setitem_by_name(self, __name, __val):
        """Assigning item based on string
        """
        if __name in self._field_names:
            setattr(self, __name, __val)
        else:
            raise KeyError
    
    def __iter__(self):
        """Iterator
        """
        self.n_iter = 0
        return self
    
    def __next__(self):
        """Used together with iterator
        """
        if self.n_iter < self.n_fields:
            item = self._getitem_by_idx(self.n_iter)
            self.n_iter += 1
            if self._iter_filter and item is None:
                return self.__next__()
            if self._iter_name:
                item = (self._field_names[self.n_iter], item)
            return item
        else:
            raise StopIteration
    
    def __len__(self):
        return self.n_fields
    
    def _extract_subset(self, sub_slice):
        return LabeledSubCollection(self, sub_slice)
        
    def copy(self) -> "LabeledCollection":
        """Returns a deep copy of itself
        """
        return LabeledCollection(self._field_names, 
            **{fname: self[fname] for fname in self._field_names})
    
    def as_empty(self) -> "LabeledCollection":
        """Return an object with the same configuration of attributes
        but with fields initiated as None. May be slightly faster than copy?
        """
        return LabeledCollection(self._field_names)
    
    def apply(self, fun: Callable[..., Any], 
        inplace: bool = False, metadata: bool = False) -> "LabeledCollection":
        """Apply a function iteratively to all collection items
        
        :param Callable fun: determines how the collection entries
            are processed. The signature of the function should take
            the form ``fun(type(self[i]))`` when metadata is False, and
            the form ``fun(str, type(self[i]))`` when metadata is True
        :param bool inplace: whether to write changes in situ.
        :param bool metadata: whether to pass field name to the function
        
        :returns: Updated object. If inplace, then the 
            current object itself is returned.
        :rtype: LabeledCollection
        """
        if inplace:
            apply_to = self
        else:
            apply_to = self.as_empty()
        for i_field in range(self.n_fields):
            if metadata:
                apply_to[i_field] = fun(self._field_names[i_field], self[i_field])
            else:
                apply_to[i_field] = fun(self[i_field])
        return apply_to
    
    def subs(self, sub_map: dict, inplace: bool = False):
        """Substitute variables iteratively in all fields. This utility
        is for collections with `sympy.Expr` elements. See warning below.
        
        :param dict sub_map: a substitution map
        :param bool inplace: whether the change is made in place; default is False
        
        .. warning::
        
            To use this method, all elements in the collection need to have
            implemented the ``subs`` method. This is intended for a collection
            of which all elements are :class:`sympy.Expr` type. Then, this
            method calls :meth:`apply` internally to substitute all the variables.
        """
        return self.apply(lambda eq: eq.subs(sub_map), inplace=inplace)
    
    def generate_collection(self, index_array: List[bool]) -> "LabeledCollection":
        """Generate a new collection based on indices
        
        :param List[bool] index_array: array of booleans indicating whether
            a field is to be included in the new collection
        :returns: new collection with elements specified by `index_array`
        """
        assert len(index_array) == self.n_fields
        new_names = [fname for idx, fname in enumerate(self._field_names) 
                     if index_array[idx]]
        return LabeledCollection(new_names, **{fname: self[fname]
            for idx, fname in enumerate(self._field_names) if index_array[idx]})

    def serialize(self, serializer: Callable[[Any], str]=str) -> List[tuple]:
        """Serialize the object
        
        :param Callable[[Any],str] serializer: a callable that maps an element
            to a string. Default is the :func:`str` method.
        :returns: a list of serialized objects, in the format
            [(`fname_1`, `serialized_element_1`),
            (`fname_2`, `serialized_element_2`), ...]
        """
        return [(fname, serializer(self[fname])) for fname in self._field_names]
    
    def save_json(self, fp: TextIO, serializer: Callable[[Any], str] = str) -> None:
        """Serialize the object in string format and save to json file
        
        :param TextIO fp: file handle of the output file
        :param Callable[[Any], str] serializer: a callable that maps an element
            to a string. Default is the :func:`str` method.
        """
        save_array = self.serialize(serializer=serializer)
        json.dump(save_array, fp, indent=4)
    
    @staticmethod
    def deserialize(obj: List[tuple], 
        parser: Callable[[str], Any]=lambda x: x) -> "LabeledCollection":
        """Deserialize an object
        
        :param List[tuple] obj: a serialized object of `LabeledCollection`
        :param Callable[[str], Any] parser: a parser that defines how each
            string can be parsed into meaningful objects
        
        :returns: `LabeledCollection`, that is deserialized from the input object
        
        .. warning::

            Sanitized input needed. Unexpected behaviour if input is not a
            legitimate serialized object
        """
        field_names = [field[0] for field in obj]
        field_dict = {field[0]: parser(field[1]) for field in obj}
        return LabeledCollection(field_names, **field_dict)
    
    @staticmethod
    def load_json(fp: TextIO, 
        parser: Callable[[str], Any] = lambda x: x) -> "LabeledCollection":
        """Load LabeledCollection object from json
        
        convenient wrapper for :meth:`deserialize`
        """
        load_array = json.load(fp)
        return LabeledCollection.deserialize(load_array, parser=parser)


class LabeledSubCollection:
    """Base class that gives a subset of the labeled collection.
    
    LabeledSubCollection, similar to LabeledCollection,
    implements the following operations:
    
    * indexing by integer; in other words, it is sorted;
    * indexing by string; in other words, it is labeled;
    
    In addition, LabeledSubCollection supports the following operations:
    
    * iteration: the object can be traversed as an iterator.
    
    Since the day I built this class, I have almost never found this
    class useful. Perhaps we can remove it in a later version?
    """
    
    def __init__(self, base_collection: LabeledCollection, sub_slice) -> None:
        """Initialization
        """
        # Base collection
        self.base_collection = base_collection
        tmp_idx = list(range(self.base_collection.n_fields))
        self._sub_names = self.base_collection._field_names[sub_slice]
        self._sub_idx = tmp_idx[sub_slice]
        self.n_fields = len(self._sub_names)
        # Iteration modes
        self._iter_name = False
        self._iter_filter = False

    @property
    def iter_name(self):
        """iter_name: bool
        determines whether the string of the field is also returned.
        """
        return self._iter_name
    
    @iter_name.setter
    def iter_name(self, option):
        if not isinstance(option, bool):
            raise TypeError
        self._iter_name = option
    
    @property
    def iter_filter(self):
        """iter_filter: bool
        determines whether None fields are skipped.
        """
        return self._iter_filter
    
    @iter_filter.setter
    def iter_filter(self, option):
        if not isinstance(option, bool):
            raise TypeError
        self._iter_filter = option

    def __getitem__(self, __key):
        """Indexing
        
        :param items: int or str. Index of the item to be accessed;
            unlike LabeledCollection, slice is no longer allowed.
        :returns: the indexed or sliced items
        """
        if isinstance(__key, int):
            return self._getitem_by_idx(__key)
        elif isinstance(__key, slice):
            return self._getitem_by_slice(__key)
        elif isinstance(__key, str):
            return self._getitem_by_name(__key)
        else:
            raise TypeError
    
    def _getitem_by_idx(self, __idx):
        """Accessing item based on integer index.
        """
        name = self._sub_names[__idx]
        return self.base_collection._getitem_by_name(name)
    
    def _getitem_by_slice(self, __slice):
        """Accessing items based on slice.
        """
        name_list = self._sub_names[__slice]
        item_list = [self.base_collection._getitem_by_name(name) 
            for name in name_list]
        return item_list
    
    def _getitem_by_name(self, name):
        """Accessing item based on string.
        """
        if name in self._sub_names:
            return self.base_collection._getitem_by_name(name)
        else:
            raise KeyError
        
    def __setitem__(self, __key, __val):
        """Indexed assignment
        
        :param __key: int or str, index/key of the field to be assigned
        :param __val: Any, value to be assigned to the field
        """
        if isinstance(__key, int):
            self._setitem_by_idx(__key, __val)
        elif isinstance(__key, str):
            self._setitem_by_name(__key, __val)
        else:
            return TypeError
    
    def _setitem_by_idx(self, __idx, __val):
        """Assigning item based on index
        """
        name = self._sub_names[__idx]
        self.base_collection._setitem_by_name(name, __val)
    
    def _setitem_by_name(self, __name, __val):
        """Assigning item based on string
        """
        if __name in self._sub_names:
            self.base_collection._setitem_by_name(__name, __val)
        else:
            raise KeyError
    
    def __iter__(self):
        """Iterator
        """
        self.n_iter = 0
        return self
    
    def __next__(self):
        """Used together with iterator
        """
        if self.n_iter < self.n_fields:
            item = self._getitem_by_idx(self.n_iter)
            self.n_iter += 1
            if self._iter_filter and item is None:
                return self.__next__()
            if self._iter_name:
                item = (self._field_names[self.n_iter], item)
            return item
        else:
            raise StopIteration



class CollectionPG(LabeledCollection):
    """Base class for the collection of Plesio-Geostrophy (PG) variables
    
    The field names are pre-defined, and arranged in the following way
    
    * Stream function (`Psi`)
    * Quadratic moments of magnetic field (`Mss`, `Mpp`, `Msp`, `Msz`, `Mpz`, `zMss`, `zMpp`, `zMsp`)
    * Magnetic fields in the equatorial plane (`Bs_e`, `Bp_e`, `Bz_e`, `dBs_dz_e`, `dBp_dz_e`)
    * Magnetic fields at the boundary (`Br_b`, `Bs_p`, `Bp_p`, `Bz_p`, `Bs_m`, `Bp_m`, `Bz_m`)
    
    """
    pg_field_names = [
        "Psi", 
        # "Mss", "Mpp", "Msp", "Msz", "Mpz", "zMss", "zMpp", "zMsp", 
        # "Bs_e", "Bp_e", "Bz_e", "dBs_dz_e", "dBp_dz_e", 
        "Mss", "Mpp", "Msp", "Mzz", "zMsz", "zMpz", "z2Mss", "z2Mpp", "z2Msp", 
        "Br_b", "Bs_p", "Bp_p", "Bz_p", "Bs_m", "Bp_m", "Bz_m"]
    
    def __init__(self, **fields) -> None:
        """Constructor"""
        super().__init__(self.pg_field_names, **fields)
        # No longer accepts attribution addition
        self._disable_attribute_addition()
    
    def vorticity(self):
        """Extract vorticity equation.
        Basically an alias as Psi
        """
        return self.Psi
    
    def subset_mag(self):
        """Extract subset of magnetic quantities.
        """
        return self._extract_subset(slice(1, None))
    
    def subset_moments(self):
        """Extract subset of magnetic moments.
        """
        return self._extract_subset(slice(1, 9))
    
    def subset_B_equator(self):
        """Extract subset of B field on equatorial plane.
        """
        return self._extract_subset(slice(9, 14))
    
    def subset_B_bound(self):
        """Extract subset of B field at the boundary.
        """
        return self._extract_subset(slice(14, None))
    
    def subset_B_bound_cyl(self):
        """Extract subset of B field at the boundary, cylindrical coordinates.
        """
        return self._extract_subset(slice(15, None))
    
    def copy(self) -> "CollectionPG":
        """Deep copy, overriding the :meth:`LabeledCollection.copy` method
        """
        return CollectionPG(**{fname: self[fname] for fname in self._field_names})
    
    def as_empty(self) -> "CollectionPG":
        """Overriding the :meth:`LabeledCollection.as_empty` method
        """
        return CollectionPG()
    
    def apply(self, fun: Callable[..., Any], inplace: bool = False,
        metadata: bool = False) -> "CollectionPG":
        """Overriding the :meth:`LabeledCollection.copy` method"""
        return super().apply(fun, inplace, metadata)
    
    @staticmethod
    def deserialize(obj: List[tuple], 
        parser: Callable[[str], Any] = lambda x: x) -> "CollectionPG":
        """Deserialize an object
        
        overriding the :meth:`LabeledCollection.deserialize` method
        """
        field_names = [field[0] for field in obj]
        assert field_names == CollectionPG.pg_field_names
        field_dict = {field[0]: parser(field[1]) for field in obj}
        return CollectionPG(**field_dict)
    
    @staticmethod
    def load_json(fp: TextIO, 
        parser: Callable[[str], Any] = lambda x: x) -> "CollectionPG":
        """Load CollectionPG object from json
        
        overriding the :meth:`LabeledCollection.load_json` method
        """
        load_array = json.load(fp)
        return CollectionPG.deserialize(load_array, parser=parser)



class CollectionConjugate(LabeledCollection):
    """Base class for the collection of conjugate variables
     
    These correspond to the conjugate counterpart
    of PG variables, fields and equations.
    
    The field names are pre-defined, and arranged in the following way
    
    * Stream function (`Psi`)
    * Conjugate variables of the quadratic moments of magnetic field
        (`M_1`, `M_p`, `M_m`, `M_zp`, `M_zm`, `zM_1`, `zM_p`, `zM_m`)
    * Conjugate variables of the magnetic fields in the equatorial plane
        (`B_ep`, `B_em`, `Bz_e`, `dB_dz_ep`, `dB_dz_em`)
    * Magnetic fields at the boundary 
        (`Br_b`, `B_pp`, `B_pm`, `Bz_p`, `B_mp`, `B_mm`, `Bz_m`)

    """
    cg_field_names = [
        "Psi", 
        # "M_1", "M_p", "M_m", "M_zp", "M_zm", "zM_1", "zM_p", "zM_m", 
        # "B_ep", "B_em", "Bz_e", "dB_dz_ep", "dB_dz_em", 
        "M_1", "M_p", "M_m", "Mzz", "zM_zp", "zM_zm", "z2M_1", "z2M_p", "z2M_m", 
        "Br_b", "B_pp", "B_pm", "Bz_p", "B_mp", "B_mm", "Bz_m"]
    
    def __init__(self, **fields) -> None:
        """Constructor
        
        No longer accepts attribution addition
        """
        super().__init__(self.cg_field_names, **fields)
        # No longer accepts attribution addition
        self._disable_attribute_addition()
    
    def vorticity(self):
        """Extract vorticity equation.
        Basically an alias as Psi
        """
        return self.Psi
    
    def subset_mag(self):
        """Extract subset of magnetic quantities.
        """
        return self._extract_subset(slice(1, None))
    
    def subset_moments(self):
        """Extract subset of magnetic moments.
        """
        return self._extract_subset(slice(1, 9))
    
    def subset_B_equator(self):
        """Extract subset of B field on equatorial plane.
        """
        return self._extract_subset(slice(9, 14))
    
    def subset_B_bound(self):
        """Extract subset of B field at the boundary.
        """
        return self._extract_subset(slice(14, None))
    
    def subset_B_bound_cyl(self):
        """Extract subset of B field at the boundary, cylindrical coordinates.
        """
        return self._extract_subset(slice(15, None))
    
    def copy(self) -> "CollectionConjugate":
        """Deep copy
        
        overriding the :meth:`LabeledCollection.copy` method
        """
        return CollectionConjugate(
            **{fname: self[fname] for fname in self._field_names})
    
    def as_empty(self) -> "CollectionConjugate":
        """overriding the :meth:`LabeledCollection.as_empty` method
        """
        return CollectionConjugate()
    
    def apply(self, fun: Callable[..., Any], inplace: bool = False,
        metadata: bool = False) -> "CollectionConjugate":
        """Overriding the :meth:`LabeledCollection.apply` method
        """
        return super().apply(fun, inplace, metadata)

    @staticmethod
    def deserialize(obj: List[tuple], 
        parser: Callable[[str], Any]=lambda x: x) -> "CollectionConjugate":
        """Deserialize object
        
        overriding the :meth:`LabeledCollection.deserialize` method
        """
        field_names = [field[0] for field in obj]
        assert field_names == CollectionConjugate.cg_field_names
        field_dict = {field[0]: parser(field[1]) for field in obj}
        return CollectionConjugate(**field_dict)
    
    @staticmethod
    def load_json(fp: TextIO, 
        parser: Callable[[str], Any] = lambda x: x) -> "CollectionConjugate":
        """Load CollectionConjugate object from json
        
        overriding the :meth:`LabeledCollection.load_json` method
        """
        load_array = json.load(fp)
        return CollectionConjugate.deserialize(load_array, parser=parser)


def map_collection(maps_from: LabeledCollection, maps_to: LabeledCollection) -> dict:
    """Construct mapping from one Collection object to another Collection
    
    :param LabeledCollection maps_from: Collection of fields to be mapped from
    :param LabeledCollection maps_to: Collection of fields to be mapped to
    :returns: a dictionary of the map
    """
    assert maps_from._field_names == maps_to._field_names
    return {maps_from[fname]: maps_to[fname] for fname in maps_to._field_names}


def map_PG_fields(maps_from: CollectionPG, maps_to: CollectionPG) -> dict:
    """Create mapping from one CollectionPG object to another CollectionPG
    
    :param maps_from: CollectionPG of fields to be mapped from
    :param maps_to: CollectionPG of fields to be mapped to
    :returns: a dictionary
    """
    pg_map = {maps_from[i_field]: maps_to[i_field] 
        for i_field in range(maps_from.n_fields)}
    return pg_map
