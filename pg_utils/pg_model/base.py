# -*- coding: utf-8 -*-
"""
Base classes for use in PG model
Jingtao Min @ ETH-EPM, 09.2023
"""


from typing import Any


class LabeledCollection:
    """Abstract base class for collections to be used in PG model
    
    LabeledCollection is a base class that defines following behaviours
        (1) indexing by integer; in other words, it is sorted;
        (2) indexing by string; in other words, it is labeled;
        (3) the elements can be accessed as attributes.
    In addition, LabeledCollection supports the following operations
        (1) iteration: the object can be traversed as an iterator;
        (2) subcollection: a subset can be extracted from it.
    """
    
    def __init__(self, names, **fields) -> None:
        """Initialization
        
        :param names: list or array-like [str], 
            list of names to be used as field attributes
        :param \**fields: keyword arguments to be initiated as attributes;
            the keys must be part of the `names`, 
            otherwise the key-value pair is ignored;
            fields in `names` not in `fields` will be initiated as None;
            keys in `fields` not in `names` raises an error.
        """
        # Allow attribute assignment
        super().__setattr__("_is_locked", False)
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
                raise TypeError
  
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
        if self._is_locked and not hasattr(self, __name):
            raise AttributeError
        super().__setattr__(__name, __value)
        
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
    
    def _extract_subset(self, sub_slice):
        return LabeledSubCollection(self, sub_slice)



class LabeledSubCollection:
    """Base class that gives a subset of the labeled collection.
    
    LabeledSubCollection, similar to LabeledCollection,
    implements the following operations:
        (1) indexing by integer; in other words, it is sorted;
        (2) indexing by string; in other words, it is labeled.
    In addition, LabeledSubCollection supports the following operations:
        (1) iteration: the object can be traversed as an iterator.
    """
    
    def __init__(self, base_collection: LabeledCollection, sub_slice) -> None:
        """Initialization
        """
        # Base collection
        self.base_collection = base_collection
        tmp_idx = list(range(self.base_collection.n_fields))
        self._sub_names = self.base_collection._field_names[sub_slice]
        self._sub_idx = tmp_idx[sub_slice]
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
    """Base class for the collection of Plesio-Geostrophy (PG) variables, 
    fields, equations, etc.
    """
    
    """Arrangement of variables:
    Vorticity
    Magnetic moments
    Magnetic fields in the equatorial plane
    Magnetic fields at the boundary
    """
    pg_field_names = [
        "Psi", 
        "Mss", "Mpp", "Msp", "Msz", "Mpz", "zMss", "zMpp", "zMsp", 
        "Bs_e", "Bp_e", "Bz_e", "dBs_dz_e", "dBp_dz_e", 
        "Br_b", "Bs_p", "Bp_p", "Bz_p", "Bs_m", "Bp_m", "Bz_m"]
    
    def __init__(self, **fields) -> None:
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
    
    # def __init__(self, 
    #     Psi=None, 
    #     Mss=None, Mpp=None, Msp=None, Msz=None, Mpz=None, zMss=None, zMpp=None, zMsp=None,
    #     Bs_e=None, Bp_e=None, Bz_e=None, dBs_dz_e=None, dBp_dz_e=None,
    #     Br=None, Bs_p=None, Bp_p=None, Bz_p=None, Bs_m=None, Bp_m=None, Bz_m=None) -> None:
    #     """Initialization
    #     """
    #     # Base init
    #     super().__init__()
    #     # Collection variables / fields
    #     self.Psi = Psi
    #     self.Mss, self.Mpp, self.Msp, self.Msz, self.Mpz, self.zMss, self.zMpp, self.zMsp = Mss, Mpp, Msp, Msz, Mpz, zMss, zMpp, zMsp
    #     self.Bs_e, self.Bp_e, self.Bz_e, self.dBs_dz_e, self.dBp_dz_e = Bs_e, Bp_e, Bz_e, dBs_dz_e, dBp_dz_e
    #     self.Br, self.Bs_p, self.Bp_p, self.Bz_p, self.Bs_m, self.Bp_m, self.Bz_m = Br, Bs_p, Bp_p, Bs_m, Bp_m, Bz_m
    #     # Update number of fields
    #     self.n_fields = len(self.list_names)

