#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from collections import OrderedDict
from collections.abc import MutableSet
from pyomo.common.autoslots import AutoSlots


class OrderedSet(AutoSlots.Mixin, MutableSet):
    __slots__ = ('_dict',)

    def __init__(self, iterable=None):
        # Starting in Python 3.7, dict is ordered (and is faster than
        # OrderedDict).  dict began supporting reversed() in 3.8.
        self._dict = {}
        if iterable is not None:
            self.update(iterable)

    def __str__(self):
        """String representation of the mapping."""
        return "OrderedSet(%s)" % (', '.join(repr(x) for x in self))

    def update(self, iterable):
        if isinstance(iterable, OrderedSet):
            self._dict.update(iterable._dict)
        else:
            self._dict.update((val, None) for val in iterable)

    #
    # Implement MutableSet abstract methods
    #

    def __contains__(self, val):
        return val in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def add(self, val):
        """Add an element."""
        self._dict[val] = None

    def discard(self, val):
        """Remove an element. Do not raise an exception if absent."""
        if val in self._dict:
            del self._dict[val]

    #
    # The remaining MutableSet methods have slow default
    # implementations.
    #

    def clear(self):
        """Remove all elements from this set."""
        self._dict.clear()

    def remove(self, val):
        """Remove an element. If not a member, raise a KeyError."""
        del self._dict[val]

    def intersection(self, other):
        other = set(other)
        res = OrderedSet(filter(other.__contains__, self))
        return res

    def union(self, other):
        res = OrderedSet(self)
        res.update(other)
        return res

    #
    # Not strictly part of MutableSet, but it makes sense that OrderedSet
    # should be reversible
    #
    def __reversed__(self):
        return reversed(self._dict)
