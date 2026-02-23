import functools
import jittor as jt
import copy
from collections import OrderedDict


class TensorDict(OrderedDict):
    """Container mainly used for dicts of jittor tensors."""

    def concat(self, other):
        return TensorDict(self, **other)

    def copy(self):
        return TensorDict(super(TensorDict, self).copy())

    def __deepcopy__(self, memodict={}):
        return TensorDict(copy.deepcopy(list(self), memodict))

    def __getattr__(self, name):
        if not hasattr(jt.Var, name):
            raise AttributeError('\'TensorDict\' object has not attribute \'{}\''.format(name))

        def apply_attr(*args, **kwargs):
            return TensorDict({n: getattr(e, name)(*args, **kwargs) if hasattr(e, name) else e for n, e in self.items()})
        return apply_attr

    def attribute(self, attr: str, *args):
        return TensorDict({n: getattr(e, attr, *args) for n, e in self.items()})

    def apply(self, fn, *args, **kwargs):
        return TensorDict({n: fn(e, *args, **kwargs) for n, e in self.items()})

    def to(self, device):
        """Jittor auto manages device, this is a no-op for compatibility."""
        return self

    @staticmethod
    def _iterable(a):
        return isinstance(a, (TensorDict, list))


class TensorList(list):
    """Container mainly used for lists of jittor tensors."""

    def __init__(self, list_of_tensors=None):
        if list_of_tensors is None:
            list_of_tensors = list()
        super(TensorList, self).__init__(list_of_tensors)

    def __deepcopy__(self, memodict={}):
        return TensorList(copy.deepcopy(list(self), memodict))

    def __getitem__(self, item):
        if isinstance(item, int):
            return super(TensorList, self).__getitem__(item)
        elif isinstance(item, (tuple, list)):
            return TensorList([super(TensorList, self).__getitem__(i) for i in item])
        else:
            return TensorList(super(TensorList, self).__getitem__(item))

    def __add__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 + e2 for e1, e2 in zip(self, other)])
        return TensorList([e + other for e in self])

    def __radd__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 + e1 for e1, e2 in zip(self, other)])
        return TensorList([other + e for e in self])

    def __mul__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 * e2 for e1, e2 in zip(self, other)])
        return TensorList([e * other for e in self])

    def __rmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 * e1 for e1, e2 in zip(self, other)])
        return TensorList([other * e for e in self])

    def concat(self, other):
        return TensorList(super(TensorList, self).__add__(other))

    def copy(self):
        return TensorList(super(TensorList, self).copy())

    def attribute(self, attr: str, *args):
        return TensorList([getattr(e, attr, *args) for e in self])

    def apply(self, fn):
        return TensorList([fn(e) for e in self])

    def __getattr__(self, name):
        if not hasattr(jt.Var, name):
            raise AttributeError('\'TensorList\' object has not attribute \'{}\''.format(name))

        def apply_attr(*args, **kwargs):
            return TensorList([getattr(e, name)(*args, **kwargs) for e in self])

        return apply_attr

    @staticmethod
    def _iterable(a):
        return isinstance(a, (TensorList, list))
