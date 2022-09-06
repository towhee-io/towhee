from typing import Any
from typing import Callable
from towhee.hparam import param_scope


class DynamicDispatch:
    """
    Dynamic call dispatch

    Examples:

    >>> @dynamic_dispatch
    ... def debug_print(*args, **kws):
    ...     hp = param_scope()
    ...     name = hp._name
    ...     index = hp._index
    ...     return (name, index, args, kws)

    >>> debug_print()
    (None, None, (), {})
    >>> debug_print.a()
    ('a', None, (), {})
    >>> debug_print.a.b.c()
    ('a.b.c', None, (), {})
    >>> debug_print[1]()
    (None, 1, (), {})
    >>> debug_print[1,2]()
    (None, (1, 2), (), {})
    >>> debug_print(1,2, a=1,b=2)
    (None, None, (1, 2), {'a': 1, 'b': 2})

    >>> debug_print.a.b.c[1,2](1, 2, a=1, b=2)
    ('a.b.c', (1, 2), (1, 2), {'a': 1, 'b': 2})
    """

    def __init__(self, func: Callable, name=None, index=None):
        self._func = func
        self._name = name
        self._index = index

    def __call__(self, *args, **kws) -> Any:
        with param_scope(_index=self._index, _name=self._name):
            return self._func(*args, **kws)

    def __getattr__(self, name: str) -> Any:
        if self._name is not None:
            name = f'{self._name}.{name}'
        return dynamic_dispatch(self._func, name, self._index)

    def __getitem__(self, index):
        return dynamic_dispatch(self._func, self._name, index)


def dynamic_dispatch(func, name=None, index=None):
    """Wraps function with a class to allow __getitem__ and __getattr__ on a function.
    """
    new_class = type(func.__name__, (
        DynamicDispatch,
        object,
    ), dict(__doc__=func.__doc__))
    return new_class(func, name, index)
