from typing import Any
from typing import Callable


class RuntimeParser:
    """Runtime parsing of unkown attributes.

    Example:
    >>> from towhee.pipeline.runtime_parser import runtime_parse
    >>> @runtime_parse
    ... def foo(name, *args, **kwargs):
    ...     return str((name, args, kwargs))
    >>> print(foo.bar.zed(1, 2, 3))
    ('bar.zed', (1, 2, 3), {})
    """

    def __init__(self, func: Callable, name=None):
        self._func = func
        self._name = name

    def __call__(self, *args, **kws) -> Any:
        return self._func(self._name, *args, **kws)

    def __getattr__(self, name: str) -> Any:
        if self._name is not None:
            name = f'{self._name}.{name}'
        return runtime_parse(self._func, name)


def runtime_parse(func, name=None):
    """Wraps function with a class to allow __getattr__ on a function.
    """
    new_class = type(func.__name__, (
        RuntimeParser,
        object,
    ), dict(__doc__=func.__doc__))
    return new_class(func, name)
