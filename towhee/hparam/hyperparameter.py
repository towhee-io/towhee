# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import json
import threading

from typing import Any, Dict, Set
from typing import Callable
# pylint: disable=pointless-string-statement
"""
Trackers that record all hyperparameter accesses.
"""
_read_tracker: Set[str] = set()
_write_tracker: Set[str] = set()


def reads():
    """
    Get hyperparameter read operations.

    Returns:
        List[str]: hyperparameter read operations

    Examples:
    >>> _read_tracker.clear()
    >>> hp = HyperParameter(a=1, b={'c': 2})
    >>> reads() # no read operations
    []

    >>> hp.a    # accessing parameter directly
    1
    >>> reads() # not tracked
    []

    >>> hp().a() # accessing with accessor
    1
    >>> reads()  # tracked!
    ['a']
    """
    retval = list(_read_tracker)
    retval.sort()
    return retval


def writes():
    """
    Get hyperparameter write operations.

    Returns:
        List[str]: hyperparameter write operations

    Examples:
    >>> _write_tracker.clear()
    >>> hp = HyperParameter(a=1, b={'c': 2})
    >>> writes()
    []

    >>> hp.a = 1
    >>> writes()
    []

    >>> hp().a = 1
    >>> hp().a.b.c = 1
    >>> writes()
    ['a', 'a.b.c']
    """
    retval = list(_write_tracker)
    retval.sort()
    return retval


def all_params():
    """
    Get all tracked hyperparameters.
    """
    retval = list(_read_tracker.union(_write_tracker))
    retval.sort()
    return retval


class _Accessor(dict):
    """
    Helper for accessing hyper-parameters.

    When reading an undefined parameter, the accessor will:
    1. return false in `if` statement:
    >>> params = HyperParameter()
    >>> if not params.undefined_int: print("parameter undefined")
    parameter undefined

    2. support default value for undefined parameter
    >>> params = HyperParameter()
    >>> params.undefined_int.get_or_else(10)
    10

    3. support to create nested parameter:
    >>> params = HyperParameter()
    >>> params.undefined_object.undefined_prop = 1
    >>> print(params)
    {'undefined_object': {'undefined_prop': 1}}
    """

    def __init__(self, root, path=None):
        super().__init__()
        self._root = root
        self._path = path

    def get_or_else(self, default: Any = None):
        """
        Get value for the parameter, or get default value if the parameter is not defined.
        """
        _read_tracker.add(self._path)
        value = self._root.get(self._path)
        return default if not value else value

    def __getattr__(self, name: str) -> Any:
        # _path and _root are not allowed as keys for user.
        if name in ['_path', '_root']:
            return self[name]

        if self._path:
            name = '{}.{}'.format(self._path, name)
        return _Accessor(self._root, name)

    def __setattr__(self, name: str, value: Any):
        # _path and _root are not allowed as keys for user.
        if name in ['_path', '_root']:
            return self.__setitem__(name, value)
        full_name = '{}.{}'.format(self._path,
                                   name) if self._path is not None else name
        _write_tracker.add(full_name)
        root = self._root
        root.put(full_name, value)
        return value

    def __str__(self):
        return ''

    def __bool__(self):
        return False

    def __call__(self, default: Any = None) -> Any:
        """
        shortcut for get_or_else
        """
        return self.get_or_else(default)

    __nonzero__ = __bool__


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
            name = '{}.{}'.format(self._name, name)
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


class HyperParameter(dict):
    """
    HyperParameter is an extended dict with features for better parameter management.

    A HyperParameter can be created with:
    >>> hp = HyperParameter(param1=1, param2=2, obj1={'propA': 'A'})

    or

    >>> hp = HyperParameter(**{'param1': 1, 'param2': 2, 'obj1': {'propA': 'A'}})

    Once the HyperParameter object is created, you can access the values using the object-style api:
    >>> hp.param1
    1
    >>> hp.obj1.propA
    'A'

    or using the dict-style api (for legacy codes):
    >>> hp['param1']
    1
    >>> hp['obj1']['propA']
    'A'

    The object-style api also support creating or updating the parameters:
    >>> hp.a.b.c = 1

    which avoid maintaining the dict data manually like this:
    >>> hp = {}
    >>> if 'a' not in hp: hp['a'] = {}
    >>> if 'b' not in hp['a']: hp['a']['b'] = {}
    >>> hp['a']['b']['c'] = 1

    You can also create a parameter with a string name:
    >>> hp = HyperParameter()
    >>> hp.put('a.b.c', 1)
    """

    def __init__(self, **kws):
        super().__init__()
        self.update(kws)

    def update(self, kws):
        for k, v in kws.items():
            if isinstance(v, dict):
                if k in self and isinstance(self[k], dict):
                    vv = HyperParameter(**self[k])
                    vv.update(v)
                    v = vv
                else:
                    v = HyperParameter(**v)
            self[k] = v

    def put(self, name: str, value: Any):
        """
        put/update a parameter with a string name

        Args:
            name (str): parameter name, 'obj.prop' is supported
            value (Any): parameter value

        Examples:
        >>> cfg = HyperParameter()
        >>> cfg.put('param1', 1)
        >>> cfg.put('obj1.propA', 'A')

        >>> cfg.param1
        1
        >>> cfg.obj1.propA
        'A'
        """
        path = name.split('.')
        obj = self
        for p in path[:-1]:
            if p not in obj or (not isinstance(obj[p], dict)):
                obj[p] = HyperParameter()
            obj = obj[p]
        obj[path[-1]] = safe_numeric(value)

    def get(self, name: str) -> Any:
        """
        get a parameter by a string name

        Args:
            name (str): parameter name

        Returns:
            Any: parameter value

        Examples:
        >>> cfg = HyperParameter(a=1, b = {'c':2, 'd': 3})
        >>> cfg.get('a')
        1
        >>> cfg.get('b.c')
        2
        """
        path = name.split('.')
        obj = self
        for p in path[:-1]:
            if p not in obj:
                return _Accessor(obj, p)
            obj = obj[p]
        return obj[path[-1]] if path[-1] in obj else _Accessor(self, name)

    def __setitem__(self, key, value):
        """
        set value and convert the value into `HyperParameter` if necessary
        """
        if isinstance(value, dict):
            return dict.__setitem__(self, key, HyperParameter(**value))
        return dict.__setitem__(self, key, value)

    def __getattr__(self, name):
        """
        read parameter with object-style api

        Examples:

        for simple parameters:
        >>> hp = HyperParameter(a=1, b = {'c':2, 'd': 3})
        >>> hp.a
        1

        for nested parameters:
        >>> hp.b.c
        2

        >>> getattr(hp, 'b.c')
        2
        """
        return self.get(name)
        # if name in self.keys():
        #     return self[name]
        # else:
        #     if name in self.__dict__.keys():
        #         return self.__dict__[name]
        #     return _Accessor(self, name)

    def __setattr__(self, name, value):
        """
        create/update parameter with object-style api

        Examples:
        >>> hp = HyperParameter(a=1, b = {'c':2, 'd': 3})
        >>> hp.e = 4

        >>> hp['e']
        4

        >>> setattr(hp, 'A.B.C', 1)
        >>> hp.A.B.C
        1
        """
        self.put(name, value)
        #self[name] = value

    def __call__(self) -> Any:
        """
        Return a parameter accessor.

        Returns:
            Any: holder of the current parameter

        Examples:
        >>> cfg = HyperParameter(a=1, b = {'c':2, 'd': 3})
        >>> cfg().a.get_or_else('default')   # default value for simple parameter
        1
        >>> cfg().b.c.get_or_else('default') # default value for nested parameter
        2
        >>> cfg().b.undefined.get_or_else('default')
        'default'
        """

        return _Accessor(self, None)

    def dispatch(self, callback: Callable = None):
        """
        Return a call holder.

        Examples:
        >>> def debug_print(path, index, *arg, **kws):
        ...     return (path, index, arg, kws)
        >>> ch = param_scope().dispatch(debug_print)
        >>> ch.my.foo(a=1,b=2)
        ('my.foo', None, (), {'a': 1, 'b': 2})

        >>> ch.myspace2.gee(c=1,d=2)
        ('myspace2.gee', None, (), {'c': 1, 'd': 2})
        """

        # pylint: disable=protected-access
        @dynamic_dispatch
        def wrapper(*arg, **kws):
            with param_scope() as hp:
                name = hp._name
                index = hp._index
            return callback(name, index, *arg, **kws)

        return wrapper

    def callholder(self, callback: Callable = None):
        return self.dispatch(callback)

    @staticmethod
    def loads(s):
        """
        Load parameters from JSON string, similar as `json.loads`.
        """
        obj = json.loads(s)
        return HyperParameter(**obj)

    @staticmethod
    def load(f):
        """
        Load parameters from json file, similar as `json.load`.
        """
        obj = json.load(f)
        return HyperParameter(**obj)


class param_scope(HyperParameter):  # pylint: disable=invalid-name
    """
    thread-safe scoped hyperparameter

    Examples:
    create a scoped HyperParameter
    >>> with param_scope(**{'a': 1, 'b': 2}) as cfg:
    ...     print(cfg.a)
    1

    read parameter in a function
    >>> def foo():
    ...    with param_scope() as cfg:
    ...        return cfg.a
    >>> with param_scope(**{'a': 1, 'b': 2}) as cfg:
    ...     foo() # foo should get cfg using a with statement
    1

    update some config only in new scope
    >>> with param_scope(**{'a': 1, 'b': 2}) as cfg:
    ...     cfg.b
    ...     with param_scope(**{'b': 3}) as cfg2:
    ...         cfg2.b
    2
    3
    """
    tls = threading.local()

    def __init__(self, *args, **kws):
        # Check if nested param_scope, if so, update current scope to include previous.
        if hasattr(param_scope.tls,
                   'history') and len(param_scope.tls.history) > 0:
            self.update(param_scope.tls.history[-1])
        self.update(kws)
        for line in args:
            if '=' in line:
                k, v = line.split('=', 1)
                self.put(k, v)

    def __enter__(self):
        if not hasattr(param_scope.tls, 'history'):
            param_scope.tls.history = []
        param_scope.tls.history.append(self)
        return param_scope.tls.history[-1]

    def __exit__(self, exc_type, exc_value, traceback):
        param_scope.tls.history.pop()

    @staticmethod
    def init(params):
        """
        init param_scope for a new thread.
        """
        if not hasattr(param_scope.tls, 'history'):
            param_scope.tls.history = []
            param_scope.tls.history.append(params)


"""
Tracker callback for auto_param
"""
_callback: Callable = None


def set_auto_param_callback(func: Callable[[Dict[str, Any]], None]):
    """ report hyperparameter value to a tracker, for example, `mlflow.tracking`
    """
    global _callback
    _callback = func


def auto_param(name_or_func):
    """
    Convert keyword arguments into hyperparameters

    Examples:

    >>> @auto_param
    ... def foo(a, b=2, c='c', d=None):
    ...     print(a, b, c, d)

    >>> foo(1)
    1 2 c None

    >>> with param_scope('foo.b=3'):
    ...     foo(2)
    2 3 c None

    classes are also supported:
    >>> @auto_param
    ... class foo:
    ...     def __init__(self, a, b=2, c='c', d=None):
    ...         print(a, b, c, d)

    >>> obj = foo(1)
    1 2 c None

    >>> with param_scope('foo.b=3'):
    ...     obj = foo(2)
    2 3 c None

    >>> @auto_param('my')
    ... def foo(a, b=2, c='c', d=None):
    ...     print(a, b, c, d)
    >>> foo(1)
    1 2 c None

    >>> with param_scope('foo.b=3'):
    ...     foo(2)
    2 2 c None

    >>> with param_scope('my.foo.b=3'):
    ...     foo(2)
    2 3 c None
    """

    if callable(name_or_func):
        return auto_param(None)(name_or_func)

    def wrapper(func):
        predef_kws = {}
        predef_val = {}

        if name_or_func is None:
            namespace = func.__name__
        else:
            namespace = name_or_func + '.' + func.__name__

        signature = inspect.signature(func)
        for k, v in signature.parameters.items():
            if v.default != v.empty:
                name = '{}.{}'.format(namespace, k)
                predef_kws[k] = name
                _read_tracker.add(name)
                predef_val[name] = v.default

        def inner(*arg, **kws):
            with param_scope() as hp:
                local_params = {}
                for k, v in predef_kws.items():
                    if getattr(
                            hp(),
                            v).get_or_else(None) is not None and k not in kws:
                        kws[k] = hp.get(v)
                        local_params[v] = hp.get(v)
                    else:
                        local_params[v] = predef_val[v]
                if _callback is not None:
                    _callback(local_params)
                return func(*arg, **kws)

        return inner

    return wrapper


def safe_numeric(value):
    if isinstance(value, str):
        try:
            return int(value)
        except:  # pylint: disable=bare-except
            pass
        try:
            return float(value)
        except:  # pylint: disable=bare-except
            pass
    return value
