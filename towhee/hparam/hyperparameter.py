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
        return default if value is None else value

    def __getattr__(self, name: str) -> Any:
        if name in ['_path', '_root']:
            return self[name]

        if self._path:
            name = '{}.{}'.format(self._path, name)
        return _Accessor(self._root, name)

    def __setattr__(self, name: str, value: Any):
        if name in ['_path', '_root']:
            return self.__setitem__(name, value)
        full_name = '{}.{}'.format(self._path,
                                   name) if self._path is not None else name
        _write_tracker.add(full_name)
        root = self._root
        root.put(full_name, value)
        # for path in self._path.split('.'):
        #     root[path] = HyperParameter()
        #     root = root[path]
        # root[name] = value
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


class _CallHolder(dict):
    """
    Helper for tracking function calls.

    Examples:
    >>> ch = _CallHolder()
    >>> ch.my.foo(a=1,b=2)
    ('my.foo', (), {'a': 1, 'b': 2})

    >>> ch.myspace2.gee(c=1,d=2)
    ('myspace2.gee', (), {'c': 1, 'd': 2})
    """

    @staticmethod
    def default_callback(path, *arg, **kws):
        return (path, arg, kws)

    def __init__(self, callback: Callable=None, path=None):
        super().__init__()
        self._callback = callback if callback is not None else _CallHolder.default_callback
        self._path = path

    def __getattr__(self, name: str) -> Any:
        if name in ['_path', '_callback']:
            return self[name]

        if self._path:
            name = '{}.{}'.format(self._path, name)
        return _CallHolder(self._callback, name)

    def __setattr__(self, name: str, value: Any):
        if name in ['_path', '_callback']:
            return self.__setitem__(name, value)
        return

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._callback(self._path, *args, **kwds)


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
        super(HyperParameter, self).__init__()  # pylint: disable=super-with-arguments
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
        _write_tracker.add(name)
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
                return None
            obj = obj[p]
        _read_tracker.add(name)
        return obj[path[-1]] if path[-1] in obj else None

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
        """
        if name in self.keys():
            return self[name]
        else:
            if name in self.__dict__.keys():
                return self.__dict__[name]
            return _Accessor(self, name)

    def __setattr__(self, name, value):
        """
        create/update parameter with object-style api

        Examples:
        >>> hp = HyperParameter(a=1, b = {'c':2, 'd': 3})
        >>> hp.e = 4

        >>> hp['e']
        4
        """
        self[name] = value

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

    def callholder(self, callback: Callable = None):
        """
        Return a call holder.

        Examples:
        >>> ch = param_scope().callholder()
        >>> ch.my.foo(a=1,b=2)
        ('my.foo', (), {'a': 1, 'b': 2})

        >>> ch.myspace2.gee(c=1,d=2)
        ('myspace2.gee', (), {'c': 1, 'd': 2})
        """
        return _CallHolder(callback)

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


def auto_param(func):
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
    """
    predef_kws = {}
    predef_val = {}

    namespace = func.__module__
    if namespace == '__main__':
        namespace = None
    if namespace is not None:
        namespace += '.{}'.format(func.__name__)
    else:
        namespace = func.__name__

    signature = inspect.signature(func)
    for k, v in signature.parameters.items():
        if v.default != v.empty:
            name = '{}.{}'.format(namespace, k)
            predef_kws[k] = name
            _read_tracker.add(name)
            predef_val[name] = v.default

    def wrapper(*arg, **kws):
        with param_scope() as hp:
            local_params = {}
            for k, v in predef_kws.items():
                if hp.get(v) is not None and k not in kws:
                    kws[k] = hp.get(v)
                    local_params[v] = hp.get(v)
                else:
                    local_params[v] = predef_val[v]
            if _callback is not None:
                _callback(local_params)
            return func(*arg, **kws)

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


if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=False)

    import unittest

    class TestHyperParameter(unittest.TestCase):
        """
        tests for HyperParameter
        """

        def test_parameter_create(self):
            param1 = HyperParameter(a=1, b=2)
            self.assertEqual(param1.a, 1)
            self.assertEqual(param1.b, 2)

            param2 = HyperParameter(**{'a': 1, 'b': 2})
            self.assertEqual(param2.a, 1)
            self.assertEqual(param2.b, 2)

        def test_parameter_update_with_holder(self):
            param1 = HyperParameter()
            param1.a = 1
            param1.b = 2
            param1.c.b.a = 3
            self.assertDictEqual(param1, {
                'a': 1,
                'b': 2,
                'c': {
                    'b': {
                        'a': 3
                    }
                }
            })

        def test_parameter_update(self):
            param1 = HyperParameter()
            param1.put('c.b.a', 1)
            self.assertDictEqual(param1, {'c': {'b': {'a': 1}}})

        def test_parameter_patch(self):
            param1 = HyperParameter()
            param1.update({'a': 1, 'b': 2})
            self.assertEqual(param1.a, 1)
            self.assertEqual(param1.b, 2)

    class TestAccesscor(unittest.TestCase):
        """
        tests for Accesscor
        """

        def test_holder_as_bool(self):
            param1 = HyperParameter()
            self.assertFalse(param1.a.b)

            param1.a.b = False
            self.assertFalse(param1.a.b)

            param1.a.b = True
            self.assertTrue(param1.a.b)

    class TestParamScope(unittest.TestCase):
        """
        tests for param_scope
        """

        def test_scope_create(self):
            with param_scope(a=1, b=2) as hp:
                self.assertEqual(hp.a, 1)
                self.assertEqual(hp.b, 2)

            with param_scope(**{'a': 1, 'b': 2}) as hp:
                self.assertEqual(hp.a, 1)
                self.assertEqual(hp.b, 2)

        def test_nested_scope(self):
            with param_scope(a=1, b=2) as hp1:
                self.assertEqual(hp1.a, 1)

                with param_scope(a=3) as hp2:
                    self.assertEqual(hp2.a, 3)

        def test_scope_with_function_call(self):

            def read_a():
                with param_scope() as hp:
                    return hp.a

            self.assertFalse(read_a())

            with param_scope(a=1):
                self.assertEqual(read_a(), 1)
            with param_scope(a=2):
                self.assertEqual(read_a(), 2)

            with param_scope(a=1):
                self.assertEqual(read_a(), 1)
                with param_scope(a=2):
                    self.assertEqual(read_a(), 2)
                self.assertEqual(read_a(), 1)

    unittest.main()
