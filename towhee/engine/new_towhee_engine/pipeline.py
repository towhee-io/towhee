
from pprint import pprint
from towhee.engine.new_towhee_engine.dynamic_dispatch import dynamic_dispatch
from towhee.hparam import param_scope
import uuid
# pylint: disable=protected-access
# pylint: disable=redefined-builtin


class Pipeline:
    """New Pipeline Design

    Starting code for new pipeline design. This class allows the user to define any pipeline, and will
    generate a dag for that pipeline with all the details.
    """
    def __init__(self):
        self._dag = {}
        self._input = None
    
    @classmethod
    def input(cls, output):
        uid = uuid.uuid4().hex
        pipeline = cls()
        pipeline._input = uid
        pipeline._dag[uid] = {
            'op': None,
            'input': None,
            'output': output,
            'op_type': 'input'
        }
        return pipeline
    
    def map(self, input, output, op):
        uid = uuid.uuid4().hex
        if isinstance(op, tuple):
            op_type = 'operator'
        elif getattr(op, '__name__', None) == '<lambda>':
            op_type = 'lambda'
        elif callable(op):
            op_type = 'callable'

        self._dag[uid] = {
            'op': op,
            'input': input,
            'output': output,
            'op_type': op_type
        }
        return self


class Operation:
    def init(self):
        pass

    @classmethod
    def __getattr__(cls, name):

        @dynamic_dispatch
        def wrapper(*args, **kwargs):
            with param_scope() as hp:
                path = hp._name

            return (path, args, kwargs)

        return getattr(wrapper, name)

ops = Operation()

if __name__ == '__main__':
    def f(x):
        return x + 1
    x = Pipeline.input(('a,', 'b')).map('a', 'c', ops.towhee.decode('a', b ='b')).map(('a', 'b'), 'd', lambda x, y: x + y).map('a', 'e', f)
    pprint (x._dag)


