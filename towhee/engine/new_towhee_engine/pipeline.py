
from pprint import pprint
from towhee.engine.new_towhee_engine.dynamic_dispatch import dynamic_dispatch
from towhee.engine.new_towhee_engine.utils import flatten
from towhee.hparam import param_scope
import uuid
from towhee.engine.new_towhee_engine.array import Array
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
        self._output = None

    @classmethod
    def input(cls, output):
        uid = uuid.uuid4().hex
        pipeline = cls()
        pipeline._input = uid
        pipeline._dag[uid] = {
            'op': None,
            'input': None,
            'output': output,
            'op_type': 'input',
            'iteration': 'map'
        }
        return pipeline

    def output(self, input):
        uid = uuid.uuid4().hex
        self._output = uid
        self._dag[uid] = {
            'op': None,
            'input': input,
            'output': None,
            'op_type': 'output',
            'iteration': 'map'
        }
        # self._compiled_dag = self._compile_graph()
        self._connect_outputs()
        return self


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
            'op_type': op_type,
            'iteration': 'map'
        }
        return self

    def flat_map(self, input, output, op):
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
            'op_type': op_type,
            'iteration': 'flat_map'
        }
        return self

    def filter(self, input, output, filter):
        uid = uuid.uuid4().hex
        if isinstance(filter, tuple):
            op_type = 'operator'
        elif getattr(filter, '__name__', None) == '<lambda>':
            op_type = 'lambda'
        elif callable(filter):
            op_type = 'callable'

        self._dag[uid] = {
            'filter': filter,
            'input': input,
            'output': output,
            'op_type': op_type,
            'iteration': 'filter'
        }
        return self

    def window(self, input, output, window_param, op):
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
            'op_type': op_type,
            'iteration': 'window',
            'window': window_param
        }
        return self

    def _connect_outputs(self):
        """Connect the inputs to outputs for nodes.

        Each node output will be a seperate towhee.Array, the corresponding array to output is done in postorder

        Returns:
            _type_: _description_
        """
        #  Holds the connection for column -> node_id. 
        self._output_to_node = {}
        #  Holds the connection for column -> towhee array.
        self._output_to_array = {}
        for x in self._dag.values():
            if isinstance(x['output'], str):
                #  set the singular output array for specified column.
                x['output_array'] = [Array()]
                #  initialize the column reader tracker array.
                self._output_to_node[x['output']] = []
                #  Track which array is assigned to the column.
                self._output_to_array[x['output']] = x['output_array']
                
                
            elif isinstance(x['output'], tuple):
                x['output_array'] = []
                for y in flatten(x['output']):
                    new_array = Array()
                    x['output_array'].append(new_array)
                    self._output_to_node[y] = []
                    self._output_to_array[y] = [new_array]


        for node_id, values in self._dag.items():
            if isinstance(values['input'], str):
                values['input_array'] = self._output_to_array[values['input']]
                self._output_to_node[values['input']].append(node_id)
                
            elif isinstance(values['input'], tuple):
                values['input_array'] = []
                for x in flatten(values['input']):
                    values['input_array'].append(self._output_to_array[x])
                    self._output_to_node[x].append(node_id)



class Operation:
    """Class to allow Dynamic Dispatching of operators.
    """
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
    x = Pipeline.input(('a', 'b')) \
        .map('a', 'c', ops.towhee.decode('a', b ='b')) \
        .map(('a', 'b'), 'd', lambda x, y: x + y) \
        .map('a', 'e', f).output('e')
    pprint (x._dag)
    pprint(x._output_to_node)
    pprint(x._output_to_array)


