
from pprint import pprint
from queue import Queue
from towhee.engine.new_towhee_engine.operation import ops
from towhee.engine.new_towhee_engine.dataframe import DataFrame
from towhee.engine.new_towhee_engine.iterator import BatchIterator, FlatMapIterator, MapIterator, WindowIterator, FilterIterator
from towhee.engine.new_towhee_engine.utils import flatten
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
        self._output = None

    @classmethod
    def input(cls, output):
        uid = uuid.uuid4().hex
        pipeline = cls()
        pipeline._input = uid
        pipeline._dag[uid] = {
            'fn': None,
            'input': None,
            'output': output,
            'fn_type': 'input',
            'iteration': 'map'
        }
        return pipeline

    def output(self, input):
        uid = uuid.uuid4().hex
        self._output = uid
        self._dag[uid] = {
            'fn': None,
            'input': input,
            'output': None,
            'fn_type': 'output',
            'iteration': 'map'
        }
        # self._compiled_dag = self._compile_graph()
        self._connect_outputs()
        return self


    def map(self, input, output, fn):
        uid = uuid.uuid4().hex
        if isinstance(fn, tuple):
            fn_type = 'operator'
        elif getattr(fn, '__name__', None) == '<lambda>':
            fn_type = 'lambda'
        elif callable(fn):
            fn_type = 'callable'

        self._dag[uid] = {
            'fn': fn,
            'input': input,
            'output': output,
            'fn_type': fn_type,
            'iteration': 'map'
        }
        return self
    
 

    def flat_map(self, input, output, fn):
        uid = uuid.uuid4().hex
        if isinstance(fn, tuple):
            fn_type = 'operator'
        elif getattr(fn, '__name__', None) == '<lambda>':
            fn_type = 'lambda'
        elif callable(fn):
            fn_type = 'callable'

        self._dag[uid] = {
            'fn': fn,
            'input': input,
            'output': output,
            'fn_type': fn_type,
            'iteration': 'flat_map'
        }
        return self

    def filter(self, input, output, fn):
        uid = uuid.uuid4().hex
        if isinstance(fn, tuple):
            fn_type = 'operator'
        elif getattr(fn, '__name__', None) == '<lambda>':
            fn_type = 'lambda'
        elif callable(fn):
            fn_type = 'callable'

        self._dag[uid] = {
            'fn': fn,
            'input': input,
            'output': output,
            'filter_type': fn_type,
            'iteration': 'filter'
        }
        return self

    def time_window(self, input, output, timestamp_col, size, step, fn):
        uid = uuid.uuid4().hex
        if isinstance(fn, tuple):
            fn_type = 'operator'
        elif getattr(fn, '__name__', None) == '<lambda>':
            fn_type = 'lambda'
        elif callable(fn):
            fn_type = 'callable'

        self._dag[uid] = {
            'fn': fn,
            'input': input,
            'output': output,
            'fn_type': fn_type,
            'iteration': 'time_window',
            'step': step,
            'size': size,
            'timestamp_col': timestamp_col
        }
        return self


    def batch(self, input, output, size, step, fn):
        uid = uuid.uuid4().hex
        if isinstance(fn, tuple):
            fn_type = 'operator'
        elif getattr(fn, '__name__', None) == '<lambda>':
            fn_type = 'lambda'
        elif callable(fn):
            fn_type = 'callable'

        self._dag[uid] = {
            'fn': fn,
            'input': input,
            'output': output,
            'fn_type': fn_type,
            'iteration': 'batch',
            'step': step,
            'size': size,
        }
        return self
    
    def batch_all(self, input, output, fn):
        uid = uuid.uuid4().hex
        if isinstance(fn, tuple):
            fn_type = 'operator'
        elif getattr(fn, '__name__', None) == '<lambda>':
            fn_type = 'lambda'
        elif callable(fn):
            fn_type = 'callable'

        self._dag[uid] = {
            'fn': fn,
            'input': input,
            'output': output,
            'fn_type': fn_type,
            'iteration': 'batch_all'
        }
        return self

    

    def _connect_outputs(self):
        """Connect the inputs to outputs for nodes.

        Each node output will be a seperate towhee.Array, the corresponding array to output is done in postorder

        Returns:
            _type_: _description_
        """
        self._col_to_dataframe = {}
        for values in self._dag.values():
            if isinstance(values['output'], str):
                data = DataFrame(name=values['output'])
                values['output_datafarmes'] = [data]
                self._col_to_dataframe[values['output']] = data

            elif isinstance(values['output'], tuple):
                values['output_dataframes'] = []
                for outputs in flatten(values['output']):
                    data = DataFrame(name=outputs)
                    values['output_dataframes'].append(data)
                    self._col_to_dataframe[outputs] = data

        for values in self._dag.values():
            pprint(values)
            if isinstance(values['input'], str):
                queue = Queue()
                values['input_queues'] = [queue]
                self._col_to_dataframe[values['input']].add_queue(queue)

            elif isinstance(values['input'], tuple):
                values['input_queues'] = []
                for inputs in flatten(values['input']):
                    queue = Queue()
                    values['input_queues'].append(queue)
                    self._col_to_dataframe[inputs].add_queue(queue)



if __name__ == '__main__':
    def f(x):
        return x + 1
    x = Pipeline.input(('a', 'b')) \
        .map('a', 'c', ops.towhee.decode('a', b ='b')) \
        .map(('a', 'b'), 'd', lambda x, y: x + y) \
        .map('a', 'e', f) \
        .map('a', (('f', 'g'), 'h'), lambda x: x).output('e')
    pprint (x._dag)


