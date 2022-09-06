
from pprint import pprint
from towhee.engine.new_towhee_engine.dynamic_dispatch import dynamic_dispatch
from towhee.hparam import param_scope
import uuid
# pylint: disable=protected-access
# pylint: disable=redefined-builtin
__all__ = ['input']

@dynamic_dispatch
def input(*args, **kwargs):
    pipeline = Pipeline()
    with param_scope() as hp:
        index = hp._index

    return getattr(pipeline, 'input')[index](*args, **kwargs)

class Pipeline:
    """New Pipeline Design

    Starting code for new pipeline design. This class allows the user to define any pipeline, and will
    generate a dag for that pipeline with all the details.
    """
    def __init__(self):
        self._dag = {}
        self._input = None

    def __getattr__(self, name):

        @dynamic_dispatch
        def wrapper(*args, **kwargs):
            with param_scope() as hp:
                path = hp._name
                index = hp._index

            unique_id = uuid.uuid4().hex
            dag_info = {}

            if self._input is None:
                self._input = unique_id
            #     if isinstance(index, str) or len(index) == 1:
            #         dag_info['input'] = None
            #         dag_info['output'] = (index, ) if isinstance(index, str) else index
            #     else:
            #         dag_info['input'] = None
            #         dag_info['output'] = index[1]
            # else:
            #     dag_info = {}
            #     if isinstance(index, str) or len(index) == 1:
            #         dag_info['input'] = (index, ) if isinstance(index, str) else index
            #         dag_info['output'] = None
            #     else:
            #         dag_info['input'] = index[0]
            #         dag_info['output'] = index[1]
            dag_info['input'] = index[0]
            dag_info['output'] = index[1]
            dag_info['op'] = path
            dag_info['args'] = args
            dag_info['kwargs'] = kwargs

            self._dag[unique_id] = dag_info

            return self

        return getattr(wrapper, name)


if __name__ == '__main__':
    import pipeline as p
    x = p.input[('output1'), ('output2')](input_arg = 'input_arg').towhee.function['test_schema'](test_arg = 'test_arg')
    pprint(x._dag)

