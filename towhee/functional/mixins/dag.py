from functools import wraps
from uuid import uuid4
from abc import ABCMeta

from towhee.hparam import param_scope

def register_dag(f):
    @wraps(f)
    def wrapper(self, *arg, **kws):
        # Get the result DataCollections
        children = f(self, *arg, **kws)
        # Need the dc type while avoiding circular imports
        dc_type = type(children[0]) if isinstance(children, list) else type(children)

        # Grab the schema index
        with param_scope() as hp:
            # pylint: disable=protected-access
            index = hp._index

        # TODO: Update for schema fix
        # If tuple, first index is input, second is output
        if index is None:
            input_index = None
            output_index = None
        elif isinstance(index, tuple):
            input_index = list(index[0]) if isinstance(index[0], tuple) else [index[0]]
            output_index = list(index[1]) if isinstance(index[1], tuple) else [index[1]]
        else:
            input_index = None
            output_index = list(index) if isinstance(index, tuple) else [index]
        # If the function is called from an existing dc
        if isinstance(self, dc_type):
            self.op = f.__name__
            self.call_args = {'*arg': arg, '*kws': kws}
            # check if list of dc or just dc
            if isinstance(children, dc_type):
                self.child_ids = [children.id]
            else:
                self.child_ids = [x.id for x in children]
            info = {'op': self.op,
                    'op_name': self.op_name,
                    'is_stream': self.is_stream,
                    'init_args': self.init_args,
                    'call_args': self.call_args,
                    'parent_ids': self.parent_ids,
                    'child_ids':  self.child_ids,
                    'input_schema': input_index,
                    'output_schema': output_index}
            self.get_control_plane().dag[self.id] = info
            return children
        # If not called from a dc, think static or class method.
        else:
            op = f.__name__
            # if the method is being called from a classmethod, avoiding passing in self
            if isinstance(self, (dc_type, ABCMeta)):
                pass_args = arg
            else:
                pass_args = (self,) + arg
            call_args = {'*arg': pass_args, '*kws': kws}
            if isinstance(children, dc_type):
                child_ids = [children.id]
            else:
                child_ids = [x.id for x in children]
            info = {'op': op,
                    'op_name': None,
                    'is_stream': None,
                    'init_args': None,
                    'call_args': call_args,
                    'parent_ids': [],
                    'child_ids':  child_ids,
                    'input_schema': input_index,
                    'output_schema': output_index}
            # If not called from a dc, it means that it is a start method
            # so it must be added to the childrens dags.
            for x in children if isinstance(children, list) else  [children]:
                x.get_control_plane().dag['start'] = info
            return children

    return wrapper


class DagMixin:
    #pylint: disable=import-outside-toplevel
    """
    Mixin for creating DAGs and their corresponding yamls from a DC
    """
    def __init__(self) -> None:
        super().__init__()
        # Unique id for current dc
        self.id = str(uuid4().hex[:8])
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        if parent is None:
            self.parent_ids = ['start']
            self._control_plane = ControlPlane()
        else:
            self.parent_ids = [parent.id]
            self._control_plane = parent._control_plane
        self.op = None
        self.op_name = None
        self.init_args = None
        self.call_args = None
        self.child_ids = []

    def register_dag(self, children):
        # check if list of dc or just dc
        if isinstance(children, type(self)):
            self.child_ids = [children.id]
        else:
            self.child_ids = [x.id for x in children]

        # Grab the schema index
        with param_scope() as hp:
            # pylint: disable=protected-access
            index = hp._index
        # TODO: Update for schema fix
        # If tuple, first index is input, second is output
        if index is None:
            input_index = None
            output_index = None
        elif isinstance(index, tuple):
            input_index = list(index[0]) if isinstance(index[0], tuple) else [index[0]]
            output_index = list(index[1]) if isinstance(index[1], tuple) else [index[1]]
        else:
            input_index = None
            output_index = list(index) if isinstance(index, tuple) else [index]

        info = {'op': self.op,
                'op_name': self.op_name,
                'is_stream': self.is_stream,
                'init_args': self.init_args,
                'call_args': self.call_args,
                'parent_ids': self.parent_ids,
                'child_ids':  self.child_ids,
                'input_schema': input_index,
                'output_schema': output_index}
        self._control_plane.dag[self.id] = info
        return children

    def notify_consumed(self, new_id):
        info = {'op': 'nop', 'op_name': None, 'init_args': None, 'call_args': None, 'parent_ids': self.parent_ids, 'child_ids':  [new_id]}
        self._control_plane.dag[self.id] = info

    def compile_dag(self):
        """
        Clean and return the process dag.

        Returns:
            (dag: dict, schema_dict: dict):
                Returns both the dag and a dictionary that has the mapping of output to schema.
                The values in schema_dict are the (node_id, index of its output).
        Examples:

        >>> import towhee
        >>> from pprint import pprint
        >>> dc = towhee.dummy_input['input']()
        >>> dc = dc.runas_op['input', 'add_one'](lambda x: x+ 1)
        >>> dag, schema_dict = dc.compile_dag()
        >>> result = []
        >>> for x in dag.values():
        ...     if 'input_schema' in x.keys():
        ...         result.append((x['op_name'], x['input_schema'], x['output_schema']))
        >>> result
        [('towhee/runas-op', ['input'], ['add_one']), ('dummy_input', None, ['input'])]
        """
        info = {'op': 'nop','op_name': None, 'init_args': None, 'call_args': None, 'parent_ids': self.parent_ids, 'child_ids':  ['end']}
        self._control_plane.dag[self.id] = info
        info = {'op': 'end', 'op_name': None, 'init_args': None, 'call_args': None, 'parent_ids': [self.id], 'child_ids':  []}
        self._control_plane.dag['end'] = info
        # return self._control_plane.dag
        dag = self._clean_nops(self._control_plane.dag)
        dag = self._add_op_name_and_init_args(dag)
        dag, schema_dict = self._create_schema_dict(dag)
        return (dag, schema_dict)

    def netx(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        compiled_dag, _ = self.compile_dag()
        new_dict = {}
        label_dict = {}
        for key,value in compiled_dag.items():
            new_dict[key] = value['child_ids']
            label_dict[key] = value['op']
        label_dict['end'] = 'end'
        g = nx.DiGraph(new_dict)
        pos = nx.nx_pydot.graphviz_layout(g, 'dot')
        nx.draw_networkx(g, pos, labels=label_dict, with_labels = True)
        plt.show()

    def _clean_nops(self, dag):
        dag_copy = dag.copy()
        removals = []
        for key, val in dag.items():
            if val['op'] == 'nop':
                removals.append(key)
                for parent in val['parent_ids']:
                    dag_copy[parent]['child_ids'].remove(key)
                    dag_copy[parent]['child_ids'] = list(set(dag_copy[parent]['child_ids'] + val['child_ids']))
                for child in val['child_ids']:
                    dag_copy[child]['parent_ids'].remove(key)
                    dag_copy[child]['parent_ids'] = list(set(dag_copy[child]['parent_ids'] +val['parent_ids']))
        for x in removals:
            del dag_copy[x]
        return dag_copy

    def _create_schema_dict(self, dag):
        ret = {}
        for key, val in dag.items():
            schema = val.get('output_schema', None)
            if schema is not None:
                for i, x in enumerate(schema):
                    ret[x] = (key, i)
        return dag, ret
    

    def _add_op_name_and_init_args(self, dag):
        for key, val in dag.items():
            if (val['op'] == 'map' or val['op'] == 'filter') or val['call_args'] is not None:
                if val['call_args']['*arg'] != () and hasattr(val['call_args']['*arg'][0], '_kws'):
                    dag[key]['init_args'] = val['call_args']['*arg'][0].init_args
                    if len(val['call_args']['*arg'][0].function.split('/')) > 1:
                        dag[key]['op_name'] = val['call_args']['*arg'][0].function
                    else:
                        dag[key]['op_name'] = 'towhee/' + val['call_args']['*arg'][0].function
                else:
                    dag[key]['op_name'] = 'dummy_input'
                    dag[key]['parent_ids'] = []
                    start = key
            else:
                dag[key]['op_name'] = 'end'
        dag['start'] = dag[start]
        del dag[start]
        return dag


    def _clean_streams(self, dag):
        raise NotImplementedError

    def get_control_plane(self):
        return self._control_plane

class ControlPlane:
    def __init__(self) -> None:
        self._dag = {}

    @property
    def dag(self):
        return self._dag

if __name__ == '__main__':
    import doctest
    doctest.testmod()
