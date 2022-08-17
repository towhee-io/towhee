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
        # If the function is called from an existing dc
        index_info = param_scope()._index # pylint: disable=protected-access
        if index_info is None:
            input_info = None
            output_info = None
        elif isinstance(index_info, tuple):
            input_info = list(index_info[0]) if isinstance(index_info[0], tuple) else [index_info[0]]
            output_info = list(index_info[1]) if isinstance(index_info[1], tuple) else [index_info[1]]
        else:
            input_info = None
            output_info = list(index_info) if isinstance(index_info, tuple) else [index_info]
        if isinstance(self, dc_type):
            self.op = f.__name__
            self.call_args = {'*arg': arg, '*kws': kws}
            if arg != tuple():
                if hasattr(arg[0], '_op_config'):
                    self.op_config = arg[0].op_config
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
                    'op_config': self.get_pipeline_config(),
                    'input_info': input_info,
                    'output_info': output_info,
                    'parent_ids': self.parent_ids,
                    'child_ids':  self.child_ids}
            # if has op_config, update op_config
            if self.op_config is not None:
                info['op_config'].update(self.op_config)
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
                    'op_config': None,
                    'input_info': input_info,
                    'output_info': output_info,
                    'parent_ids': [],
                    'child_ids':  child_ids}
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
        self.op_config = None
        self.input_info = None
        self.output_info = None
        self.child_ids = []

    def register_dag(self, children):
        # check if list of dc or just dc
        if isinstance(children, type(self)):
            self.child_ids = [children.id]
        else:
            self.child_ids = [x.id for x in children]
        info = {'op': self.op,
                'op_name': self.op_name,
                'is_stream': self.is_stream,
                'init_args': self.init_args,
                'call_args': self.call_args,
                'op_config': self.op_config,
                'input_info': self.input_info,
                'output_info': self.output_info,
                'parent_ids': self.parent_ids,
                'child_ids':  self.child_ids}
        self._control_plane.dag[self.id] = info
        return children

    def notify_consumed(self, new_id):
        info = {'op': 'nop',
                'op_name': None,
                'init_args': None,
                'call_args': None,
                'op_config': None,
                'input_info': None,
                'output_info': None,
                'parent_ids': self.parent_ids,
                'child_ids': [new_id]}
        self._control_plane.dag[self.id] = info

    def compile_dag(self):
        info = {'op': 'nop',
                'op_name': None,
                'init_args': None,
                'call_args': None,
                'op_config': None,
                'input_info': None,
                'output_info': None,
                'parent_ids': self.parent_ids,
                'child_ids': ['end']}
        self._control_plane.dag[self.id] = info
        info = {'op': 'end',
                'op_name': None,
                'init_args': None,
                'call_args': None,
                'op_config': None,
                'input_info': None,
                'output_info': None,
                'parent_ids': [self.id],
                'child_ids': []}
        self._control_plane.dag['end'] = info
        # return self._control_plane.dag
        return self._clean_nops(self._control_plane.dag)

    def netx(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        compiled_dag = self.compile_dag()
        new_dict = {}
        label_dict = {}
        for key,value in compiled_dag.items():
            # print(key, value)
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
        return self._add_op_name_and_init_args(dag_copy)

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
        return self._solve_child_ids(dag, start)

    def _solve_child_ids(self, dag, start):
        co_dag = dag.copy()
        for k, v in dag.items():
            if start in v['parent_ids']:
                co_dag[k]['parent_ids'].append('start')
                co_dag[k]['parent_ids'].remove(start)
        return self._solve_input_info(co_dag)

    def _solve_input_info(self, dag):
        copy_dag = dag.copy()
        for key, value in dag.items():
            if isinstance(value['input_info'], list):
                input_info = []
                for i in value['input_info']:
                    for j in value['parent_ids']:
                        if j == 'start':
                            input_info.append(tuple(['start', i]))
                        elif dag[j]['output_info'] is not None and (i in dag[j]['output_info']) :
                            input_info.append(tuple([j, i]))
                    copy_dag[key]['input_info'] = input_info
        return copy_dag

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
