from uuid import uuid4

from towhee.hparam import param_scope
from towhee.functional.control_plane import ControlPlane


OP_EQUIVALENTS = {
    'split': 'nop'
}


def close_dc(f):
    def wrapper(self, *args):
        #ADD SSA STUFF HERE
        info = {'op': f.__name__, 'init_args': None, 'call_args': args, 'parent_id': self.parent_id, 'child_id':  ['end']}
        self.control_plane.dag[self.id] = info
        return f(self, *args)
    return wrapper

def register_dag(f):
    def wrapper(self, *args, **kwargs):
        children = f(self, *args, **kwargs)
        # check if list of dc or just dc
        if isinstance(children, type(self)):
            children_ids = [children.id]
        else:
            children_ids = [x.id for x in children]
        info = {'op': self.op, 'is_stream': self.is_stream, 'init_args': self.init_args, 'call_args': self.call_args, 'parent_id': self.parent_id, 'child_id':  children_ids}
        self.control_plane.dag[self.id] = info
        return children

    return wrapper
        

class DagMixin:
    """
    Mixin for creating DAGs and their corresponding yamls from a DC
    """

    def __init__(self) -> None:
        super().__init__()
        # Unique id for current dc
        self._id = str(uuid4().hex[:8])
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        if parent is None:
            self._parent_id = ['start']
            self._control_plane = ControlPlane()
        else:
            self._parent_id = [parent.id]
            self._control_plane = parent.control_plane
        self._op = None
        self._init_args = None
        self._call_args = None
        self._child_id = []

    @property
    def id(self):
        return self._id

    @property
    def parent_id(self):
        return self._parent_id

    @property
    def child_id(self):
        return self._child_id

    @property
    def control_plane(self):
        return self._control_plane

    @property
    def init_args(self):
        return self._init_args

    @property
    def call_args(self):
        return self._call_args

    @property
    def op(self):
        return self._op

    def register_dag(self, children):
        # check if list of dc or just dc
        if isinstance(children, type(self)):
            children_ids = [children.id]
        else:
            children_ids = [x.id for x in children]
        info = {'op': self.op, 'is_stream': self.is_stream, 'init_args': self.init_args, 'call_args': self.call_args, 'parent_id': self.parent_id, 'child_id':  children_ids}
        self.control_plane.dag[self.id] = info
        return children

    def notify_consumed(self, new_id):
        print(self.parent_id)
        info = {'op': 'nop', 'init_args': None, 'call_args': None, 'parent_id': self.parent_id, 'child_id':  [new_id]}
        self.control_plane.dag[self.id] = info

    def compile_dag(self):
        info = {'op': 'nop', 'init_args': None, 'call_args': None, 'parent_id': self.parent_id, 'child_id':  ['end']}
        self.control_plane.dag[self.id] = info
        return self.control_plane.dag

    # def __getattribute__(self, __name: str):
    #     if super().__getattribute__('_consumed'):
    #         print('Already consumed.')
    #     else:
    #         return super().__getattribute__(__name)

        

