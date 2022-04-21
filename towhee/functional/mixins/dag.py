from uuid import uuid4
from copy import deepcopy

from towhee.hparam import param_scope
from towhee.functional.control_plane import ControlPlane


OP_EQUIVALENTS = {
    'split': 'nop'
}


class DagMixin:
    """
    Mixin for creating DAGs and their corresponding yamls from a DC
    """

    def __init__(self) -> None:
        super().__init__()
        # Unique id for current operation
        self._id = str(uuid4().hex[:8])
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        if parent == None:
            self._control_plane = ControlPlane()
            self._control_plane._ssa = {}
        self._parent_id = parent._id
        self._control_plane = parent._control_plane
        self._other_ops = []
        self._child_id = None
        self._op = None
        self._init_args = None
        self._consumed = False


    def add_ssa(op, init_args, input_dc_)
    