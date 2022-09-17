from towhee.engine.factory import create_op, op
from towhee.engine.new_towhee_engine.dynamic_dispatch import dynamic_dispatch
from towhee.hparam import param_scope


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