from queue import Queue
import asyncio
import threading

from towhee.utils.log import engine_log
from towhee.functional.option import Option, Empty, _Reason
from towhee.functional.mixins.parallel import EOS


def _map_task_ray(unary_op):
    def map_wrapper(x):
        try:
            if isinstance(x, Option):
                return x.map(unary_op)
            else:
                return unary_op(x)
        except Exception as e:  # pylint: disable=broad-except
            engine_log.warning(f'{e}, please check {x} with op {unary_op}. Continue...')  # pylint: disable=logging-fstring-interpolation
            return Empty(_Reason(x, e))

    return map_wrapper


class RayMixin:
    """
    Mixin for parallel ray execution.
    """

    def _ray_pmap(self, unary_op, num_worker=None):
        import ray #pylint: disable=import-outside-toplevel
        ###### ACTOR: Currently much slower, may have future use.

        # @ray.remote
        # class RemoteActor:
        #     def remote_runner(self, val):
        #         return _map_task_ray(unary_op)(val)

        # pool = ray.util.ActorPool([RemoteActor.remote() for _ in range(num_worker)])

        # pool.submit(lambda a, v: a.remote_runner.remote(v), x)
        # if pool.has_next():
        #    queue.put(pool.get_next())

        ###### TASK: Need a good way of transferring the model.
        if num_worker is not None:
            pass
        elif self.get_executor() is not None:
            num_worker = self._num_worker
        else:
            num_worker = 2

        queue = Queue(num_worker)
        loop = asyncio.new_event_loop()

        def inner():
            while True:
                x = queue.get()
                if isinstance(x, EOS):
                    break
                else:
                    yield x

        @ray.remote
        def remote_runner(val):
            return _map_task_ray(unary_op)(val)

        async def worker():
            buff = []
            for x in self:
                if len(buff) == num_worker:
                    queue.put(await buff.pop(0))
                buff.append(asyncio.wrap_future(remote_runner.remote(x).future()))
            while len(buff) > 0:
                queue.put(await buff.pop(0))
            queue.put(EOS())

        def worker_wrapper():
            loop.run_until_complete(worker())
            loop.close()

        t = threading.Thread(target=worker_wrapper)
        t.start()

        return self._factory(inner())
