from queue import Queue
import asyncio
import threading
import uuid
from towhee.hub.file_manager import FileManagerConfig

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

    def ray_resolve(self, call_mapping, path, index, *arg, **kws):
        import ray #pylint: disable=import-outside-toplevel

        # TODO: call mapping solution
        y = call_mapping #pylint: disable=unused-variable

        @ray.remote
        class OperatorActor:
            """Ray actor that runs hub operators."""

            def __init__(self, path1, index1, uid, *arg1, **kws1):
                from towhee import engine #pylint: disable=import-outside-toplevel
                from towhee.engine.factory import _ops_call_back #pylint: disable=import-outside-toplevel
                from pathlib import Path #pylint: disable=import-outside-toplevel

                engine.DEFAULT_LOCAL_CACHE_ROOT = Path.home() / ('.towhee/ray_actor_cache_' + uid)
                engine.LOCAL_PIPELINE_CACHE = engine.DEFAULT_LOCAL_CACHE_ROOT / 'pipelines'
                engine.LOCAL_OPERATOR_CACHE = engine.DEFAULT_LOCAL_CACHE_ROOT / 'operators'
                x = FileManagerConfig()
                x.update_default_cache(engine.DEFAULT_LOCAL_CACHE_ROOT)
                self.op = _ops_call_back(path1, index1, *arg1, **kws1)

            def __call__(self, *arg1, **kwargs1):
                return self.op(*arg1, **kwargs1)

            def cleanup(self):
                from shutil import rmtree #pylint: disable=import-outside-toplevel
                from towhee import engine #pylint: disable=import-outside-toplevel
                rmtree(engine.DEFAULT_LOCAL_CACHE_ROOT)

        actors = [OperatorActor.remote(path, index, str(uuid.uuid4().hex[:12].upper()), *arg, **kws) for _ in range(self._num_worker)]
        pool = ray.util.ActorPool(actors)
        queue = Queue(self._num_worker)

        def inner():
            while True:
                x = queue.get()
                if isinstance(x, EOS):
                    break
                else:
                    yield x
            for x in actors:
                x.cleanup.remote()

        def worker():
            for x in self:
                while pool.has_free() is False:
                    if pool.has_next():
                        queue.put(pool.get_next())
                pool.submit(lambda a, v: a.__call__.remote(v), x)
            while pool.has_next():
                queue.put(pool.get_next())
            queue.put(EOS())

        t = threading.Thread(target=worker)
        t.start()

        return self._factory(inner())


    def _ray_pmap(self, unary_op, num_worker=None):
        import ray #pylint: disable=import-outside-toplevel

        if num_worker is not None:
            pass
        elif self.get_executor() is not None:
            num_worker = self._num_worker
        else:
            num_worker = 2

        #If not streamed, we need to be able to hold all values within queue
        if self.is_stream:
            queue = Queue(num_worker)
        else:
            queue = Queue()

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
