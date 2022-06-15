from queue import Queue
import asyncio
import threading
import uuid
from towhee.hub.file_manager import FileManagerConfig

from towhee.utils.log import engine_log
from towhee.functional.option import Option, Empty, _Reason
from towhee.functional.mixins.parallel import EOS


def _map_task_ray(unary_op): # pragma: no cover
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


class RayMixin: # pragma: no cover
    #pylint: disable=import-outside-toplevel
    """
    Mixin for parallel ray execution.
    """

    def ray_start(self, address = None, local_packages: list = None, pip_packages: list = None, silence = True):
        """
        Start the ray service. When using a remote cluster, all dependencies for custom functions
        and operators defined locally will need to be sent to the ray cluster. If using ray locally,
        within the runtime, avoid passing in any arguments.

        Args:
            address (str):
                The address for the ray service being connected to. If using ray cluster
                remotely with kubectl forwarded port, the most likely address will be "ray://localhost:10001".

            local_packages (list[str]):
                Whichever locally defined modules that are used within a custom function supplied to the pipeline,
                whether it be in lambda functions, locally registered operators, or functions themselves.

            pip_packages (list[str]):
                Whichever pip installed modules that are used within a custom function supplied to the pipeline,
                whether it be in lambda functions, locally registered operators, or functions themselves.
        """
        import ray

        local_packages = [] if local_packages is None else local_packages
        pip_packages = [] if pip_packages is None else pip_packages

        if ('towhee' not in pip_packages and 'towhee' not in [str(x.__name__) for x in local_packages]) and (address is not None):
            pip_packages.append('towhee')
        runtime_env={'py_modules': local_packages, 'pip': pip_packages }

        ray.init(address = address, runtime_env = runtime_env, ignore_reinit_error=True, log_to_driver = silence)
        self._backend_started = True
        return self

    def ray_resolve(self, call_mapping, path, index, *arg, **kws):
        import ray

        # if self.get_backend_started() is None:
        #     self.ray_start()

        #TODO: Make local functions work with ray
        if path in call_mapping:
            return self.map(call_mapping[path](*arg, **kws))

        @ray.remote
        class OperatorActor:
            """Ray actor that runs hub operators."""

            def __init__(self, path1, index1, uid, *arg1, **kws1):
                from towhee import engine
                from towhee.engine.factory import _OperatorLazyWrapper
                from pathlib import Path
                engine.DEFAULT_LOCAL_CACHE_ROOT = Path.home() / ('.towhee/ray_actor_cache_' + uid)
                engine.LOCAL_PIPELINE_CACHE = engine.DEFAULT_LOCAL_CACHE_ROOT / 'pipelines'
                engine.LOCAL_OPERATOR_CACHE = engine.DEFAULT_LOCAL_CACHE_ROOT / 'operators'
                x = FileManagerConfig()
                x.update_default_cache(engine.DEFAULT_LOCAL_CACHE_ROOT)
                self.op = _OperatorLazyWrapper.callback(path1, index1, *arg1, **kws1)

            def __call__(self, *arg1, **kwargs1):
                return self.op(*arg1, **kwargs1)

            def cleanup(self):
                from shutil import rmtree
                from towhee import engine
                try:
                    rmtree(engine.DEFAULT_LOCAL_CACHE_ROOT)
                except FileNotFoundError:
                    pass

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

        child = self._factory(inner())
        return child


    def _ray_pmap(self, unary_op, num_worker=None):
        import ray

        # if self.get_backend_started() is None:
        #     self.ray_start()

        if num_worker is not None:
            pass
        elif self.get_executor() is not None:
            num_worker = self._num_worker
        else:
            num_worker = 2

        # TODO: Dynamic queue size
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
