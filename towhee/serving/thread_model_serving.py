from typing import List, Union
import importlib
import threading
import copy
import queue
import weakref
import time
from pathlib import Path


class ThreadModelServing:
    """
    ThreadModelServing
    """

    def __init__(self, model, batch_size: int, max_latency: int, device_ids: List[int], handler=None):
        assert batch_size > 0 and max_latency > 0 and len(device_ids) > 0
        self._model = model
        self._max_latency = max_latency
        self._batch_size = batch_size
        self._input_queue = queue.Queue()

        self.future_cache = _TaskFutureCache()
        self._lock = threading.Lock()
        self._task_id = 0
        self._tworkers = []
        self._need_stop = False

        self._handler_class = self._load_handler(handler)
        self.models = self._create_model_workers(device_ids)

    def _load_handler_file(self, handler):
        handler_path = Path(handler)
        if not handler_path.exists():
            module, class_name = self._load_internal_handler(handler)
        else:
            module_name = handler.stem
            module_spec = importlib.util.spec_from_file_location(module_name, handler)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            class_name = ''.join([item.capitalize() for item in module_name.split('_')])

        if hasattr(module, class_name):
            return getattr(module, class_name)
        else:
            raise RuntimeError('Load model handler %s.%s error' % (handler, class_name))

    def _load_internal_handler(self, handler):
        module_name = '.{0}'.format(handler)
        try:
            model = importlib.import_module(module_name, 'towhee.operator')
            return model, ''.join([item.capitalize() for item in handler.split('_')])
        except ImportError:
            RuntimeError('Can not find model handler: %s' % handler)

    def _load_handler(self, handler):
        if handler is None:
            from towhee.serving.torch_model_handler import TorchModelHandler  # pylint: disable=import-outside-toplevel
            return TorchModelHandler
        return self._load_handler_file(handler)

    def _create_model_workers(self, device_ids: List[int]):
        if len(device_ids) == 1:
            return [self._handler_class(self._model, device_ids[0])]

        models = []
        for d_id in device_ids:
            if d_id == -1:
                models.append(self._handler_class(self._model, -1))
            else:
                models.append(self._handler_class(copy.deepcopy(self._model), d_id))
        return models

    @ property
    def model(self):
        return self._model

    @ property
    def _auto_task_id(self):
        with self._lock:
            self._task_id += 1
        return self._task_id

    def _recv(self, data: Union['tensor', List['tensor']]) -> 'Future':
        if self._need_stop:
            raise RuntimeError('Serving is stopped')

        if not isinstance(data, list):
            data = [data]

        f = _TaskFuture(self._auto_task_id, len(data), weakref.ref(self.future_cache))

        self.future_cache[f.task_id] = f

        for idx, item in enumerate(data):
            self._input_queue.put(_Task(f.task_id, idx, item))
        return f

    def __call__(self, data: Union['tensor', List['tensor']]) -> Union['tensor', List['tensor']]:
        f = self._recv(data)
        return f.result()

    def get_batch_data(self):
        wait_time = self._max_latency
        batch_cache = []
        while len(batch_cache) != self._batch_size and wait_time > 0:
            time_start = time.time()
            try:
                if batch_cache:
                    data = self._input_queue.get(timeout=wait_time)
                else:
                    data = self._input_queue.get()
            except queue.Empty:
                return batch_cache

            if data is None:
                # when return None, the serving is stopped and send a None to notify other workers.
                self._input_queue.put(None)
                return None

            batch_cache.append(data)
            wait_time = wait_time - (time.time() - time_start)

        return batch_cache

    def _loop(self, model):
        while not self._need_stop:
            batch_data = self.get_batch_data()
            if batch_data is None:
                break

            batch_data_input = [item.content for item in batch_data]
            batch_data_output = model(batch_data_input)
            for i in range(len(batch_data_output)):
                self.future_cache[batch_data[i].task_id].set_result(batch_data[i].sub_id, batch_data_output[i])

    def start(self):
        assert not self._need_stop, 'Serving only can starts once'

        num_workers = len(self.models)
        for replica_id in range(num_workers):
            tworker = threading.Thread(target=self._loop, args=(self.models[replica_id],), daemon=True)
            tworker.start()
            self._tworkers.append(tworker)

    def join(self):
        for t in self._tworkers:
            t.join()

    def set_stop(self):
        self._need_stop = True
        self._input_queue.put(None)

    def stop(self):
        if self._need_stop:
            return
        self.set_stop()
        self.join()

    def proxy(self):
        return ThreadServingProxy(self)

    def __del__(self):
        self.stop()


class ThreadServingProxy:
    def __init__(self, model_serving: ThreadModelServing):
        self._model_serving = model_serving

    @ property
    def model(self):
        return self._model_serving.model

    def __call__(self, data: Union['tensor', List['tensor']]):
        return self._model_serving(data)


class _Task:
    def __init__(self, task_id: int, sub_id: int, content: any):
        self.task_id = task_id
        self.sub_id = sub_id
        self.content = content


class _TaskFutureCache(dict):
    pass


class _TaskFuture(object):
    """
    _TaskFuture
    """

    def __init__(self, task_id, size, future_cache_ref):
        assert size > 0
        self.task_id = task_id
        self._outputs = []
        self.size = size
        self._future_cache_ref = future_cache_ref
        self._finish_event = threading.Event()
        self._batch_result = None

    def result(self, timeout=None):
        if self._batch_result is not None:
            return self._batch_result

        finished = self._finish_event.wait(timeout)

        if not finished:
            raise TimeoutError('Result timetout')

        future_cache = self._future_cache_ref()

        if finished and future_cache is not None:
            del future_cache[self.task_id]

        self._outputs.sort(key=lambda i: i[0])
        self._batch_result = [item[1] for item in self._outputs]
        if len(self._batch_result) == 1:
            return self._batch_result[0]
        return self._batch_result

    def set_result(self, index, result):
        self._outputs.append((index, result))
        if len(self._outputs) >= self.size:
            self._finish_event.set()

    def done(self):
        if self._finish_event.is_set():
            return True
