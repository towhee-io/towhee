from typing import List, Union
import threading
import copy
import queue

from towhee.serving.model_serving import ModelServing
from towhee.serving.torch_model_worker import TorchModelWorker

class ThreadModelServing(ModelServing):
    """
    ThreadModelServing
    """
    def __init__(self, model, batch_size: int, max_latency: int, device_ids: List[int]):
        self.models = [TorchModelWorker(copy.deepcopy(model, id)) for id in range(device_ids)]
        self._max_latency = max_latency
        self._batch_size = batch_size
        self._input_queue = queue.Queue()

    def recv(self, data: Union['tensor', List['tensor']]) -> 'Future':
        if not isinstance(data, list):
            data = [data]
        #TODO: workaround
        return data

    def _response(self, outputs):
        pass

    def _run_once(self):
        inputs = self._input_queue.get(self._batch_size, self._max_latency)
        if not inputs:
            return

        outputs = self._model.stack(inputs)
        self._response(outputs)

    def _loop(self):
        while not self._need_stop and not self._input_queue.empty():
            self._run_once()

    def start(self):
        self._loop()

    def stop(self):
        self._need_stop = True

class ThreadProxy():
    def __init__(self):
        pass

    def __call__(self, data: Union['tensor', List['tensor']]):
        pass

class TaskFutureCache(dict):
    pass

class TaskFuture(object):
    """
    TaskFuture
    """
    def __init__(self, task_id, size, future_cache_ref):
        self.task_id = task_id
        self._outputs = []
        assert size > 0
        self.size = size
        self._future_cache_ref = future_cache_ref
        self._finish_event = threading.Event()
        self._batch_result = None

    def result(self, timeout=None):
        if self._batch_result is not None:
            return self._batch_result

        finished = self._finish_event.wait(timeout)

        if not finished:
            raise TimeoutError('TaskFuture result timetout')

        future_cache = self._future_cache_ref()

        if finished and future_cache is not None:
            del future_cache[self.task_id]

        self._outputs.sort(key=lambda i: i[0])
        self._batch_result = [i[1] for i in self._outputs]
        return self._batch_result

    def set_result(self, index, result):
        self._outputs.append((index, result))
        if len(self._outputs) >= self.size:
            self._finish_event.set()

    def done(self):
        if self._finish_event.is_set():
            return True
