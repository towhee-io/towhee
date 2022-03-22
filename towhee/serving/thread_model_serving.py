from typing import List, Union
import threading
import copy
import queue
import weakref
import time

from towhee.serving.torch_model_worker import TorchModelWorker

class ThreadModelServing:
    """
    ThreadModelServing
    """
    def __init__(self, model, batch_size: int, max_latency: int, device_ids: List[int]):
        self.models = [TorchModelWorker(copy.deepcopy(model), idx) for idx in device_ids]
        self._max_latency = max_latency
        self._batch_size = batch_size
        self._input_queue = queue.Queue()

        self.future_cache = TaskFutureCache()
        self._lock = threading.Lock()
        self._task_id = 0
        self._need_stop = False

    @property
    def task_id(self):
        with self._lock:
            self._task_id += 1
        return self._task_id

    def recv(self, data: Union['tensor', List['tensor']]) -> 'Future':
        if not isinstance(data, list):
            data = [data]
        f = TaskFuture(self.task_id, len(data), weakref.ref(self.future_cache))

        self.future_cache[f.task_id] = f

        for idx, item in enumerate(data):
            self._input_queue.put((f.task_id, idx, item))
        return f

    def _response(self, outputs):
        pass

    def _run_once(self, replica_id, batch_data: List):
        batch_data_input = [item[2] for item in batch_data]
        batch_data_output = self.models[replica_id](batch_data_input)
        for i in range(len(batch_data_output)):
            task_id, idx, _ = batch_data[i]
            self.future_cache[task_id].set_result(idx, batch_data_output[i])

    def get_batch_data(self):
        wait_time = self._max_latency
        batch_cache = []
        while True:
            time_start = time.time()
            try:
                if len(batch_cache) == 0:
                    data = self._input_queue.get()
                else:
                    data = self._input_queue.get(timeout=wait_time)
            except queue.Empty:
                return batch_cache
            time_end = time.time()
            wait_time = wait_time - (time_end - time_start)

            if data is None:
                #when return None, the Serving is stopped and send a None to notify other workers.
                self._input_queue.put(None)
                return None
            batch_cache.append(data)
            if len(batch_cache) == self._batch_size:
                return batch_cache

            if wait_time <= 0:
                if len(batch_cache) > 0:
                    return  batch_cache
                else:
                    wait_time = self._max_latency

    def _loop(self, replica_id):
        while not self._need_stop:
            batch_data = self.get_batch_data()
            self._run_once(replica_id, batch_data)

    def start(self):
        num_workers = len(self.models)
        workers = []
        for replica_id in range(num_workers):
            tworker = threading.Thread(target=self._loop, args=(replica_id,), daemon=True)
            tworker.start()
            workers.append(tworker)
        for worker in workers:
            worker.join()

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
