from abc import  ABC
from typing import Union, List

class ModelServing(ABC):
    """
    ModelServing
    """
    def recv(self, data: Union['tensor', List['tensor']]):
        pass

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

class Proxy:
    def __init__(self):
        pass

    def __call__(self, data: Union['tensor', List['tensor']]):
        pass

