from abc import ABC, abstractmethod


class HandlerBase(ABC):

    @abstractmethod
    def __init__(self, model, device_id):
        pass

    @abstractmethod
    def __call__(self, batch_data):
        pass
