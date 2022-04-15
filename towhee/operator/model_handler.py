from abc import ABC, abstractclassmethod


class HandlerBase(ABC):

    @abstractclassmethod
    def __init__(self, model, device_id):
        pass

    @abstractclassmethod
    def __call__(self, batch_data):
        pass
