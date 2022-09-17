from copy import deepcopy

class DataFrame():
    def __init__(self, name):
        self._name = name
        self._outputs = []

    def add_queue(self, queue):
        self._outputs.append(queue)

    def append(self, value):
        for x in self._outputs:
            x.put(deepcopy(value))
