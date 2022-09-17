class Iterator:
    def __init__(self):
        """Create an iterator.
        
        Iterators are what provides the op the values to act on. There can be many types:
        1. batch: can provide map, batch, rolling window, etc based on step and batch args
        2. Generator
        3. more can be added later

        THe iterator iterates on one or many towhee Arrays (one or many schema columns).
        The idea for now is to do a yield based iterator, but that may require some extra work.

        Raises:
            NotImplemented: _description_
        """
        raise NotImplementedError


class MapIterator(Iterator):
    def __init__(self):
        raise NotImplementedError

class FlatMapIterator(Iterator):
    def __init__(self):
        raise NotImplementedError

class FilterIterator(Iterator):
    def __init__(self):
        raise NotImplementedError

class BatchIterator(Iterator):
    def __init__(self):
        raise NotImplementedError
class WindowIterator(Iterator):
    def __init__(self):
        raise NotImplementedError
    