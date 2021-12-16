import weakref

from towhee.dataframe.dataframe_v2 import DataFrame


class MapIterator:
    """
    A row-based `DataFrame` iterator.
    """
    def __init__(self, df: DataFrame, block = False):
        self._df_ref = weakref.ref(df)
        self._offset = 0
        self._block = block
        self._id = df.register_iter(self)

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns:
            (`Tuple[Any, ...]`)
                In the normal case, the iterator will return a `Tuple` at each call.
            (`None`)
                In the case that the `DataFrame` is not sealed and the new rows are
                not ready yet, the iterator will return `None`. The caller should
                determine whether to block the iteration or exit the loop.
        Raises:
            (`StopIteration`)
                The iteration end iff the `DataFrame` is sealed and the last row is
                reached.
        """
        df = self._df_ref()
        if len(df) == self._offset:
            if df.is_sealed():
                raise StopIteration

        rows = df.get(self._id, self._offset, block=self._block)

        if rows == -1:
            raise StopIteration

        elif rows is None:
            return None

        else:
            self._offset += 1
            df.ack(self._id, self._offset)
            return rows

class BatchIterator:
    """
    A row-based `DataFrame` iterator.
    """
    def __init__(self, df: DataFrame, batch_size = 1, step = 1, block = False):
        self._df_ref = weakref.ref(df)
        self._offset = 0
        self._batch_size = batch_size
        self._step = step
        self._block = block
        self._id = df.register_iter(self)

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns:
            (`Tuple[Any, ...]`)
                In the normal case, the iterator will return a `Tuple` at each call.
            (`None`)
                In the case that the `DataFrame` is not sealed and the new rows are
                not ready yet, the iterator will return `None`. The caller should
                determine whether to block the iteration or exit the loop.
        Raises:
            (`StopIteration`)
                The iteration end iff the `DataFrame` is sealed and the last row is
                reached.
        """
        df = self._df_ref()
        if len(df) == self._offset:
            if df.is_sealed():
                raise StopIteration

        rows = df.get(self._id, self._offset, self._batch_size, block=self._block)

        if rows == -1:
            raise StopIteration

        elif rows is None:
            return None

        else:
            self._offset += self._step
            df.ack(self._id, self._offset)
            return rows
