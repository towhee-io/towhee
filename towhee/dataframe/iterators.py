import threading
import weakref

from towhee.dataframe.dataframe_v2 import DataFrame, Responses

class BaseIterator:
    """
    The Base Iterator
    """
    def __init__(self, df: DataFrame, batch_size = 1, step = 1, block = False):
        self._df_ref = weakref.ref(df)
        self._offset = 0
        self._block = block
        self._id = df.register_iter()
        self._done = False
        self._event = threading.Event()
        self._batch_size = batch_size
        self._step = step

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
        if self._done:
            raise StopIteration

        df = self._df_ref()

        code, row = df.get(self._offset, count = self._batch_size, iter_id = self._id)

        if code == Responses.INDEX_GC:
            raise IndexError

        elif code == Responses.INDEX_OOB_UNSEALED:
            if self._block:
                df.notify_map_block(self._id, self._event, self._offset, self._batch_size)
                self._event.wait()
                self._event.clear()
                return self.__next__()

            return None

        elif code == Responses.APPROVED_CONTINUE:
            # subtract one due to step.
            df.ack(self._id, self._offset + self._step - 1)
            self._offset += self._step
            return row

        elif code == Responses.INDEX_OOB_SEALED:
            raise StopIteration

        elif code == Responses.APPROVED_DONE:
            self._done = True
            df.ack(self._id, float('inf'))
            self._offset = 0
            return row

        elif code == Responses.KILLED:
            raise StopIteration

        else: # 'unkown_error'
            raise Exception

    @property
    def id(self):
        return self._id


class MapIterator(BaseIterator):
    """
    A row-based map `DataFrame` iterator.
    """
    def __init__(self, df: DataFrame, block = True):
        super().__init__(df, batch_size = 1, step = 1, block = block)


class BatchIterator(BaseIterator):
    """
    A row-based batch `DataFrame` iterator.
    """
    def __init__(self, df: DataFrame, batch_size = 1, step = 1, block = True):
        super().__init__(df, batch_size = batch_size, step = step, block = block)


class WindowIterator(BaseIterator):
    """
    A row-based window `DataFrame` iterator.
    """
    def __init__(self, df: DataFrame, window_size = 1, block = True, use_timestamp = False):
        self._window_size = window_size
        self._current_window = window_size
        if use_timestamp:
            self._comparator = 'timestamp'
        else:
            self._comparator = 'row_id'
        super().__init__(df, block = block)

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
        if self._done:
            raise StopIteration

        df = self._df_ref()
        cutoff = (self._comparator, self._current_window)
        code, rows = df.get_window(offset = self._offset, cutoff = cutoff, iter_id = self._id)

        if code == Responses.INDEX_GC:
            raise IndexError

        elif code == Responses.WINDOW_NOT_DONE:
            if self._block:
                df.notify_window_block(self._id, self._event, (self._comparator, self._current_window))
                self._event.wait()
                self._event.clear()
                x = self.__next__()
                return x

            # line if window doesnt need to wait for all if not blocking
            # return rows if len(rows) > 0 else None
            return None

        elif code == Responses.APPROVED_CONTINUE:
            # subtract one due to step.
            df.ack(self._id, self._offset + len(rows) - 1)
            self._offset += len(rows)
            self._current_window += self._window_size
            return rows

        elif code == Responses.APPROVED_DONE:
            self._done = True
            df.ack(self._id, float('inf'))
            self._offset = 0
            return rows

        elif code == Responses.KILLED:
            raise StopIteration

        else: # 'unkown_error'
            raise Exception
