import threading
from typing import List, Tuple
import weakref

from towhee.dataframe.dataframe import DataFrame, Responses

class DataFrameIterator:
    """
    Base iterator implementation. All iterators should be subclasses of `DataFrameIterator`.

    Args:
        df (`Dataframe`):
            The dataframe to iterate over.
        block (`bool`):
            Whether to block on call.
    """

    def __init__(self, df: DataFrame, block: bool = True):
        self._df_ref = weakref.ref(df)
        self._df_name = df.name
        self._offset = 0
        self._block = block
        self._id = df.register_iter()
        self._done = False
        self._event = threading.Event()

    def __iter__(self):
        return self

    def __next__(self) -> List[Tuple]:
        # The base iterator is purposely defined to have exatly 0 elements.
        raise StopIteration

    @property
    def df_name(self) -> str:
        """
        Returns the df name.
        """
        return self._df_name

    @property
    def current_index(self) -> int:
        """
        Returns the current index.
        """
        return self._offset

    @property
    def accessible_size(self) -> int:
        """
        Returns current accessible data size.
        """
        return self._df_ref().size - self._offset

    @property
    def id(self) -> int:
        return self._id

    def notify(self):
        """
        Remove unblock and remove iterator.
        """
        if self._done is not True:
            df = self._df_ref()
            df.remove_iter(self._id)
            self._done = True

class MapIterator(DataFrameIterator):
    """
    A row-based map `DataFrame` iterator.

    Args:
        df (`towhee.Dataframe`):
            The dataframe that is being iterated.
        block (`bool`):
            Whether to block when data not present.
    """
    def __init__(self, df: DataFrame, block: bool = True):
        super().__init__(df, block = block)
        self._batch_size = 1
        self._step = 1

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns:
            (`list(tuple([Any, ...])`)
                In the normal case, the iterator will return a list of tuples at each call.
            (`None`)
                In the case that the `DataFrame` is not sealed and the new rows are
                not ready yet, the iterator will return `None`.
        Raises:
            (`StopIteration`)
                The iteration end if the `DataFrame` is sealed and the last row is
                reached.
        """
        df = self._df_ref()

        if self._done:
            raise StopIteration

        code, row, _ = df.get(self._offset, count = self._batch_size, iter_id = self._id)

        if code == Responses.INDEX_GC:
            raise IndexError

        elif code == Responses.INDEX_OOB_UNSEALED:
            if self._block:
                df.notify_map_block(self._event, self._offset, self._batch_size, self._id)
                self._event.wait()
                self._event.clear()
                return self.__next__()

            return None

        elif code == Responses.APPROVED_CONTINUE:
            df.ack(self._offset + self._step, self._id)

            self._offset += self._step
            return row

        elif code == Responses.INDEX_OOB_SEALED:
            df.remove_iter(self._id)
            self._offset = 0
            self._done = True
            raise StopIteration

        elif code == Responses.KILLED:
            raise StopIteration

        else: # 'unkown_error'
            raise Exception


class BatchIterator(MapIterator):
    """
    A row-based batched map `DataFrame` iterator.

    Args:
        df (`towhee.Dataframe`):
            The dataframe that is being iterated.
        batch_size (`int`):
            How many values to read.
        step (`int`):
            How many steps to take after each read.
        block (`bool`):
            Whether to block when data not present.
    """
    def __init__(self, df: DataFrame, batch_size: int = 1, step: int = 1, block: bool = True):
        super().__init__(df, block = block)
        self._batch_size = batch_size
        self._step = step


class WindowIterator(DataFrameIterator):
    """
    A row-based window `DataFrame` iterator.

    Args:
        df (`towhee.Dataframe`):
            The dataframe that is being iterated.
        start (`int`):
            Where to start the window from.
        window_size (`int`):
            How large of a window.
        step (`int`):
            How far to iterate window per read.
        use_timestamp (`bool`):
            Whether to use timestamp instead of row_id.
        block (`bool`):
            Whether to block when data not present.
    """
    def __init__(self, df: DataFrame, start: int = 0, window_size: int = 1, step: int = None, use_timestamp: bool = False, block: bool = True):
        super().__init__(df, block = block)

        if use_timestamp:
            start *= 1000
            window_size *= 1000
            if step is not None:
                step *= 1000

        self._window_size = window_size
        self._current_window = (start, (start + window_size))
        self._step = step

        if step is None:
            self._step = window_size
        if use_timestamp:
            self._comparator = 'timestamp'
        else:
            self._comparator = 'row_id'


    def __next__(self):
        """
        Returns:
            (`list(Tuple[Any, ...])`)
                In the normal case, the iterator will return a list of `tuple` at each call.
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

        if self._done:
            df.remove_iter(self._id)
            raise StopIteration

        code, rows, offset = df.get_window(self._current_window[0], self._current_window[1], self._step, self._comparator, self._id)
        if code in (Responses.EMPTY_SEALED, Responses.FUTURE_WINDOW_SEALED):
            self._done = True
            df.remove_iter(self._id)
            raise StopIteration

        elif code == Responses.EMPTY:
            df.notify_window_block(self._event, 'start', (self._comparator, self._current_window[0]), self._id)
            self._event.wait()
            self._event.clear()
            return self.__next__()

        elif code == Responses.WINDOW_NOT_DONE:
            df.notify_window_block(self._event, 'end', (self._comparator, self._current_window[1]), self._id)
            self._event.wait()
            self._event.clear()
            return self.__next__()

        elif code == Responses.FUTURE_WINDOW:
            df.ack(offset, self._id) #TODO: Garbage collection logic if waiting on window that is far.
            df.notify_window_block(self._event, 'start', (self._comparator, self._current_window[0]), self._id)
            self._event.wait()
            self._event.clear()
            return self.__next__()

        elif code == Responses.OLD_WINDOW:
            self._current_window = offset
            return self.__next__()

        elif code == Responses.APPROVED_DONE:
            self._done = True
            return rows

        elif code == Responses.APPROVED_CONTINUE:
            self._current_window = (self._current_window[0] + self._step, self._current_window[1] + self._step)
            df.ack(offset, self._id)
            return rows

        elif code == Responses.KILLED:
            self._done = True
            raise StopIteration

        else: # 'unkown_error'
            raise Exception
