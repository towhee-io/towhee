import weakref
import time

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
        self._done = False

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

        code, row = df.get(self._offset, count = 1, iter_id = self._id)

        if code == 'Index_GC':
            raise IndexError

        elif code == 'Not_Available_Unsealed':
            if self._block:
                while code == 'Not_Available_Unsealed':
                    time.sleep(1)
                    code, row = df.get(self._offset, count = 1, iter_id = self._id)

                if code == 'Killed':
                    raise StopIteration

                else:
                    df.ack(self._id, self._offset)
                    self._offset += 1
                    return row

            return None

        elif code == 'Approved_Continue':
            df.ack(self._id, self._offset)
            self._offset += 1
            return row

        elif code == 'Index_OOB_Sealed':
            raise StopIteration

        elif code == 'Approved_Done':
            self._done = True
            return row

        else: # 'unkown_error'
            raise Exception


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
        self._done = False

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

        if code == 'Index_GC':
            raise IndexError

        elif code == 'Not_Available_Unsealed':
            if self._block:
                while code == 'Not_Available_Unsealed':
                    time.sleep(1)
                    code, row = df.get(self._offset, count = self._batch_size, iter_id = self._id)

                if code == 'Killed':
                    raise StopIteration

                else:
                    df.ack(self._id, self._offset)
                    self._offset += self._step
                    return row

            return None

        elif code == 'Approved_Continue':
            df.ack(self._id, self._offset)
            self._offset += self._step
            return row

        elif code == 'Index_OOB_Sealed':
            raise StopIteration

        elif code == 'Approved_Done':
            self._done = True
            return row

        else: # 'unkown_error'
            raise Exception
