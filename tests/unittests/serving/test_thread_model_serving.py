import unittest
import time
import weakref
import threading
import queue

from towhee.serving.thread_model_serving import TaskFuture, TaskFutureCache

class ThreadModelServing(unittest.TestCase):
    """
    ThreadModelServing
    """
    def test_single_taskfuture(self):
        tasks = ['hello','world']

        cache = TaskFutureCache()
        time_start = time.time()
        fut = TaskFuture( 0, 2, weakref.ref(cache))
        cache[0] = fut

        try:
            fut.result(1)
        except TimeoutError:
            pass
        time_end = time.time()
        self.assertGreater(time_end - time_start, 1)
        self.assertEqual(fut.done(), None)

        fut.set_result(1, tasks[1])
        try:
            fut.result(0.1)
        except TimeoutError:
            pass
        self.assertEqual(fut.done(), None)

        fut.set_result(0, tasks[0])
        self.assertEqual(fut.result(0.1), ['hello', 'world'])
        self.assertEqual(fut.result(0.1), ['hello', 'world'])

    def test_mult_taskfuture(self):
        cache = TaskFutureCache()
        q = queue.Queue()
        future_list = []

        def producer():
            for task_id in range(1, 10):
                fut = TaskFuture(task_id, task_id, weakref.ref(cache))
                cache[task_id] = fut
                future_list.append(fut)
                for index in range(task_id):
                    q.put((task_id, index))
            q.put((None,None))

        def consumer(batch_size):
            batch_data = []

            def process_batch(batch_data):
                return [2 * data[0] + data[1] for data in batch_data]

            def handle_output(batch_data) :
                batch_res = process_batch(batch_data)
                for i in range(len(batch_res)):
                    task_id, index = batch_data[i]
                    cache[task_id].set_result(index, batch_res[i])

            while True:
                task_id, index = q.get()
                if task_id is None :
                    if len(batch_data) != 0:
                        handle_output(batch_data)
                        batch_data = []
                    q.put((None,None))
                    return

                batch_data.append((task_id, index))
                if len(batch_data) == batch_size:
                    handle_output(batch_data)
                    batch_data = []

        tproducer = threading.Thread(target=producer)
        for i in range(1,5):
            tconsumer = threading.Thread(target=consumer, args=(i,))
            tconsumer.start()

        tproducer.start()

        tproducer.join()

        for fut in future_list:
            result = fut.result()
            self.assertEqual(result, [ fut.task_id * 2 + i for i in range(fut.task_id)])


if __name__ == '__main__':
    unittest.main()
