import unittest
import time
import weakref
import threading
import queue
import torch
from torch import nn

from towhee.serving.thread_model_serving import _TaskFuture, _TaskFutureCache, ThreadModelServing


class TestThreadModelServing(unittest.TestCase):
    """
    ThreadModelServing
    """

    def test_single_taskfuture(self):
        tasks = ['hello', 'world']

        cache = _TaskFutureCache()
        time_start = time.time()
        fut = _TaskFuture(0, 2, weakref.ref(cache))
        cache[0] = fut

        try:
            fut.result(1)
        except TimeoutError:
            pass
        time_end = time.time()
        self.assertGreater(time_end - time_start, 0.9)
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
        cache = _TaskFutureCache()
        q = queue.Queue()
        future_list = []

        def producer():
            for task_id in range(1, 10):
                fut = _TaskFuture(task_id, task_id, weakref.ref(cache))
                cache[task_id] = fut
                future_list.append(fut)
                for index in range(task_id):
                    q.put((task_id, index))
            q.put((None, None))

        def consumer(batch_size):
            batch_data = []

            def process_batch(batch_data):
                return [2 * data[0] + data[1] for data in batch_data]

            def handle_output(batch_data):
                batch_res = process_batch(batch_data)
                for i in range(len(batch_res)):
                    task_id, index = batch_data[i]
                    cache[task_id].set_result(index, batch_res[i])

            while True:
                task_id, index = q.get()
                if task_id is None:
                    if len(batch_data) != 0:
                        handle_output(batch_data)
                        batch_data = []
                    q.put((None, None))
                    return

                batch_data.append((task_id, index))
                if len(batch_data) == batch_size:
                    handle_output(batch_data)
                    batch_data = []

        tproducer = threading.Thread(target=producer)
        for i in range(0, 5):
            tconsumer = threading.Thread(target=consumer, args=(i,))
            tconsumer.start()

        tproducer.start()

        tproducer.join()

        for fut in future_list:
            result = fut.result()
            rvalue = [fut.task_id * 2 + i for i in range(fut.task_id)]
            if len(rvalue) == 1:
                rvalue = rvalue[0]
            self.assertEqual(result, rvalue)

    def test_threadmodelserving_cpu(self):
        model = nn.Identity()
        serving = ThreadModelServing(model=model, batch_size=4, max_latency=0.5, device_ids=[-1])

        serving.start()
        batch_data = [torch.randn(3, 3) for _ in range(10)]
        fut = serving._recv(batch_data)  # pylint: disable=protected-access
        fut.result()
        self.assertEqual(torch.stack(fut.result()).sum(), torch.stack(batch_data).sum())

        begin = time.time()
        fut = serving._recv(torch.randn(3, 3))  # pylint: disable=protected-access
        fut.result()
        self.assertGreater(time.time() - begin, 0.48)
        serving.stop()

    def test_threadmodelserving_multithread_cpu(self):
        model = nn.Identity()
        serving = ThreadModelServing(model=model, batch_size=4, max_latency=1, device_ids=[-1, -1, -1, -1])
        serving.start()

        def sending_request(idx):
            batch_sizes = [1, 2, 3, 4]
            bs = batch_sizes[idx]

            batch_data = [torch.randn(3, 3) for _ in range(bs)]
            if len(batch_data) == 1:
                batch_data = batch_data[0]
            fut = serving._recv(batch_data)  # pylint: disable=protected-access
            if idx == 0:
                self.assertEqual(fut.result().sum(), batch_data.sum())
            else:
                self.assertEqual(torch.stack(fut.result()).sum(), torch.stack(batch_data).sum())

        tpools = []
        for i in range(4):
            t = threading.Thread(target=sending_request, args=(i,), daemon=True)
            tpools.append(t)
            t.start()

        for t in tpools:
            t.join()

        serving.stop()

    def test_proxy_multithread(self):
        model = nn.Identity()
        serving = ThreadModelServing(model=model, batch_size=4, max_latency=1, device_ids=[-1, -1, -1, -1])
        serving.start()

        proxy = serving.proxy()

        def sending_request(idx):
            batch_sizes = [1, 2, 3, 4]
            bs = batch_sizes[idx]

            batch_data = [torch.randn(3, 3) for _ in range(bs)]
            if len(batch_data) == 1:
                batch_data = batch_data[0]
            res = proxy(batch_data)
            if idx == 0:
                self.assertEqual(res.sum(), batch_data.sum())
            else:
                self.assertEqual(torch.stack(res).sum(), torch.stack(batch_data).sum())

        tpools = []
        for i in range(4):
            t = threading.Thread(target=sending_request, args=(i,), daemon=True)
            tpools.append(t)
            t.start()

        for t in tpools:
            t.join()

        serving.stop()


if __name__ == '__main__':
    unittest.main()
