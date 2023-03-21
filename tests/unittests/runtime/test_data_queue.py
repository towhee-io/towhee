# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT_ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
import threading
import time
from functools import partial

from towhee.runtime.data_queue import DataQueue, ColumnType, Empty


# pylint: disable=invalid-name
class TestDataQueue(unittest.TestCase):
    '''
    DataQueue test
    '''

    def _internal_max_size(self, DataQueueClass):
        max_size = 5
        que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)], max_size)

        size = 20
        def write():
            for i in range(size):
                que.put(('http://towhee.io', 'image' + str(i)))
            que.seal()

        t = threading.Thread(target=write)
        t.start()
        time.sleep(0.01)
        self.assertEqual(que.size, max_size)
        output = []
        while True:
            ret = que.get()
            if ret is None:
                break
            output.append(ret)
            time.sleep(0.01)
            if len(output) <= size - max_size:
                self.assertEqual(que.size, max_size)
            else:
                self.assertEqual(que.size, size - len(output))
        t.join()
        self.assertTrue(que.sealed)

    def _internal_normal(self, DataQueueClass):
        que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE), ('vec', ColumnType.QUEUE)])
        que.put(('http://towhee.io', 'image1', 'vec1'))
        self.assertEqual(que.size, 1)
        que.put(('http://towhee.io', 'image2', 'vec2'))
        self.assertEqual(que.size, 2)

        self.assertEqual(que._data[0]._data, 'http://towhee.io')  # pylint: disable=protected-access
        ret = que.get()
        self.assertEqual(ret[1], 'image1')
        self.assertEqual(que.size, 1)

        ret = que.get()
        self.assertEqual(ret[1], 'image2')
        self.assertEqual(que.size, 0)

    def _internal_in_map(self, DataQueueClass):
        '''
        url, image -> url, image, vec

        input_schema: image
        output_schema: vec
        '''
        input_que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])
        output_que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE), ('vec', ColumnType.QUEUE)])

        size = 20

        def write():
            for i in range(1, size + 1):
                input_que.put(('http://towhee.io', 'image' + str(i)))
                time.sleep(0.01)
            input_que.seal()

        t = threading.Thread(target=write)
        t.start()

        i = 1
        while True:
            ret = input_que.get()
            if ret is None:
                output_que.seal()
                break
            url, image = ret
            vec = 'vec' + str(i)
            output_que.put((url, image, vec))
            i += 1

        self.assertEqual(output_que.size, size)
        i = 1
        while True:
            ret = output_que.get()
            if ret is None:
                break
            self.assertEqual(ret[0], 'http://towhee.io')
            self.assertEqual(ret[1], 'image' + str(i))
            self.assertEqual(ret[2], 'vec' + str(i))
            i += 1
        t.join()


    def _internal_in_flat_map(self, DataQueueClass):
        '''
        url -> url, image

        input_schema: url
        output_schema: image
        '''
        input_que = DataQueueClass([('url', ColumnType.SCALAR)])
        input_que.put(('http://towhee.io', ))
        input_que.seal()

        output_que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])

        size = 20
        ret = input_que.get()
        url = ret[0]
        output_que.put_dict({'url': url})
        for i in range(size):
            image = 'image' + str(i)
            output_que.put_dict({'image': image})
        output_que.seal()

        self.assertEqual(output_que.size, size)
        i = 0
        while True:
            ret = output_que.get()
            if ret is None:
                break
            self.assertEqual(ret[0], 'http://towhee.io')
            self.assertEqual(ret[1], 'image' + str(i))
            i += 1

    def _internal_in_window(self, DataQueueClass):
        '''
        url, image -> url, image, vec

        input_schema: image
        output_schema: vec
        '''
        input_que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])
        output_que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE), ('vec', ColumnType.QUEUE)])

        size = 20
        def write():
            for i in range(size):
                input_que.put(('http://towhee.io', 'image' + str(i)))
                time.sleep(0.01)
            input_que.seal()

        def new_container():
            items = []
            for _ in range(output_que.col_size):
                items.append([])
            return items

        t = threading.Thread(target=write)
        t.start()

        batch = 3
        i = 0

        items = new_container()
        while True:
            ret = input_que.get()

            if ret is None:
                if not items[0]:
                    break
                else:
                    vecs = ['vec', 'vec']
                    items[-1] = vecs
                    output_que.batch_put(items)
                    items = new_container()
                    break

            for i in range(len(ret)):
                items[i].append(ret[i])

            if len(items[0]) == batch:
                vecs = ['vec', 'vec']
                items[-1] = vecs
                output_que.batch_put(items)
                items = new_container()

        output_que.seal()
        self.assertEqual(output_que.size, size)
        i = 0
        vec_size = size // batch
        if size % batch != 0:
            vec_size += 1
        vec_size = vec_size * 2
        while True:
            ret = output_que.get()
            if ret is None:
                break

            self.assertEqual(ret[0], 'http://towhee.io')
            self.assertEqual(ret[1], 'image' + str(i))
            if i < vec_size:
                self.assertEqual(ret[2], 'vec')
            else:
                self.assertIsInstance(ret[2], Empty)
            i += 1
        t.join()


    def _internal_in_filter(self, DataQueueClass):
        '''
        url, image -> url, image, filter_image

        input_schema: image
        output_schema: filter_image
        filter_columns: image
        '''

        input_que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])
        output_que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE), ('filter_image', ColumnType.QUEUE)])

        size = 20
        def write():
            for i in range(size):
                input_que.put(('http://towhee.io', 'image' + str(i)))
                time.sleep(0.01)
            input_que.seal()


        t1 = threading.Thread(target=write)
        t1.start()

        def filter_process():
            while True:
                ret = input_que.get()
                if ret is None:
                    break
                i = int(ret[1][len('image'):])
                if i % 2 == 0:
                    output_que.put_dict({'url': ret[0], 'image': ret[1]})
                    continue

                url, image = ret
                filter_image = 'filter_image' + str(i)
                output_que.put((url, image, filter_image))

        t2 = threading.Thread(target=filter_process)
        t2.start()

        t1.join()
        t2.join()
        self.assertEqual(output_que.size, size / 2)
        ret = []
        while output_que.size > 0:
            ret.append(output_que.get())

        output_que.seal()
        self.assertEqual(output_que.size, size / 2)
        while output_que.size > 0:
            ret.append(output_que.get())

    def _internal_seal(self, DataQueueClass):
        input_que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])
        self.assertTrue(input_que.put(('1', '2')))
        input_que.seal()
        self.assertFalse(input_que.put(('1', '2')))

        input_que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])
        self.assertTrue(input_que.put(('1', '2')))
        self.assertTrue(input_que.put(('1', '2')))
        self.assertEqual(input_que.size, 2)
        input_que.clear_and_seal()
        self.assertFalse(input_que.put(('1', '2')))
        self.assertEqual(input_que.size, 0)
        self.assertEqual(input_que.get(), None)
        self.assertEqual(input_que.get_dict(), None)

    def _internal_get_dict(self, DataQueueClass):
        input_que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])
        self.assertTrue(input_que.put(('1', '2')))
        self.assertEqual(input_que.size, 1)
        data = input_que.get_dict()
        self.assertEqual(data, {
            'url': '1',
            'image': '2'
        })
        self.assertEqual(input_que.size, 0)

    def _internal_put_dict(self, DataQueueClass):
        input_que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])
        self.assertTrue(input_que.put_dict({
            'url': '1',
            'image': 1,
            'other': 1
        }))
        self.assertEqual(input_que.size, 1)

    def _internal_batch_put_dict(self, DataQueueClass):
        input_que = DataQueueClass([('url', ColumnType.SCALAR),
                               ('image', ColumnType.QUEUE),
                               ('vec', ColumnType.QUEUE)])
        self.assertTrue(input_que.batch_put_dict({
            'url': ['1'],
            'image': [1, 2, 3, 4],
            'vec': [1, 2],
            'other': [1, 3, 9, 8, 8]
        }))
        self.assertEqual(input_que.size, 2)
        input_que.seal()
        self.assertEqual(input_que.size, 4)

        input_que = DataQueueClass([('url', ColumnType.SCALAR),
                               ('vec', ColumnType.QUEUE)])
        self.assertTrue(input_que.batch_put_dict({
            'url': ['1'],
            'image': [1, 2, 3, 4],
            'vec': [1]
        }))
        self.assertEqual(input_que.size, 1)
        input_que.seal()
        self.assertEqual(input_que.size, 1)

    def _internal_all_scalar(self, DataQueueClass):
        input_que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.SCALAR)])
        input_que.put_dict({'url': 1})
        self.assertEqual(input_que.size, 0)
        input_que.put_dict({'image': 1})
        self.assertEqual(input_que.size, 1)
        self.assertEqual(input_que.get(), [1, 1])
        self.assertEqual(input_que.size, 0)
        input_que.seal()
        self.assertEqual(input_que.size, 0)


        input_que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.SCALAR)])
        input_que.put_dict({'url': 1})
        self.assertEqual(input_que.size, 0)
        input_que.put_dict({'image': 1})
        self.assertEqual(input_que.size, 1)
        input_que.seal()
        self.assertEqual(input_que.size, 1)
        self.assertEqual(input_que.get(), [1, 1])
        self.assertEqual(input_que.size, 0)

        input_que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.SCALAR)])
        input_que.put_dict({'url': 1})
        self.assertEqual(input_que.size, 0)
        input_que.seal()
        self.assertEqual(input_que.size, 0)
        self.assertEqual(input_que.get(), None)
        self.assertEqual(input_que.size, 0)

    def _internal_all_scalar_multithread(self, DataQueueClass):
        input_que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.SCALAR)])

        def write():
            input_que.put_dict({'url': 1})
            time.sleep(0.05)
            input_que.put_dict({'image': 2})
            input_que.seal()

        t = threading.Thread(target=write)
        t.start()

        ret = []
        while True:
            data = input_que.get()
            if data is None:
                break
            ret.append(data)
        self.assertEqual(ret, [[1, 2]])
        t.join()

    def _internal_empty(self, DataQueueClass):
        que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.SCALAR)])
        que.put_dict({})
        self.assertEqual(que.size, 0)
        que.seal()
        self.assertEqual(que.size, 0)

        que = DataQueueClass([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])
        que.put_dict({})
        self.assertEqual(que.size, 0)
        que.seal()
        self.assertEqual(que.size, 0)

        que = DataQueueClass([('url', ColumnType.QUEUE), ('image', ColumnType.QUEUE)])
        que.put_dict({})
        self.assertEqual(que.size, 0)
        que.seal()
        self.assertEqual(que.size, 0)

    def _internal_to_list(self, DataQueueClass):
        input_que = DataQueueClass([('url', ColumnType.SCALAR),
                               ('image', ColumnType.QUEUE),
                               ('vec', ColumnType.QUEUE)])
        self.assertTrue(input_que.batch_put_dict({
            'url': ['1'],
            'image': [1, 2, 3, 4],
            'vec': [1, 2],
            'other': [1, 3, 9, 8, 8]
        }))
        with self.assertRaises(RuntimeError):
            input_que.to_list()

        input_que.seal()
        data = input_que.to_list()
        self.assertEqual(4, len(data))
        self.assertEqual(0, input_que.size)
        self.assertEqual(data[0], ['1', 1, 1])
        self.assertEqual(data[1], ['1', 2, 2])
        self.assertEqual(data[2], ['1', 3, Empty()])
        self.assertEqual(data[3], ['1', 4, Empty()])


    def _internal_to_kv(self, DataQueueClass):
        input_que = DataQueueClass([('url', ColumnType.SCALAR),
                               ('image', ColumnType.QUEUE),
                               ('vec', ColumnType.QUEUE)])
        self.assertTrue(input_que.batch_put_dict({
            'url': ['1'],
            'image': [1, 2, 3, 4],
            'vec': [1, 2],
            'other': [1, 3, 9, 8, 8]
        }))
        with self.assertRaises(RuntimeError):
            input_que.to_list()

        input_que.seal()
        data = input_que.to_list(True)
        self.assertEqual(4, len(data))
        self.assertEqual(0, input_que.size)
        self.assertEqual(data[0], {'url': '1', 'image': 1, 'vec': 1})
        self.assertEqual(data[1], {'url': '1', 'image': 2, 'vec': 2})
        self.assertEqual(data[2], {'url': '1', 'image': 3, 'vec': Empty()})
        self.assertEqual(data[3], {'url': '1', 'image': 4, 'vec': Empty()})


    def test_data_queue(self):
        for item in self.__dir__():
            if item.startswith('_internal_'):
                getattr(self, item)(DataQueue)

    def test_debug_data_queue(self):
        for item in self.__dir__():
            if item.startswith('_internal_'):
                getattr(self, item)(partial(DataQueue, keep_data=True))

    def test_debug_data_queue_reset(self):
        que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE), ('vec', ColumnType.QUEUE)], keep_data=True)
        que.put(('http://towhee.io', [1], 'vec1'))
        self.assertEqual(que.size, 1)
        que.put(('http://towhee.io', [2], 'vec2'))
        self.assertEqual(que.size, 2)
        ret = que.get()
        self.assertEqual(ret[1], [1])
        ret[1].append(2)
        self.assertEqual(que.size, 1)
        ret = que.get()
        self.assertEqual(ret[1], [2])
        ret[1].append(3)
        self.assertEqual(que.size, 0)

        que.reset_size()
        ret = que.get()
        self.assertEqual(ret[1], [1])
        self.assertEqual(que.size, 1)
        ret = que.get()
        self.assertEqual(ret[1], [2])
        self.assertEqual(que.size, 0)
