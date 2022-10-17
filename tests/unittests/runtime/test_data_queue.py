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

from towhee.runtime.data_queue import DataQueue, ColumnType


class TestDataQueue(unittest.TestCase):
    '''
    DataQueue test
    '''

    def test_max_size(self):
        max_size = 5
        que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)], max_size)

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

    def test_normal(self):
        que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE), ('vec', ColumnType.QUEUE)])
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

    def test_in_map(self):
        '''
        url, image -> url, image, vec

        input_schema: image
        output_schema: vec
        '''
        input_que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])
        output_que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE), ('vec', ColumnType.QUEUE)])

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


    def test_in_flat_map(self):
        '''
        url -> url, image

        input_schema: url
        output_schema: image
        '''
        input_que = DataQueue([('url', ColumnType.SCALAR)])
        input_que.put(('http://towhee.io', ))
        input_que.seal()

        output_que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])

        size = 20
        ret = input_que.get()
        url = ret[0]
        for i in range(size):
            image = 'image' + str(i)
            output_que.put((url, image))
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

    def test_in_window(self):
        '''
        url, image -> url, image, vec

        input_schema: image
        output_schema: vec
        '''
        input_que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])
        output_que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE), ('vec', ColumnType.QUEUE)])

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
                self.assertEqual(ret[2], None)
            i += 1
        t.join()


    def test_in_filter(self):
        '''
        url, image -> url, image, filter_image

        input_schema: image
        output_schema: filter_image
        filter_columns: image
        '''

        input_que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])
        output_que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE), ('filter_image', ColumnType.QUEUE)])

        size = 20
        def write():
            for i in range(size):
                input_que.put(('http://towhee.io', 'image' + str(i)))
                time.sleep(0.01)
            input_que.seal()


        t = threading.Thread(target=write)
        t.start()

        while True:
            ret = input_que.get()
            if ret is None:
                output_que.seal()
                break
            i = int(ret[1][len('image'):])
            if i % 2 == 0:
                continue

            url, image = ret
            filter_image = 'filter_image' + str(i)
            output_que.put((url, image, filter_image))

        self.assertEqual(output_que.size, size / 2)
        while True:
            ret = output_que.get()
            if ret is None:
                break

            self.assertEqual(int(ret[1][len('image'):]), int(ret[2][len('filter_image'): ]))
        t.join()

    def test_seal(self):
        input_que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])
        self.assertTrue(input_que.put(('1', '2')))
        input_que.seal()
        self.assertFalse(input_que.put(('1', '2')))

        input_que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])
        self.assertTrue(input_que.put(('1', '2')))
        self.assertTrue(input_que.put(('1', '2')))
        self.assertEqual(input_que.size, 2)
        input_que.clear_and_seal()
        self.assertFalse(input_que.put(('1', '2')))
        self.assertEqual(input_que.size, 0)
        self.assertEqual(input_que.get(), None)
        self.assertEqual(input_que.get_dict(), None)

    def test_get_dict(self):
        input_que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])
        self.assertTrue(input_que.put(('1', '2')))
        self.assertEqual(input_que.size, 1)
        data = input_que.get_dict()
        self.assertEqual(data, {
            'url': '1',
            'image': '2'
        })
        self.assertEqual(input_que.size, 0)

    def test_put_dict(self):
        input_que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.QUEUE)])
        self.assertTrue(input_que.put_dict({
            'url': '1',
            'image': 1,
            'other': 1
        }))
        self.assertEqual(input_que.size, 1)

    def test_batch_put_dict(self):
        input_que = DataQueue([('url', ColumnType.SCALAR),
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

        input_que = DataQueue([('url', ColumnType.SCALAR),
                               ('vec', ColumnType.QUEUE)])
        self.assertTrue(input_que.batch_put_dict({
            'url': ['1'],
            'image': [1, 2, 3, 4],
            'vec': [1]
        }))
        self.assertEqual(input_que.size, 1)
        input_que.seal()
        self.assertEqual(input_que.size, 1)        

    def test_all_scalar(self):
        input_que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.SCALAR)])
        input_que.put_dict({'url': 1})
        self.assertEqual(input_que.size, 0)
        input_que.put_dict({'image': 1})
        self.assertEqual(input_que.size, 1)
        self.assertEqual(input_que.get(), [1, 1])
        self.assertEqual(input_que.size, 0)
        input_que.seal()
        self.assertEqual(input_que.size, 0)


        input_que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.SCALAR)])
        input_que.put_dict({'url': 1})
        self.assertEqual(input_que.size, 0)
        input_que.put_dict({'image': 1})
        self.assertEqual(input_que.size, 1)
        input_que.seal()
        self.assertEqual(input_que.size, 1)
        self.assertEqual(input_que.get(), [1, 1])
        self.assertEqual(input_que.size, 0)

        input_que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.SCALAR)])
        input_que.put_dict({'url': 1})
        self.assertEqual(input_que.size, 0)
        input_que.seal()
        self.assertEqual(input_que.size, 0)
        self.assertEqual(input_que.get(), None)
        self.assertEqual(input_que.size, 0)

    def test_all_scalar_multithread(self):
        input_que = DataQueue([('url', ColumnType.SCALAR), ('image', ColumnType.SCALAR)])

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
