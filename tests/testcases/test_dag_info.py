# coding : UTF-8
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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import towhee
import unittest


class TestDagInfo(unittest.TestCase):
    """ Test case of dag info """
    def test_image_decode_clip_dag_information(self):
        """
        Unittest dag info

        example of dag info
        {
            '89be9ec1': {
                'op': 'map',
                'op_name': 'towhee/image-decode',
                'is_stream': True,
                'init_args': {},
                'call_args': {
                    '*arg': ( < towhee.engine.factory._OperatorLazyWrapper object at 0x13faba040 > , ),
                    '*kws': {}
                },
                'op_config': {
                    'chunksize': None,
                    'jit': None,
                    'format_priority': ['onnx'],
                    'ac': '123',
                    'asd': 'wea'
                },
                'input_info': [
                    []
                ],
                'output_info': ['img'],
                'parent_ids': [],
                'child_ids': ['end'],
                'dc_sequence': ['start', '89be9ec1', 'end']
            },
            'end': {
                'op': 'end',
                'op_name': 'end',
                'init_args': None,
                'call_args': None,
                'op_config': None,
                'input_info': None,
                'output_info': None,
                'parent_ids': ['89be9ec1'],
                'child_ids': []
            },
            'start': {
                'op': 'stream',
                'op_name': 'dummy_input',
                'is_stream': False,
                'init_args': None,
                'call_args': {
                    '*arg': (),
                    '*kws': {}
                },
                'op_config': {
                    'parallel': None,
                    'chunksize': None,
                    'jit': None,
                    'format_priority': None
                },
                'input_info': None,
                'output_info': None,
                'parent_ids': [],
                'child_ids': [],
                'dc_sequence': ['start', '89be9ec1', 'end']
            }
        }
        """
        dc = towhee.dummy_input() \
                .image_decode['path', 'img'](op_config={'ac':'123', 'asd':'wea','format_priority':['onnx']}) \
                .set_jit('numba')\
                .as_function() 
        
        a = dc.dag_info['end']['parent_ids'][0]
        self.assertEqual(dc.dag_info[a]['op_name'], 'towhee/image-decode')

        for key, val in dc.dag_info.items():
            if val['op'] == 'stream':
                self.assertEqual(val['op_name'], 'dummy_input')
            if key == 'start':
                expect = 3
                self.assertEqual(len(val['dc_sequence']), expect)

        expect_no_config = {'parallel': None, 'chunksize': None, 'jit': None, 'format_priority': None}
        for i in dc.dag_info.values():
            if i['op_name'] == 'towhee/image-decode':
                
                self.assertEqual(i['op_config']['format_priority'], ['onnx'])
                self.assertEqual(i['op_config']['ac'], '123')
                self.assertEqual(i['op_config']['asd'], 'wea')
            elif i['op_name'] == 'dummy_input':
                self.assertEqual(i['op_config'], expect_no_config)
            elif i['op_name'] == 'end':
                self.assertEqual(i['op_config'], None)
            else:
                self.assertEqual(i['op_config']['format_priority'], None)

        expect_input = 'path'
        expect_output = ['img']
        for j in dc.dag_info.values():
            if j['op_name'] == 'towhee/clip':
                self.assertEqual(j['input_info'][0][1], expect_input)
                self.assertEqual(j['output_info'], expect_output)
                
    def test_schema_dag(self):
        """
        Unittest schema dag info
        """
        f = towhee.dummy_input[(),('video_url', 'video_id')]()\
            .set_parallel(5).video_decode.ffmpeg[('video_url'), ('video_frame')]()\
            .image_text_embedding.clip_image[('video_frame'), ('embedding')]()\
            .image_decode.test[('video_id', 'video_frame', 'embedding'),()]()\
            .as_function()
        for i in f.dag_info.values():
            if i['op_name'] == 'start':
                expect_input = []
                expect_output = ['video_url', 'video_id']
                self.assertEqual(i['input_info'], expect_input)
                self.assertEqual(i['output_info'], expect_output)
            elif i['op_name'] == 'video-decode/ffmpeg':
                expect_input = [('start', 'video_url')]
                expect_output = ['video_frame']
                self.assertEqual(i['input_info'], expect_input)
                self.assertEqual(i['output_info'], expect_output)
            elif i['op_name'] == 'image-text-embedding/clip-image':
                expect_input = 'video_frame'
                expect_output = ['embedding']
                self.assertEqual(i['input_info'][0][1], expect_input)
                self.assertEqual(i['output_info'], expect_output)
            elif i['op_name'] == 'image-decode/test':
                expect_input_1 = 'video_id'
                expect_input_2 = 'video_frame'
                expect_input_3 = 'embedding'
                expect_output = []
                self.assertEqual(i['input_info'][0][1], expect_input_1)
                self.assertEqual(i['input_info'][1][1], expect_input_2)
                self.assertEqual(i['input_info'][2][1], expect_input_3)
                self.assertEqual(i['output_info'], expect_output)
            elif i['op_name'] == 'end':
                self.assertEqual(i['input_info'], None)
                self.assertEqual(i['output_info'], None)

if __name__ == '__main__':
    unittest.main()
