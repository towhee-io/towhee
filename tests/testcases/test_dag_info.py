# coding : UTF-8
import towhee
import unittest


class TestDagInfo(unittest.TestCase):
    """ Test case of dag info """
    def test_image_decode_clip_dag_information(self):
        """
        Unittest dag info

        example of dag info
        {
            'e4b074eb': {
                'op': 'map',
                'op_name': 'towhee/image-decode',
                'is_stream': True,
                'init_args': {},
                'call_args': {
                    '*arg': ( < towhee.engine.factory._OperatorLazyWrapper object at 0x1216d4f70 > , ),
                    '*kws': {}
                },
                'op_config': None,
                'input_info': [('start', 'path')],
                'output_info': ['img'],
                'parent_ids': ['start'],
                'child_ids': ['db5377c3']
            },
            'db5377c3': {
                'op': 'map',
                'op_name': 'towhee/clip',
                'is_stream': True,
                'init_args': {
                    'model_name': 'clip_vit_b32',
                    'modality': 'image'
                },
                'call_args': {
                    '*arg': ( < towhee.engine.factory._OperatorLazyWrapper object at 0x12171a670 > , ),
                    '*kws': {}
                },
                'op_config': {
                    'ac': '123',
                    'asd': 'wea'
                },
                'input_info': [('e4b074eb', 'img')],
                'output_info': ['vec'],
                'parent_ids': ['e4b074eb'],
                'child_ids': ['end']
            },
            'end': {
                'op': 'end',
                'op_name': 'end',
                'init_args': None,
                'call_args': None,
                'op_config': None,
                'input_info': None,
                'output_info': None,
                'parent_ids': ['db5377c3'],
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
                'op_config': None,
                'input_info': None,
                'output_info': None,
                'parent_ids': [],
                'child_ids': ['e4b074eb']
            }
        }
        """
        dc = towhee.dummy_input() \
                .image_decode['path', 'img']() \
                .set_jit('numba')\
                .towhee.clip['img', 'vec'](model_name='clip_vit_b32', modality='image', op_config={'ac':'123', 'asd':'wea'}) \
                .as_function() 
        
        a = dc.dag_info['end']['parent_ids'][0]
        b = {'model_name': 'clip_vit_b32', 'modality': 'image'}
        self.assertEqual(dc.dag_info[a]['init_args'] ,b)

        for key, val in dc.dag_info.items():
            if val['op'] == 'stream':
                self.assertEqual(val['op_name'], 'dummy_input')
            if val['parent_ids'] == []:
                self.assertEqual(key, 'start')

        expect = {'ac': '123', 'asd': 'wea', 'parallel': None, 'chunksize': None, 'jit': 'numba', 'format_priority': None}
        expect_no_config = {'parallel': None, 'chunksize': None, 'jit': None, 'format_priority': None}
        print(dc.dag_info)
        for i in dc.dag_info.values():
            if i['op_name'] == 'towhee/clip':
                self.assertEqual(i['op_config'], expect)
            elif i['op_name'] == 'dummy_input':
                self.assertEqual(i['op_config'], expect_no_config)
            elif i['op_name'] == 'end':
                self.assertEqual(i['op_config'], None)
            else:
                self.assertEqual(i['op_config']['format_priority'], None)

        expect_input = 'img'
        expect_output = ['vec']
        for j in dc.dag_info.values():
            if j['op_name'] == 'towhee/clip':
                self.assertEqual(j['input_info'][0][1], expect_input)
                self.assertEqual(j['output_info'], expect_output)
                
    def test_op_config(self):
        dc = towhee.dummy_input() \
                .image_decode['path', 'img']() \
                .set_jit('numba')\
                .towhee.clip['img', 'vec'](model_name='clip_vit_b32', modality='image', op_config={'format_priority': ['onnx'],
                                                                                            'ac':'123', 'asd':'wea'}) \
                .as_function()
        expect = {'parallel': None, 'chunksize': None, 'jit': 'numba', 'format_priority': ['onnx'], 'ac': '123', 'asd': 'wea'}
        for i in dc.dag_info.values():
            if i['op_name'] == 'towhee/clip':
                self.assertEqual(i['op_config'], expect)
if __name__ == '__main__':
    unittest.main()
