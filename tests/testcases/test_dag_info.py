# coding : UTF-8
import towhee
import unittest


class TestDagInfo(unittest.TestCase):
    """ Test case of dag info """
    def test_image_decode_clip_dag_information(self):
        """
        Unittest dag info
        """
        dc = towhee.dummy_input() \
                .image_decode['path', 'img']() \
                .towhee.clip['img', 'vec'](model_name='clip_vit_b32', modality='image') \
                .as_function() 
        
        a = dc.dag_info['end']['parent_ids'][0]
        b = {'model_name': 'clip_vit_b32', 'modality': 'image'}

        self.assertEqual(dc.dag_info[a]['init_args'] ,b)

        for key, val in dc.dag_info.items():
            if val['op'] == 'stream':
                self.assertEqual(val['op_name'], 'dummy_input')
            if val['parent_ids'] == []:
                self.assertEqual(key, 'start')

if __name__ == '__main__':
    unittest.main()
