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


import unittest
import torch
import numpy

from towhee.models.utils.audio_preprocess import MelSpec, preprocess_wav


class TestAudioPreprocess(unittest.TestCase):
    """
    Test audio preprocess functions.
    """
    audio = torch.rand(2, 200)
    params = {
        'sample_rate': 80,
        'window_length': 12, 'hop_length': 10,
        'f_min': 3, 'f_max': 40, 'n_mels': 10,
        'naf_mode': False, 'mel_log': 'log', 'spec_norm': 'l2',
        'segment_size': 1, 'hop_size': 1, 'frame_shift_mul': 1
    }

    def test_melspec(self):
        mel = MelSpec(
            sample_rate=self.params['sample_rate'],
            window_length=self.params['window_length'],
            hop_length=self.params['hop_length'],
            f_min=self.params['f_min'],
            f_max=self.params['f_max'],
            n_mels=self.params['n_mels'],
            naf_mode=self.params['naf_mode'],
            mel_log=self.params['mel_log'],
            spec_norm=self.params['spec_norm']
        )

        outs = mel(self.audio)
        self.assertTrue(outs.shape == (2, 10, 21))

    def test_naf_log10_max(self):
        self.params.update(naf_mode=True, mel_log='log10', spec_norm='max')
        mel = MelSpec(
            sample_rate=self.params['sample_rate'],
            window_length=self.params['window_length'],
            hop_length=self.params['hop_length'],
            f_min=self.params['f_min'],
            f_max=self.params['f_max'],
            n_mels=self.params['n_mels'],
            naf_mode=self.params['naf_mode'],
            mel_log=self.params['mel_log'],
            spec_norm=self.params['spec_norm']
        )

        outs = mel(self.audio)
        self.assertTrue(outs.shape == (2, 10, 21))

    def test_preprocess_wav(self):
        audio = numpy.ndarray((50, 2))
        outs = preprocess_wav(
            audio,
            segment_size=int(self.params['sample_rate'] * self.params['segment_size']),
            hop_size=int(self.params['sample_rate'] * self.params['hop_size']),
            frame_shift_mul=self.params['frame_shift_mul']
        )
        self.assertTrue(outs.shape == (1, 80))

    def test_wav_input(self):
        audio = torch.rand(200)
        outs = preprocess_wav(
            audio,
            segment_size=int(self.params['sample_rate'] * self.params['segment_size']),
            hop_size=int(self.params['sample_rate'] * self.params['hop_size']),
            frame_shift_mul=self.params['frame_shift_mul']
        )
        self.assertTrue(outs.shape == (2, 80))




if __name__ == '__main__':
    unittest.main()
