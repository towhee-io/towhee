# Inspired by https://github.com/stdio2016/pfann/blob/main/datautil/melspec.py
#
# Modifications by Copyright 2021  Facebook. All rights reserved.
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
# This code is modified by Zilliz.

import numpy
import torch
import torchaudio

from towhee.utils.log import models_log


class MelSpec(torch.nn.Module):
    """
    Convert waveform to Mel Spectrogram

    Args:
        - sample_rate (`int`):
            Sample rate of audio signal.
        - window_length (`int`):
            Window & FFT size.
        - hop_length (`int`):
            Length of hop between STFT windows.
        - f_min (`int`):
            Minimum frequency.
        - f_max (`int`):
            Maximum frequency.
        - n_mels (`int`):
            Number of mel filterbanks.
        - naf_mode (`bool`):
            The flag to control other parameters of mel spectrogram transform
        - mel_log (`str`):
            The torch log type in ['log', 'log10']
        - spec_norm (`str`):
            The normalization type (max normalization for 'max').
    """
    def __init__(self,
                 sample_rate=8000,
                 window_length=1024,
                 hop_length=256,
                 f_min=300,
                 f_max=4000,
                 n_mels=256,
                 naf_mode=False,
                 mel_log='log',
                 spec_norm='l2'):
        super().__init__()
        self.naf_mode = naf_mode
        self.mel_log = mel_log
        self.spec_norm = spec_norm
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=window_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            power=1 if naf_mode else 2,
            pad_mode='constant' if naf_mode else 'reflect',
            norm='slaney' if naf_mode else None,
            mel_scale='slaney' if naf_mode else 'htk'
        )

    def forward(self, x):
        # normalize volume
        p = 1e999 if self.spec_norm == 'max' else 2
        x = torch.nn.functional.normalize(x, p=p, dim=-1)

        if self.naf_mode:
            x = self.mel(x) + 0.06
        else:
            x = self.mel(x) + 1e-8

        if self.mel_log == 'log10':
            x = torch.log10(x)
        elif self.mel_log == 'log':
            x = torch.log(x)

        if self.spec_norm == 'max':
            x = x - torch.amax(x, dim=(-2, -1), keepdim=True)
        return x


def preprocess_wav(wav, segment_size: int = 8000, hop_size: int = 8000, frame_shift_mul: int = 1):
    """
    Preprocess waveform with hop & frame shift
    """
    if isinstance(wav, numpy.ndarray):
        wav = torch.from_numpy(wav)
    if len(wav.shape) != 2:
        models_log.warning('Invalid input shape: %s.', wav.shape)
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
            models_log.warning('Converted input shape to %s.', wav.shape)
        else:
            models_log.error('Fail to auto-fix input size. Exit.')
    if wav.shape[0] not in [1, 2]:
        models_log.warning('Invalid value at dim 0 of input shape (expect 1 or 2 but got %s). Using transpose.',
                           wav.shape[0])
        wav = torch.transpose(wav, 0, 1)

    if wav.shape[0] == 2:
        pow1 = ((wav[0] - wav[1]) ** 2).mean()
        pow2 = ((wav[0] + wav[1]) ** 2).mean()
        # Check for fake stereo
        if pow1 > pow2 * 1000:
            models_log.warning('Fake stereo with opposite phase detected.')
            wav[1] *= -1
    wav = wav.mean(dim=0)

    if wav.shape[0] < segment_size:
        # too short and need to be extended, padding to minimum length
        models_log.warning('Input waveform is too short (%s), padding to segment size (%s).',
                           wav.shape[0], segment_size)
        wav = torch.nn.functional.pad(wav, (0, segment_size - wav.shape[0]))

    # slice overlapping segments
    wav = wav.unfold(0, segment_size, hop_size // frame_shift_mul)
    wav = wav - wav.mean(dim=1).unsqueeze(1)
    return wav
