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
import re
import logging
import unittest
from pathlib import Path
from io import StringIO

from towhee.config import LogConfig
from towhee.utils.log import engine_log, enable_log_config


class TestLog(unittest.TestCase):
    """
    Test towhee logs.
    """
    def test_log(self):
        f = StringIO()

        with self.assertRaises(ValueError):
            config = LogConfig(mode='test')

        with self.assertRaises(ValueError):
            config = LogConfig(rotate_mode='test')

        config = LogConfig(mode='console', rotate_mode='time', stream=f)
        root = logging.getLogger()

        self.assertEqual(root.level, 30)
        self.assertEqual(engine_log.level, 0)

        # Default stream log
        enable_log_config()
        engine_log.warning('test_warning')
        pattern = r'\d*-\d*-\d* \d*:\d*:\d*,\d* - \d* - test_log.py-test_log:\d* - WARNING: test_warning'
        self.assertTrue(re.search(pattern, f.getvalue()) is not None)
        self.assertEqual(root.level, 30)

        # Debug level stream log
        config.stream_level = 'DEBUG'
        enable_log_config(config)
        engine_log.debug('test_debug')
        pattern = r'\d*-\d*-\d* \d*:\d*:\d*,\d* - \d* - test_log.py-test_log:\d* - DEBUG: test_debug'
        self.assertTrue(re.search(pattern, f.getvalue()) is not None)
        self.assertEqual(root.level, 10)

        f.close()

        # Default file log
        config.mode = 'file'
        enable_log_config(config)
        engine_log.info('test_info')
        pattern = r'\d*-\d*-\d* \d*:\d*:\d*,\d* - \d* - test_log.py-test_log:\d* - INFO: test_info'
        with open('towhee.log', encoding='utf-8') as file:
            self.assertTrue(re.search(pattern, file.read()) is not None)
        self.assertEqual(root.level, 20)
        self.assertIsInstance(root.handlers[-1], logging.handlers.TimedRotatingFileHandler)
        Path('towhee.log').unlink()

        # Error level file log
        config.mode = 'file'
        config.rotate_mode = 'size'
        config.file_level = 'ERROR'
        enable_log_config(config)
        engine_log.error('test_error')
        pattern = r'\d*-\d*-\d* \d*:\d*:\d*,\d* - \d* - test_log.py-test_log:\d* - ERROR: test_error'
        with open('towhee.log', encoding='utf-8') as file:
            self.assertTrue(re.search(pattern, file.read()) is not None)
        self.assertEqual(root.level, 40)
        self.assertIsInstance(root.handlers[-1], logging.handlers.RotatingFileHandler)
        Path('towhee.log').unlink()

        with self.assertRaises(ValueError):
            config.mode = 'test'
            enable_log_config()

        with self.assertRaises(ValueError):
            config.rotate_mode = 'test'
            enable_log_config()
