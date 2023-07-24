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
import logging
from towhee.config.log_config import LogConfig

FORMAT = '%(asctime)s - %(thread)d - %(filename)s-%(module)s:%(lineno)s - %(levelname)s: %(message)s'

# TODO: for service `access_log`
# TODO: multi progress log
# TODO: user-defined log
logging.basicConfig(format=FORMAT)
root_log = logging.getLogger()
engine_log = logging.getLogger('towhee.engine')
trainer_log = logging.getLogger('towhee.trainer')
models_log = logging.getLogger('towhee.models')

formatter = logging.Formatter(FORMAT)


def _get_time_file_hdlr(config):
    time_rotating_handler = logging.handlers.TimedRotatingFileHandler(
        filename=config.filename,
        when=config.rotate_when,
        interval=config.rotate_interval,
        backupCount=config.backup_count,
        utc=config.utc,
        encoding='utf-8',
        delay=True
    )

    time_rotating_handler.setFormatter(formatter)
    time_rotating_handler.setLevel(config.file_level)
    time_rotating_handler.set_name('time_rotating_file')

    return time_rotating_handler


def _get_size_file_hdlr(config):
    size_rotating_handler = logging.handlers.RotatingFileHandler(
        filename=config.filename,
        maxBytes=config.file_max_size,
        backupCount=config.backup_count,
        encoding='utf-8',
        delay=True
    )
    size_rotating_handler.setFormatter(formatter)
    size_rotating_handler.setLevel(config.file_level)
    size_rotating_handler.set_name('size_rotating_file')

    return size_rotating_handler


def _get_console_hdlr(config):
    console_handler = logging.StreamHandler(config.stream)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(config.stream_level)
    console_handler.set_name('console')

    return console_handler


def _config_logger(logger, handler):
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    logger.addHandler(handler)


def enable_log_config(config: 'towhee.config.LogConfig' = None):
    if not config:
        config = LogConfig()

    time_rotating_handler = _get_time_file_hdlr(config)
    size_rotating_handler = _get_size_file_hdlr(config)
    console_handler = _get_console_hdlr(config)

    if config.mode == 'console':
        _config_logger(root_log, console_handler)
        root_log.setLevel(config.stream_level)
    elif config.mode == 'file':
        if config.rotate_mode == 'size':
            _config_logger(root_log, size_rotating_handler)
        elif config.rotate_mode == 'time':
            _config_logger(root_log, time_rotating_handler)
        else:
            raise ValueError(f'value should be either \'time\' or \'size\', not \'{config.rotate_mode}\'.')
        root_log.setLevel(config.file_level)
    else:
        raise ValueError(f'value should be either \'console\' or \'file\', not \'{config.mode}\'.')
