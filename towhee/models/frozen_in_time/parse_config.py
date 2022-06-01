# Original pytorch implementation by:
# 'Frozen in Time: A Joint Image and Video Encoder for End-to-End Retrieval'
#       - https://arxiv.org/abs/2104.00650
# Original code by / Copyright 2021, Max Bain.
# Modifications & additions by / Copyright 2021 Zilliz. All rights reserved.
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
import inspect
import logging
import os
import time
from collections import OrderedDict
from datetime import datetime
from functools import reduce
from operator import getitem
from pathlib import Path
import json


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print(f'Warning: logging configuration file is not found in {log_config}.')
        logging.basicConfig(level=default_level)


class ConfigParser:
    """
    args:
    options:
    timestamp:
    test:
    """
    def __init__(self, args=None, options='', timestamp=True, test=False):
        # parse default and custom cli options
        if args is None:
            return
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()

        if args.device:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        if args.resume is None:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            self.cfg_fname = Path(args.config)
            config = read_json(self.cfg_fname)
            self.resume = None
        else:
            self.resume = Path(args.resume)
            resume_cfg_fname = self.resume.parent / 'config.json'
            config = read_json(resume_cfg_fname)
            if args.config is not None:
                config.update(read_json(Path(args.config)))

        # load config file and apply custom cli options
        self._config = _update_config(config, options, args)

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S') if timestamp else ''

        exper_name = self.config['name']
        self._save_dir = save_dir / 'models' / exper_name / timestamp
        self._web_log_dir = save_dir / 'web' / exper_name / timestamp
        self._log_dir = save_dir / 'log' / exper_name / timestamp

        if not test:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # if set, remove all previous experiments with the current config
        if vars(args).get('purge_exp_dir', False):
            for dirpath in (self._save_dir, self._log_dir, self._web_log_dir):
                config_dir = dirpath.parent
                existing = list(config_dir.glob('*'))
                print(f'purging {len(existing)} directories from config_dir...')
                tic = time.time()
                os.system(f'rm -rf {config_dir}')
                print(f'Finished purge in {time.time() - tic:.3f}s')

        # save updated config file to the checkpoint dir
        if not test:
            write_json(self.config, self.save_dir / 'config.json')

            # configure logging module
            setup_logging(self.log_dir)
            self.log_levels = {
                0: logging.WARNING,
                1: logging.INFO,
                2: logging.DEBUG
            }

    def initialize(self, name, module,  *args, index=None, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.
        """
        if index is None:
            module_name = self[name]['type']
            module_args = dict(self[name]['args'])
            assert all(k not in module_args for k in kwargs), 'Overwriting kwargs given in config file is not allowed'
            module_args.update(kwargs)
        else:
            module_name = self[name][index]['type']
            module_args = dict(self[name][index]['args'])

        # if parameter not in config subdict, then check if it's in global config.
        signature = inspect.signature(getattr(module, module_name).__init__)
        print(module_name)
        for param in signature.parameters.keys():
            if param not in module_args and param in self.config:
                module_args[param] = self[param]

        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                       self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir


# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
