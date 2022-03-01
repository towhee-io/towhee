# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team and 2021 Zilliz.
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
from typing import Dict, Tuple
from tqdm import tqdm
from towhee.trainer.callback import  TrainerControl, Callback, CallbackList, EarlyStoppingCallback, ModelCheckpointCallback, TensorBoardCallBack, PrintCallBack, ProgressBarCallBack


class TestCallback(unittest.TestCase):
    """
    TestCallback
    """
    logs = {
        'global_step': 1,
        'step_loss': 1,
        'epoch_loss': 1,
        'epoch_metric': 1,
        'eval_global_step': 1,
        'eval_step_loss': 1,
        'eval_epoch_loss': 1,
        'eval_epoch_metric': 1,
        'lr': 1,
        'epoch': 1
    }
    class CustomCallback(Callback):
        """
        CustomCallback
        """
        def __init__(self):
            pass

        def on_batch_begin(self, batch: Tuple, logs: Dict) -> None:
            logs['on_batch_begin'] = logs['on_batch_begin'] + 1

        def on_batch_end(self, batch: Tuple, logs: Dict) -> None:
            logs['on_batch_end'] = logs['on_batch_end'] + 1

        def on_epoch_begin(self, epochs: int, logs: Dict) -> None:
            logs['on_epoch_begin'] = logs['on_epoch_begin'] + 1

        def on_epoch_end(self, epochs: int, logs: Dict) -> None:
            logs['on_epoch_end'] = logs['on_epoch_end'] + 1

        def on_train_begin(self, logs: Dict) -> None:
            logs['on_train_begin'] = logs['on_train_begin'] + 1

        def on_train_end(self, logs: Dict) -> None:
            logs['on_train_end'] = logs['on_train_end'] + 1

        def on_train_batch_begin(self, batch: Tuple, logs: Dict) -> None:
            logs['on_train_batch_begin'] = logs['on_train_batch_begin'] + 1

        def on_train_batch_end(self, batch: Tuple, logs: Dict) -> None:
            logs['on_train_batch_end'] = logs['on_train_batch_end'] + 1

        def on_eval_batch_begin(self, batch: Tuple, logs: Dict) -> None:
            logs['on_eval_batch_begin'] = logs['on_eval_batch_begin'] + 1

        def on_eval_batch_end(self, batch: Tuple, logs: Dict) -> None:
            logs['on_eval_batch_end'] = logs['on_eval_batch_end'] + 1

        def on_eval_begin(self, logs: Dict) -> Dict:
            logs['on_eval_begin'] = logs['on_eval_begin'] + 1

        def on_eval_end(self, logs: Dict) -> Dict:
            logs['on_eval_end'] = logs['on_eval_end'] + 1

    def _init_log(self) -> Dict:
        logs = {}
        logs['on_batch_begin'] = 0
        logs['on_batch_end'] = 0
        logs['on_epoch_begin'] = 0
        logs['on_epoch_end'] = 0
        logs['on_train_batch_begin'] = 0
        logs['on_train_batch_end'] = 0
        logs['on_train_begin'] = 0
        logs['on_train_end'] = 0
        logs['on_eval_batch_begin'] = 0
        logs['on_eval_batch_end'] = 0
        logs['on_eval_begin'] = 0
        logs['on_eval_end'] = 0
        return logs

    def test_callback(self) -> None:
        train_log = self._init_log()
        eval_log = self._init_log()
        callback = TestCallback.CustomCallback()

        #simulate the train loop.
        train_epoch_nums = 3
        train_batch_nums = 5
        callback.on_train_begin(train_log)
        for train_epoch in range(train_epoch_nums):
            callback.on_epoch_begin(train_epoch, train_log)
            for _ in range(train_batch_nums):
                mock_batch = (None, None)
                callback.on_train_batch_begin(mock_batch, train_log)
                callback.on_train_batch_end(mock_batch, train_log)
            callback.on_epoch_end(train_epoch, train_log)
        callback.on_train_end(train_log)

        def _assert_train_attrs(attr, gt):
            self.assertEqual(train_log[attr], gt)

        _assert_train_attrs('on_batch_begin', 0)
        _assert_train_attrs('on_batch_end', 0)
        _assert_train_attrs('on_epoch_begin', train_epoch_nums)
        _assert_train_attrs('on_epoch_end', train_epoch_nums)
        _assert_train_attrs('on_train_batch_begin', train_epoch_nums * train_batch_nums)
        _assert_train_attrs('on_train_batch_end', train_epoch_nums * train_batch_nums)
        _assert_train_attrs('on_train_begin', 1)
        _assert_train_attrs('on_train_end', 1)

        #simulate the eval loop.
        eval_epoch_nums = 5
        eval_batch_nums = 10
        eval_log = self._init_log()
        callback.on_eval_begin(eval_log)
        for eval_epoch in range(eval_epoch_nums):
            callback.on_epoch_begin(eval_epoch, eval_log)
            for _ in range(eval_batch_nums):
                mock_batch = (None, None)
                callback.on_eval_batch_begin(mock_batch, eval_log)
                callback.on_eval_batch_end(mock_batch, eval_log)
            callback.on_epoch_end(eval_epoch, eval_log)
        callback.on_eval_end(eval_log)

        def _assert_eval_attrs(attr, gt):
            self.assertEqual(eval_log[attr], gt)

        _assert_eval_attrs('on_batch_begin', 0)
        _assert_eval_attrs('on_batch_end', 0)
        _assert_eval_attrs('on_epoch_begin', eval_epoch_nums)
        _assert_eval_attrs('on_epoch_end', eval_epoch_nums)
        _assert_eval_attrs('on_eval_batch_begin', eval_epoch_nums * eval_batch_nums)
        _assert_eval_attrs('on_eval_batch_end', eval_epoch_nums * eval_batch_nums)
        _assert_eval_attrs('on_eval_begin', 1)
        _assert_eval_attrs('on_eval_end', 1)

    def test_callbackslist(self) -> None:
        callbacklist = CallbackList([TestCallback.CustomCallback(), TestCallback.CustomCallback()])

        cb = TestCallback.CustomCallback()
        callbacklist.add_callback(cb)
        print(callbacklist)
        self.assertEqual(len(callbacklist), 2)
        callbacklist.pop_callback(cb)
        self.assertEqual(len(callbacklist), 1)

        train_log = self._init_log()
        eval_log = self._init_log()

        train_epoch_nums = 3
        train_batch_nums = 5
        #simulate the train loop.
        callbacklist.on_train_begin(train_log)
        for train_epoch in range(train_epoch_nums):
            callbacklist.on_epoch_begin(train_epoch, train_log)
            for _ in range(train_batch_nums):
                mock_batch = (None, None)
                callbacklist.on_train_batch_begin(mock_batch, train_log)
                callbacklist.on_train_batch_end(mock_batch, train_log)
            callbacklist.on_epoch_end(train_epoch, train_log)
        callbacklist.on_train_end(train_log)

        def _assert_train_attrs(attr, gt):
            self.assertEqual(train_log[attr], gt)

        num_callbacks = len(callbacklist)
        _assert_train_attrs('on_batch_begin', 0 * num_callbacks)
        _assert_train_attrs('on_batch_end', 0 * num_callbacks)
        _assert_train_attrs('on_epoch_begin', train_epoch_nums * num_callbacks)
        _assert_train_attrs('on_epoch_end', train_epoch_nums * num_callbacks)
        _assert_train_attrs('on_train_batch_begin', train_epoch_nums * train_batch_nums * num_callbacks)
        _assert_train_attrs('on_train_batch_end', train_epoch_nums * train_batch_nums * num_callbacks)
        _assert_train_attrs('on_train_begin', 1 * num_callbacks)
        _assert_train_attrs('on_train_end', 1 * num_callbacks)

        #simulate the eval loop.
        eval_epoch_nums = 5
        eval_batch_nums = 10
        callbacklist.on_eval_begin(eval_log)
        for eval_epoch in range(eval_epoch_nums):
            callbacklist.on_epoch_begin(eval_epoch, eval_log)
            for _ in range(eval_batch_nums):
                mock_batch = (None, None)
                callbacklist.on_eval_batch_begin(mock_batch, eval_log)
                callbacklist.on_eval_batch_end(mock_batch, eval_log)
            callbacklist.on_epoch_end(eval_epoch, eval_log)
        callbacklist.on_eval_end(eval_log)

        def _assert_eval_attrs(attr, gt):
            self.assertEqual(eval_log[attr], gt)

        _assert_eval_attrs('on_batch_begin', 0 * num_callbacks)
        _assert_eval_attrs('on_batch_end', 0 * num_callbacks)
        _assert_eval_attrs('on_epoch_begin', eval_epoch_nums * num_callbacks)
        _assert_eval_attrs('on_epoch_end', eval_epoch_nums * num_callbacks)
        _assert_eval_attrs('on_eval_batch_begin', eval_epoch_nums * eval_batch_nums * num_callbacks)
        _assert_eval_attrs('on_eval_batch_end', eval_epoch_nums * eval_batch_nums * num_callbacks)
        _assert_eval_attrs('on_eval_begin', 1 * num_callbacks)
        _assert_eval_attrs('on_eval_end', 1 * num_callbacks)

    def test_earlystopping(self):
        trainer_control = TrainerControl()
        earlystopping_callback = EarlyStoppingCallback(trainer_control, 'loss', patience = 3, mode = 'min')

        loss_value = [5.0, 4.0, 3.0, 2.5, 2.1, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        def loss_emulator():
            for value in loss_value:
                yield value

        train_log = {}
        earlystopping_callback.on_train_begin(train_log)
        for epoch, loss_value in enumerate(loss_emulator()):
            train_log['loss'] = loss_value
            earlystopping_callback.on_epoch_end(epoch, train_log)
        earlystopping_callback.on_train_end(train_log)
        assert earlystopping_callback.trainercontrol.should_training_stop is True

    def test_modelcheckpoint(self):
        trainer_control = TrainerControl()
        modelcheckpoint_callback = ModelCheckpointCallback(trainer_control, every_n_epoch = 5)
        epoch_status = []
        for epoch in range(11):
            modelcheckpoint_callback.on_epoch_end(epoch)
            epoch_status.append(modelcheckpoint_callback.trainercontrol.should_save)
        assert(epoch_status == [True, False, False, False, False, True, False, False, False, False, True])

        modelcheckpoint_callback = ModelCheckpointCallback(trainer_control, every_n_iteration = 5)
        iteration_status = []
        for _ in range(11):
            batch_data = (None, None)
            modelcheckpoint_callback.on_batch_end(batch_data)
            iteration_status.append(modelcheckpoint_callback.trainercontrol.should_save)
        assert(iteration_status == [True, False, False, False, False, True, False, False, False, False, True])

    def test_tensorboard(self):
        class MockSW:
            def __init__(self, log_dir, comment=''):
                self.log_dir = log_dir
                self.comment = comment
                self.sw_list = []
            def __call__(self, *args, **kwargs):
                return self
            def add_scalar(self, *args):
                self.sw_list.append(args)

        mock_sw = MockSW(log_dir='runs')

        tensorboard_callback = TensorBoardCallBack(summary_writer_constructor=mock_sw)
        tensorboard_callback.on_train_batch_end(batch=(1,), logs=self.logs)
        tensorboard_callback.on_eval_batch_end(batch=(1,), logs=self.logs)

    def test_print(self):
        print_callback = PrintCallBack(total_epoch_num=10, step_frequency=1)
        print_callback.on_train_batch_end(batch=(1,), logs=self.logs)
        print_callback.on_eval_batch_end(batch=(1,), logs=self.logs)

    def test_progressbar(self):
        progressbar_callback = ProgressBarCallBack(total_epoch_num=10, train_dataloader=tqdm(range(2)))
        progressbar_callback.on_train_batch_end(batch=(1,), logs=self.logs)
        progressbar_callback.on_epoch_begin(epochs=1, logs=self.logs)
        progressbar_callback.on_eval_batch_end(batch=(1,), logs=self.logs)

if __name__ == '__main__':
    unittest.main(verbosity=1)
