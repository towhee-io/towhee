#coding=utf-8
# pylint: skip-file

import logging
import json
from pathlib import Path

import numpy as np
import dill as pickle

from towhee.serve.triton.bls.python_backend_wrapper import pb_utils
from towhee.runtime.runtime_pipeline import RuntimePipeline
from towhee.utils.np_format import NumpyArrayDecoder, NumpyArrayEncoder



logger = logging.getLogger()


class TritonPythonModel:
    '''
    Pipeline Model
    '''
    @staticmethod
    def auto_complete_config(auto_complete_model_config):

        input0 = {'name': 'INPUT0', 'data_type': 'TYPE_STRING', 'dims': [1]}
        output0 = {'name': 'OUTPUT0', 'data_type': 'TYPE_STRING', 'dims': [1]}

        auto_complete_model_config.set_max_batch_size(0)
        auto_complete_model_config.add_input(input0)
        auto_complete_model_config.add_output(output0)
        return auto_complete_model_config

    def initialize(self, args):
        self._load_pipeline()

    def _load_pipeline(self, fpath=None) -> str:
        if fpath is None:
            fpath = str(Path(__file__).parent.resolve() / 'pipe.pickle')
        with open(fpath, 'rb') as f:
            dag_repr = pickle.load(f)
            self.pipe = RuntimePipeline(dag_repr)
            self.pipe.preload()

    def _get_result(self, q):
        ret = []
        while True:
            data = q.get()
            if data is None:
                break
            ret.append(data)
        return ret

    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            arg = in_0.as_numpy()[0]
            inputs = json.loads(arg, cls=NumpyArrayDecoder)
            q = self.pipe(*inputs)
            ret = self._get_result(q)
            ret_str = json.dumps(ret, cls=NumpyArrayEncoder)
            out_tensor_0 = pb_utils.Tensor('OUTPUT0', np.array([ret_str], np.object_))
            responses.append(pb_utils.InferenceResponse([out_tensor_0]))
        return responses

    def finalize(self):
        pass
