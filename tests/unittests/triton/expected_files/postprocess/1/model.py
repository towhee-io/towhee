import towhee
import numpy
from pathlib import Path
import pickle
import importlib
import sys
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        
        # load module
        module_name = "towhee.operator.triton_nnop"
        path = ""
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # create callable object
        pickle_file_path = Path(__file__).parent / "postprocess.pickle"
        with open(pickle_file_path, 'rb') as f:
            self.callable_obj = pickle.load(f)

    def execute(self, requests):
        
        responses = []
        
        for request in requests:
            # get input tensors from request
            in0 = pb_utils.get_input_tensor_by_name(request, 'INPUT0')
            
            # create input args from tensors
            arg0 = in0.as_numpy()
            
            # call callable object
            result0 = self.callable_obj(arg0)
            
            # convert results to tensors
            out0 = pb_utils.Tensor('OUTPUT0', numpy.array(result0, numpy.float32))
            
            # organize response
            response = pb_utils.InferenceResponse(output_tensors=[out0])
            responses.append(response)
        
        return responses

    def finalize(self):
        pass
