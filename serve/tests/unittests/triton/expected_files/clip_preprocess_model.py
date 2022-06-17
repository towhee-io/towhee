import towhee
import numpy
import inspect
import pickle
import importlib
import sys
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        
        # load module
        spec = importlib.util.spec_from_file_location('clip', 'clip.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules['clip'] = module
        spec.loader.exec_module(module)
        
        # create callable object
        callable_cls = clip.Preprocess
        with open('preprocess.pickle', 'rb') as f:
            self.callable_obj = pickle.load(f)

    def execute(self, requests):
        
        responses = []
        
        for request in requests:
            # get input tensors from request
            in0 = pb_utils.get_input_tensor_by_name(request, 'INPUT0')
            in1 = pb_utils.get_input_tensor_by_name(request, 'INPUT1')
            
            # create input args from tensors
            arg0 = towhee._types.Image(in0.as_numpy(), str(in1.as_numpy()).decode('utf-8'))
            
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
