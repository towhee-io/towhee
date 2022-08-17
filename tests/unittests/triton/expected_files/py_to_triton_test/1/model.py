import towhee
import numpy
from towhee import ops
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        
        device = "cpu"
        if args["model_instance_kind"] == "GPU":
            device = int(args["model_instance_device_id"])
        # create op instance
        task = getattr(ops, 'local')
        init_args = {}
        op_wrapper = getattr(task, 'triton_py')(**init_args)
        self.op = op_wrapper.get_op()
        self.op._device = device
        if hasattr(self.op, "to_device"):
            self.op.to_device()

    def execute(self, requests):
        
        responses = []
        
        for request in requests:
            # get input tensors from request
            in0 = pb_utils.get_input_tensor_by_name(request, 'INPUT0')
            
            # create input args from tensors
            arg0 = in0.as_numpy()[0].decode('utf-8')
            
            # call callable object
            result0 = self.op(arg0)
            
            # convert results to tensors
            out0 = pb_utils.Tensor('OUTPUT0', numpy.array(result0, numpy.int8))
            out1 = pb_utils.Tensor('OUTPUT1', numpy.array([result0.mode], numpy.object_))
            
            # organize response
            response = pb_utils.InferenceResponse(output_tensors=[out0, out1])
            responses.append(response)
        
        return responses

    def finalize(self):
        pass
