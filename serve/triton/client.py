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


import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

model_name = 'pipeline'

class Client():
	def __init__(self, url, model_name=model_name, protocol='grpc'):
		self._model_name = model_name
		if protocol == 'grpc':
			self.protocol = 'grpc'
			self.client = grpcclient.InferenceServerClient(url)
		else:
			self.protocol = 'http'
			self.client = httpclient.InferenceServerClient(url)

	def get_model_name(self):
		return self._model_name

	def serve(self, *args):
		model_config = self.client.get_model_config(model_name=self.get_model_name())
		model_metadata = self.client.get_model_metadata(model_name=self.get_model_name())
		inputs = []
		if self.protocol == 'grpc':
			batch_size = model_config.config.max_batch_size
			inputs_list = model_metadata.inputs
			outputs_list = model_metadata.outputs
		else:
			batch_size = model_config['max_batch_size']
			inputs_list = model_metadata['inputs']
			outputs_list = model_metadata['outputs']
		for i in inputs_list:
			if self.protocol == 'grpc':
				inputs.append(grpcclient.InferInput(i.name, i.shape, i.datatype))
			else:
				inputs.append(httpclient.InferInput(i['name'], i['shape'], i['datatype']))

		if batch_size == 0 or len(args) <= batch_size:
			batch_size = len(args)

		image_data = []
		for idx in range(batch_size):
			image_data.append(np.array(args[idx].encode('utf-8'), dtype=np.object_))
		
		batch_image_data = np.stack(image_data, axis=0)
		if self.protocol == 'grpc':
			inputs[0].set_data_from_numpy(batch_image_data)
		else:
			inputs[0].set_data_from_numpy(batch_image_data, binary_data=True)


		output_names = [
        	output.name if self.protocol == "grpc" else output['name']
        	for output in outputs_list
    	]

		outputs = []
		
		for output_name in output_names:
			if self.protocol == 'grpc':
				outputs.append(grpcclient.InferRequestedOutput(output_name))
			else:
				outputs.append(httpclient.InferRequestedOutput(output_name, binary_data=True))

		response = self.client.infer(self.get_model_name(), inputs, outputs=outputs)
		res_dict = dict()
		for output in outputs_list:
			if self.protocol == 'grpc':
				output_name = output.name
			else:
				output_name = output['name']
			res_dict[output_name] = response.as_numpy(output_name)

		return res_dict
