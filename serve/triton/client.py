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

import sys
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from functools import partial

model_name = 'pipeline'

if sys.version_info >= (3, 0):
	import queue
else:
	import Queue as queue

class UserData:
	def __init__(self):
		self._completed_requests = queue.Queue()

# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
	# passing error raise and handling out
	user_data._completed_requests.put((result, error))

class Client():
	def __init__(self, url, model_name=model_name, protocol='grpc'):
		self._model_name = model_name
		self.user_data = UserData()
		if protocol == 'grpc':
			self.protocol = 'grpc'
			self.client = grpcclient.InferenceServerClient(url)
			self.client.start_stream(partial(completion_callback, self.user_data))
		else:
			self.protocol = 'http'
			self.client = httpclient.InferenceServerClient(url)

	def __del__(self):
		if self.protocol == 'grpc':
			self.stop_stream()

	def get_model_name(self):
		return self._model_name

	def async_set(self, iterator):
		if self.protocol != 'grpc':
			raise Exception('Streaming is only allowed with gRPC protocol')
		inputs, outputs = self._solve_inputs_outputs(iterator)

		sent_count = 0
		responses = []
		for i in range(len(iterator)):
			sent_count += 1
			if self.protocol == 'grpc':
				self.client.async_infer(self.get_model_name(), inputs, partial(completion_callback, self.user_data), request_id=str(sent_count), outputs=outputs)
		
		processed_count = 0
		while processed_count < sent_count:
			(results, error) = self.user_data._completed_requests.get()
			processed_count += 1
			if error is not None:
				sys.exit(1)
			if processed_count <= len(iterator):
				responses.append(results)
		
		return self._solve_responses(outputs, responses)

	def stream(self, iterator):
		if self.protocol != 'grpc':
			raise Exception('Streaming is only allowed with gRPC protocol')
		inputs, outputs = self._solve_inputs_outputs(iterator)

		sent_count = 0
		responses = []
		try:
			for i in range(len(iterator)):
				sent_count += 1
				self.client.async_stream_infer(self.get_model_name(), inputs, request_id=str(sent_count), outputs=outputs)
		except InferenceServerException as e:
			self.stop_stream()
			sys.exit(1)

		processed_count = 0
		while processed_count < sent_count:
			(results, error) = self.user_data._completed_requests.get()
			processed_count += 1
			if error is not None:
				sys.exit(1)
			if processed_count <= len(iterator):
				responses.append(results)
		return self._solve_responses(outputs, responses)

	def serve(self, iterator):
		if type(iterator) != type([]):
			iterator = [iterator]
		inputs, outputs = self._solve_inputs_outputs(iterator)

		sent_count = 0
		responses = []
		for i in range(len(iterator)):
			sent_count += 1
			responses.append(self.client.infer(self.get_model_name(), inputs, request_id=str(sent_count), outputs=outputs))
		return self._solve_responses(outputs, responses)

	def _solve_inputs_outputs(self, iterator):
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

		if batch_size == 0 or len(iterator) <= batch_size:
			batch_size = 1

		image_data = []
		for idx in range(len(iterator)):
			image_data.append(np.array(iterator[idx].encode('utf-8'), dtype=np.object_))
		if len(iterator) > 1:
			batch_image_data = np.stack([image_data[0]], axis=0)
		else:
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
		return inputs, outputs

	def _solve_responses(self, outputs, responses):
		res = []
		for i in range(len(responses)):
			res_dict = {}
			for j in range(len(outputs)):
				if self.protocol == 'grpc':
					output_name = outputs[j].name()
				else:
					output_name = outputs[j].name()
				res_dict[output_name] = responses[i].as_numpy(output_name)
			res.append(res_dict)
		return res

	def stop_stream(self):
		self.client.stop_stream()
