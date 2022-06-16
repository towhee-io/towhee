def create_modelconfig(model_name, max_batch_size, inputs, outputs,
                       enable_dynamic_batching=None, preferred_batch_size=None,
                       max_queue_delay_microseconds=None):
    '''
    example of input and output:
        {
            'name': 'input0',
            'data_type': 'int8',
            'dims': []
        }
    '''
    config = "name: \"{}\"\n".format(model_name)
    config += "backend: \"python\"\n"
    config += "max_batch_size: {}\n".format(max_batch_size)
    if enable_dynamic_batching:
        config += '''
dynamic_batching {
'''
        if preferred_batch_size is not None:
            config += '''
    preferred_batch_size: {}
'''.format(preferred_batch_size)
        if max_queue_delay_microseconds is not None:
            config += '''
    max_queue_delay_microseconds: {}
'''.format(max_queue_delay_microseconds)
        config += '''
}\n'''
    for input_name in inputs.keys():
        print(input_name)
        data_type, shape = inputs[input_name]
        config += '''
input [
  {{
    name: \"{}\"
    data_type: {}
    dims: {}
  }}
]\n'''.format(input_name, "TYPE_" + data_type, shape)
    for output_name in outputs.keys():
        data_type, shape = outputs[output_name]
        config += '''
output [
  {{
    name: \"{}\"
    data_type: {}
    dims: {}
  }}
]\n'''.format(output_name, "TYPE_" + data_type, shape)
    return config


def create_ensemble(dag, name='pipeline', max_batch_size=128):
    pass


if __name__ == '__main__':
    data = create_modelconfig('test', 128,
                              {'input0': ('INT8', [-1, -1, 3]),
                               'input1': ('FL16', [-1, -1, 3])},
                              {'output0': ('INT8', [-1, -1, 3]),
                               'output1': ('FL16', [-1, -1, 3])}
                              )
    print(data)
