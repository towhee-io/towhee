# Towhee Command Line Tool

## Installation 

### Install with pip

```bash
$ pip3 install towhee
```

### Install from source code

```bash
$ git clone https://github.com/towhee-io/towhee.git
$ python3 setup.py install
```

Once you have [installed Towhee](https://docs.towhee.io/get-started/install), you can use `towhee` with the following command:

> You can use `python -m towhee` instead of `towhee` to troubleshoot.

```bash
$ towhee -h
usage: towhee

optional arguments:
  -h, --help     show this help message and exit

subcommands:
  towhee command line tool.

  {init,server}
    init         Init operator and generate template file.
    server       Wrap and start pipelines as services.
```


## Usage

### Init Operator
```bash
$ towhee init -h
usage: towhee init [-h] [-d DIR] [-t {pyop,nnop}] uri

positional arguments:
  uri                   Repo uri, in the form of <repo-author>/<repo-name>.

optional arguments:
  -h, --help            show this help message and exit
  -d DIR, --dir DIR     optional, directory to the operator, defaults to current working directory.
  -t {pyop,nnop}, --type {pyop,nnop}
                        optional, operator type, defaults to 'pyop'.
```


### Start Pipeline Service

```bash
$ towhee server -h
usage: towhee server [-h] [--host HOST] [--http-port HTTP_PORT] [--grpc-port GRPC_PORT] [--uri [URI [URI ...]]] [--params [PARAMS [PARAMS ...]]]
                     [source [source ...]]

positional arguments:
  source                The source of the pipeline, could be either in the form of `python_module:api_interface` or repository from Towhee hub.

optional arguments:
  -h, --help            show this help message and exit
  --host HOST           The service host.
  --http-port HTTP_PORT
                        The http service port.
  --grpc-port GRPC_PORT
                        The grpc service port.
  --uri [URI [URI ...]]
                        The uri to the pipeline service
  --params [PARAMS [PARAMS ...]]
                        Parameters to initialize the pipeline.
```

## Examples

### Init Operator

Initialize operator from the [Towhee hub](https://towhee.io/operators). This command will clone the repository and initialize it according to different operator types.

> There are two kinds of operator namely `pyop` and `nnop`, `pyop` is the operator that contains python processing function only, and `nnop` is the operator that involves neural networks.

```bash
$ towhee init <repo-author>/<repo-name> -t <operator-type> -d <directory>
```

### Start Pipeline Service

Start pipelines as services. This command will start pipeline sevices according to the specified python file or pipeline repository.

```bash
$ towhee server <pipeline-source> --host <host> --http-port <http-port> --grpc-port <grpc-port> --uri <uri-to-service> --params <params-for-pipelines>
```

- Python File

  There are two ways of defining pipeline in a python file:
  ```python
  from towhee import api_service, AutoPipes

  service = api_service.APIService(desc='Welcome')
  emb_pipe = AutoPipes.pipeline('image-embedding')

  @service.api(path='/emb')
  def embedding(src: str):
    return emb_pipe(src).get()
  ```
  Or you can run `build_service` to build a service for you.
  ```python
  from towhee import AutoPipes
  from towhee import api_service

  img = AutoPipes.pipeline('image-embedding')
  service = api_service.build_service([(img, '/emb')])
  ```

You can start a server either from a python file and interface or repositories from towhee hub. Now Towhee supports http and grpc service:

- Start http service from python file
  ```bash
  $ towhee server my_pipeline_file:service --host 0.0.0.0 --http-port 40001
  ```

- Start grpc service from Towhee hub repository
  ```bash
  $ towhee server audio-embedding image-embedding --grpc-port 50001 --uri /emb/audio /emb/image --params none model_name=resnet34,device=0
  ```

- Access to Towhee pipeline service

  To access http service:
  ```python
  import requests
  import json

  res = requests.post('http://0.0.0.0:40001/emb', json.dumps('https://github.com/towhee-io/towhee/raw/main/towhee_logo.png'))
  ```

  To access grpc service:
  ```python
  from towhee.serve.grpc.client import Client

  grpc_client = Client(host='0.0.0.0', port=50001)
  res = grpc_client('/emb/image', 1)
  ```
