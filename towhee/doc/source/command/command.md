# Towhee Commandline Tools

Towhee now supports several commands:
- `init`: Initialize existing operators from Towhee hub.
- `server`: Start a server that provide the functionality of specified pielines.


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
usage: towhee server [-h] [-s HOST] [-p PORT] [-i INTERFACE] [-r [REPO [REPO ...]]] [-u [URI [URI ...]]] [-a [PARAMS [PARAMS ...]]] [-f PYTHON]
                     [-t] [-g]

optional arguments:
  -h, --help            show this help message and exit
  -s HOST, --host HOST  The service host.
  -p PORT, --port PORT  The service port.
  -i INTERFACE, --interface INTERFACE
                        The service interface, i.e. the APIService object defined in python file.
  -r [REPO [REPO ...]], --repo [REPO [REPO ...]]
                        Repo of the pipeline on towhee hub to start the service.
  -u [URI [URI ...]], --uri [URI [URI ...]]
                        The uri to the pipeline service
  -a [PARAMS [PARAMS ...]], --params [PARAMS [PARAMS ...]]
                        Parameters to initialize the pipeline.
  -f PYTHON, --python PYTHON
                        Path to the python file that define the pipeline.
  -t, --http            Start service by HTTP.
  -g, --grpc            Start service by GRPC.
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
$ towhee server --host <host> --port <port> --interface <interface> --python <path-to-python-file>/--repo <pipeline-repo-names> --uri <uri-to-service> --params <params-for-pipelines> --http/--grpc
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

- Create service from python file
  ```bash
  $ towhee server --host localhost --port 8000 --interface service --python my_pipeline_file.py --http
  ```

- Create service from Towhee hub repository
  ```bash
  $ towhee servr --repo audio-embedding image-embedding --uri /emb/audio /emb/image --params none model_name=resnet34,device=0 --grpc
  ```

- Access to Towhee pipeline service
  ```python
  import requests
  import json

  res = requests.post('http://localhost:8000/emb', json.dumps('https://github.com/towhee-io/towhee/raw/main/towhee_logo.png'))
  ```
