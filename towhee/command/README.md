# Towhee Command Line Tool

## Installation 

**(Recommended)** Once you have [installed Towhee](https://docs.towhee.io/get-started/install), you can use `towhee` with the following command:

```bash
$ pip3 inatsll towhee
$ towhee <command> -<option> param
```

Of course, you can also run `towhee `  with source code:

```bash
$ git clone https://github.com/towhee-io/towhee.git
$ python3 setup.py install
$ towhee <command> -<option> param
```


## Usage

- [`develop`](#develop-operator)
- [`install`](#install-operator)
- [`run`](#run-pipeline)

### `develop operator`

```bash
$ towhee develop --help
usage: towhee develop [-h] [-n NAMESPACE] path

develop operator with setup.py develop

positional arguments:
  path                  path to the operator repo, cwd is '.'

optional arguments:
  -h, --help            show this help message and exit
  -n NAMESPACE, --namespace NAMESPACE
                        repo author/namespace
```

### `install operator`

```bash
$ towhee install --help
usage: towhee install [-h] [-n NAMESPACE] path

install operator with setup.py install

positional arguments:
  path                  path to the operator repo, cwd is '.'

optional arguments:
  -h, --help            show this help message and exit
  -n NAMESPACE, --namespace NAMESPACE
                        repo author/namespace
```

### `run pipeline`

```bash
$ towhee run --help
usage: towhee run [-h] [-i INPUT] [-o OUTPUT] pipeline

run towhee pipeline

positional arguments:
  pipeline              pipeline repo or path to yaml

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input the parameter for pipeline defaults to None
  -o OUTPUT, --output OUTPUT
                        path to the file that will be used to write results], defaults to None which will print the result
```

## Examples

### 1. Develop your Operator

When you are developing the code, you can run `towhee develop` to setup your operator for testing.

```bash
$ towhee develop <path/to/your/operator>
# Or
$ towhee develop .
```

> It will generate symbolic link to the python package of towheeoperator.

### 2. Install your Operator

When you have finished the code, you can run `towhee install` to setup python package..

```bash
$ towhee install <path/to/your/operator>
# Or
$ towhee install .
```

> It will setup towheeoperator.[repo] python package.

### 3. Run Pipeline

You can run the pipeline in hub or your specific yaml file.

  ```bash
  $ towhee run <path/to/your/yaml/file>
  # OR
  $ towhee run towhee/image-embedding-resnet50
  ```

