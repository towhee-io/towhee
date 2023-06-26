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
  -h, --help            show this help message and exit

subcommands:
  towhee command line tool.

  {init}
    init      Init operator and generate template file.
```

## Usage

- Init Operator
  - [`init`](#init)

### Init Operator
#### `init`
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

## Examples

### Init Operator

- **Create Operator in Towhee hub**

  Initialize operator from the [Towhee hub](https://towhee.io/operators).This command will clone the repository and initialize it according to different operator types.

  > There are two kinds of operator namely `pyop` and `nnop`, `pyop` is the operator that contains python processing function only, and `nnop` is the operator that involves neural networks.

  ```bash
  $ towhee init <repo-author>/<repo-name> -t <operator-type> -d <directory>
  ```
