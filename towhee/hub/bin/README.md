# Towhee hub

## Overview

Towhee hub is a platform to upload and download Towhee-related data files. The data includes [Operators](https://towhee.io/operators) and [Pipelines](https://towhee.io/pipelines) etc. After towhee 0.5.0, everyone can register and contribute to the Towhee hub. You can upload your out-of-the-box pipeline, then use `pipeline(author/pipeline-name)` to download the file and run it easily.

> You may also want to learn about "How to write my first pipeline for Towhee[TBD]" and "[Contribution Guide](https://docs.towhee.io/developer-guides/contributing/contributing-guide)".

There is a `towheehub` tool that can help you easily create, download or initialize a repository. Next, I will introduce the how to use Towhee hub and `towheehub` tool.

## Prerequisites

> If you only want to run the [`download`](#download) or [`generate-yaml`](#generate-yaml) commands, there is no need to install git and git-lfs.

- Create an account for towhee hub: https://towhee.io/user/signup
- Install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- Install [git-lfs](https://git-lfs.github.com)

## Installation and Usage

**(Recommended)** Once you have [installed Towhee](https://docs.towhee.io/get-started/install), you can use `towheehub` with the following command:

```bash
$ pip3 inatsll towhee
$ towheehub <command> -<option> param
```

Of course, you can also run `towhee ` and `towheehub` with source code:

```bash
$ git clone https://github.com/towhee-io/towhee.git
$ python3 setup.py install
$ towheehub <command> -<option> param
```


## Commands

- `create`
  - [`create operator`](#create-operator)
  - [`create pipeline`](#create-pipeline)
- [`init`](#init-pyoperator--init-nnoperator--init-pipeline)
- [`generate-yaml`](#generate-yaml)
- [`download`](#download)
- [`clone`](#clone)

### `create operator`

```bash
$ towhee create operator --help
usage: towhee create {operator,pipeline} [-h] -a AUTHOR -r REPO -p [PASSWORD]

Create Repo on Towhee hub.

positional arguments:
  {operator,pipeline}   Repo class in ['operator', 'pipeline'].

optional arguments:
  -h, --help                            show this help message and exit
  -a AUTHOR, --author AUTHOR            Author of the Repo.
  -r REPO, --repo REPO                  Repo name.
  -p [PASSWORD], --password [PASSWORD]  Password of the author.
```

### `create pipeline`

```bash
$ towhee create pipeline --help
usage: towhee create {operator,pipeline} [-h] -a AUTHOR -r REPO -p [PASSWORD]

Create Repo on Towhee hub.

positional arguments:
  {operator,pipeline}   Repo class in ['operator', 'pipeline'].

optional arguments:
  -h, --help                            show this help message and exit
  -a AUTHOR, --author AUTHOR            Author of the Repo.
  -r REPO, --repo REPO                  Repo name.
  -p [PASSWORD], --password [PASSWORD]  Password of the author.
```

### `init`

```bash
$ towhee init --help
usage: towhee init [-h] -a AUTHOR -r REPO [-c {pyoperator, nnoperator, pipeline}] [-d DIR] [-f FRAMEWORK]

Initialize the file for your Repo.

positional arguments:
  {pyoperator,nnoperator,pipeline}   Repo type/class.

optional arguments:
  -h, --help                           show this help message and exit
  -a AUTHOR, --author AUTHOR           Author of the Repo.
  -r REPO, --repo REPO                 Repo name.
  -c {pyoperator,nnoperator,pipeline}, --classes {pyoperator, nnoperator, pipeline}                  Repo class in ['pyoperator', 'nnoperator', 'pipeline'].
  -d DIR, --dir DIR                    Directory to the Repo file, defaults to '.'.
  -f FRAMEWORK, --framework FRAMEWORK  The framework of nnoperator, defaults to 'pytorch'.
```

### `generate-yaml`

```bash
$ towhee generate-yaml --help
usage: towheehub generate-yaml [-h] -a AUTHOR -r REPO [-d DIR]

Generate yaml file for your Operator Repo.

optional arguments:
  -h, --help                  show this help message and exit
  -a AUTHOR, --author AUTHOR  Author of the Repo.
  -r REPO, --repo REPO        Repo name.
  -d DIR, --dir DIR           Directory to the Repo file, defaults to '.'.
```

### `download`

```bash
$ towhee download --help
usage: towhee download [-h] -a AUTHOR -r REPO [-t TAG] [-d DIR]

Download repo file to local(without git).

optional arguments:
  -h, --help                   show this help message and exit
  -a AUTHOR, --author AUTHOR   Author of the Repo.
  -r REPO, --repo REPO         Repo name.
  -t TAG, --tag TAG            Repo tag or branch, defaults to 'main'.
  -d DIR, --dir DIR            Directory to clone the Repo file, defaults to '.'.
```

### `clone`
```bash
$ towhee download --help
usage: towhee clone [-h] -a AUTHOR -r REPO [-t TAG] [-d DIR]

Clone repo file to local.

optional arguments:
  -h, --help                   show this help message and exit
  -a AUTHOR, --author AUTHOR   Author of the Repo.
  -r REPO, --repo REPO         Repo name.
  -t TAG, --tag TAG            Repo tag or branch, defaults to 'main'.
  -d DIR, --dir DIR            Directory to clone the Repo file, defaults to '.'.
```

## Examples

### 1. Create your own Repo

You can create your own [Operators](https://towhee.io/operators) and [Pipelines](https://towhee.io/pipelines) in the Towhee hub.

- Create PyOperator

  ```bash
  $ towhee create operator -a <your-account-name> -r <your-operator-name> -p <your-password>
  ```
- Create Pipeline
  ```bash
  $ towheehub create pipeline -a <your-account-name> -r <your-pipeline-name> -p
  
  Password: 
  ```
  
  > The you can enter the password and it will be hidden.

### 2. Initialize the Repo with template

You can also initialize your Repo to specific directory, and defaults to the current working directory, which will clone the repo and initialize it according to the template.

- Initialize NNOperator with your model framework

  ```bash
  $ towhee init -a <your-account-name> -r <your-operator-name> -c nnoperator -f </your/model/framework>
  ```

- Initialize Pipeline to specific directory

  ```bash
  $ towhee init -a <your-account-name> -r <your-pipeline-name> -c pipeline -d </your/workspace/path>
  ```

### 3. Generate yaml for Operator Repo

The Operator Repo are required a yaml file which contains the basic input and output information of the Operator for other developers to use, and this command will help you generate it automatically.

  ```bash
  $ towhee generate-yaml -a <your-account-name> -r <your-operator-name>
  ```

### 4. Download the Repo

It will download all files in Repo except **.git** to a specific path (defaults to the current working directory), so it's different with `clone` command.

- Download Repo to  `cwd`

  ```bash
  $ towheehub download -a <your-account-name> -r <your-repo-name>
  ```
- Download Repo to specific directory
  ```bash
  $ towheehub download -a <your-account-name> -r <your-repo-name> -d </your/workspace/path>
  ```

### 5. Clone the Repo

This command is the same as `git clone`, it clones your Repo to a specific path (defaults to the current working directory).

- Clone Repo to  `cwd`

  ```bash
  $ towhee clone -a <your-account-name> -r <your-repo-name>
  ```
- Download Repo to specific directory
  ```bash
  $ towhee clone -a <your-account-name> -r <your-repo-name> -d </your/workspace/path>
  ```

## Others

You can also log in to your account at [towheehub](https://hub.towhee.io/user/login), then go to your repo, and you can configure it, such as add categories/framework, rename and delete.

![img](./towheehub.png)
