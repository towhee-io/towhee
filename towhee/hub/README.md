# Towhee Hub

## Overview

Towhee hub is a platform based on Gitea and Git LFS (Large File Storage) to upload and download Towhee-related data files. The data includes [Operators](https://towhee.io/operators) and [Pipelines](https://towhee.io/pipelines) etc. After towhee 0.5.0, everyone can register and contribute to the Towhee hub. You can upload your out-of-the-box pipeline, then use `pipeline(author/pipeline-name)` to download the file and run it easily.

> You may also want to learn about "How to write my first pipeline for Towhee[TBD]" and "[Contribution Guide](https://docs.towhee.io/developer-guides/contributing/contributing-guide)".

There is a `towheehub` tool that can help you easily create, download or initialize a repository. Next, I will introduce the usage of `towheehub`.

## Installation and Usage

Once you have [installed Towhee](https://docs.towhee.io/get-started/install), you can use `towheehub` with the following command:

```bash
$ pip3 inatsll towhee
$ towheehub -<command> -<option> param
```

Of course, you can also run [hub_tools.py](https://github.com/towhee-io/towhee/blob/main/towhee/hub/hub_tools.py) with source code:

```bash
$ git clone https://github.com/towhee-io/towhee.git
$ python3 setup.py install
$ towheehub -<command> -<option> param
```


## Commands

- [`create`](#create /create operator/create piepline)
  - [`create operator`](#create operator)
  - [`create pipeline`](#create pipeline)
- [`init`](#init)
  - [`init operator`](#init operator)
  - [`init pipeline`](#init pipeline)
- [`generate-yaml`](#generate-yaml)
- [`download`](#download)

### `create` / `create operator` / `create piepline`

```bash
$ towheehub create --help
usage: towheehub create  [-h] {operator,pipeline} -a AUTHOR -r REPO -p [PASSWORD]

Create Repo on Towhee hub.

positional arguments:
  {operator,pipeline}   Repo type, choose one from ['operator(default)',
                        'pipeline']

optional arguments:
  -h, --help            show this help message and exit
  -a AUTHOR, --author AUTHOR
                        Author of the Repo.
  -r REPO, --repo REPO  Repo name.
  -p [PASSWORD], --password [PASSWORD]
                        Password of the author.
```

### `init` / `init operator` / `init pipeline`

```bash
$ towheehub init --help
usage: towheehub init [-h] {operator,pipeline} -a AUTHOR -r REPO [-b TAG] [-d DIR]

Initialize the file structure for your Repo.

positional arguments:
  {operator,pipeline}   Repo type, choose one from ['operator(default)',
                        'pipeline']

optional arguments:
  -h, --help            show this help message and exit
  -a AUTHOR, --author AUTHOR
                        Author of the Repo.
  -r REPO, --repo REPO  Repo name.
  -b TAG, --tag TAG     Repo tag or branch, defaults to 'main'.
  -d DIR, --dir DIR     Directory to clone the Repo file.
```

### `generate-yaml`

```bash
$ towheehub generate-yaml --help
usage: towheehub generate-yaml [-h] -a AUTHOR -r REPO

Generate yaml file for your Operator Repo.

optional arguments:
  -h, --help            show this help message and exit
  -a AUTHOR, --author AUTHOR
                        Author of the Repo.
  -r REPO, --repo REPO  Repo name.
```

### `download`

```
$ towheehub download --help
usage: towheehub download [-h] -a AUTHOR -r REPO [-b TAG] [-d DIR]

Clone repo file to local.

optional arguments:
  -h, --help            show this help message and exit
  -a AUTHOR, --author AUTHOR
                        Author of the Repo.
  -r REPO, --repo REPO  Repo name.
  -b TAG, --tag TAG     Repo tag or branch, defaults to 'main'.
  -d DIR, --dir DIR     Directory to clone the Repo file, defaults to '.'.

```

## Examples

### 1. Create your own Repo

- Create Operator

  ```bash
  $ towheehub create operator -a account-name -r your-operator-name -p your-password
  ```
- Create Pipeline
  ```bash
  $ towheehub create pipeline -a account-name -r your-pipeline-name -p
  Password: 
  ```

  > The you can enter the password and it will be hidden.

### 2. Initialize the Repo with template

- Initialize Operator

  ```bash
$ towheehub init operator -a account-name -r your-operator-name
  ```

- Initialize Pipeline

  ```bash
  $ towheehub init pipeline -a account-name -r your-pipeline-name -b my-tag 
  ```

### 3. Download the Repo

- Download Repo to `cwd`

  ```bash
  $ towheehub download -a account-name -r your-repo-name
  ```
- Download Repo to specific directory
  ```bash
  $ towheehub download [-h] -a account-name -r your-repo-name -d /your/workspace/path
  ```
### 4. Generate yaml for Operator Repo
```bash
$ towheehub generate-yaml -a account-name -r your-operator-name
```
