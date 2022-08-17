# Towhee Command Line Tool

## Installation 

**(Recommended)** Once you have [installed Towhee](https://docs.towhee.io/get-started/install), you can use `towhee` with the following command:

> You can use `python -m towhee` instead of `towhee` to trobleshot.

````bash
$ pip3 inatsll towhee
$ towhee <command> -<option> param
# OR
$ python -m towhee <command> -<option> param
````

Of course, you can also run `towhee` with source code:

```bash
$ git clone https://github.com/towhee-io/towhee.git
$ python3 setup.py install
$ towhee <command> -<option> param
# OR
$ python -m towhee <command> -<option> param
```


## Usage

- user config
  - [`login`](#login)
  - [`logout`](#logout)
  - [`whoami`](#whoami)
- create operator
  - [`create`](#create)
- setup operator
  - [`install operator`](#install-operator)
  - [`uninstall operator`](#uninstall-operator)
- execute
  - [`run`](#run-pipeline)


### User Config
#### `login`
```bash
$ towhee login -h
usage: towhee login [-h]

optional arguments:
  -h, --help  show this help message and exit
```
#### `logout`
```bash
$ towhee logout -h
usage: towhee logout [-h]

optional arguments:
  -h, --help  show this help message and exit
```
#### `whoami`
```bash
$ towhee whoami -h
usage: towhee whoami [-h]

optional arguments:
  -h, --help  show this help message and exit
```

### Create Operator
#### `create`
```bash
$  towhee create -h
usage: towhee create-op [-h] [-t {pyop,nnop}] [-f FRAMEWORK] [-d DIR] [--local] [--plain] uri

positional arguments:
  uri                   Repo uri, such as author/repo-name or repo-name(author defaults to login account).

optional arguments:
  -h, --help            show this help message and exit
  -t {pyop,nnop}, --type {pyop,nnop}
                        optional, operator repo type in ['pyop', 'nnop'] for init file, defaults to 'nnop'
  -f FRAMEWORK, --framework FRAMEWORK
                        optional, framework of nnoperator, defaults to 'pytorch'
  -d DIR, --dir DIR     optional, directory to the Repo file, defaults to '.'
  --local               optional, create and init repo in local
  --plain               optional, just create repo with init file
```

### Setup Operator
#### `install`
```bash
$ towhee install -h        
towhee install -h
usage: towhee install [-h] [-n NAMESPACE] [-p PATH] [--develop]

optional arguments:
  -h, --help            show this help message and exit
  -n NAMESPACE, --namespace NAMESPACE
                        optional, repo author or namespace, defaults to 'towhee'
  -p PATH, --path PATH  optional, path to the operator repo, defaults to cwd which is '.'
  --develop             optional, install operator with setup.py develop
```
#### `uninstall`
```bash
$ towhee uninstall -h
usage: towhee uninstall [-h] [-n NAMESPACE] [-p PATH]

optional arguments:
  -h, --help            show this help message and exit
  -n NAMESPACE, --namespace NAMESPACE
                        optional, repo author or namespace, defaults to 'towhee'
  -p PATH, --path PATH  optional, path to the operator repo, defaults to cwd which is '.'
```

### Execute
#### `run`
```bash
$ towhee run -h      
usage: towhee run [-h] -i INPUT [-o OUTPUT] pipeline

positional arguments:
  pipeline              pipeline repo or path to yaml

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input the parameter for pipeline defaults to None
  -o OUTPUT, --output OUTPUT
                        optional, path to the file that will be used to write results], defaults to None which will print the result
```


## Examples

### 1. User Config

- **Login**

Please create an account at https://towhee.io/ before logging in. Then you need to enter username and password.

```bash
$ towhee login
Username: <enter-user-name>
Password:
```
- **Whoami**

Check your logged in towhee.io account.

```bash
$ towhee whoami
Username: <certified-user-name>
```

- **Logout**

 Logout your towhee.io account.

```bash
$ towhee logout
```

### 2. Create Operator

- **Create Operator in Towhee hub**

Create your own operator named <repo-name> in the [Towhee hub](https://towhee.io/operators) using the logged in account. It will also initialize the file structure for your operator to the current working directory, which will clone the repository and initialize it according to the default "nnop" operator type template.

> The operator types are divided into “pyop” and “nnop”, “pyop” is the operator of the python function, and nnop is the operator of the neural network.

```bash
$ towhee create <repo-name>
```

Create operator and initialize it to a specific directory using the 'pyop' operator type template.

```bash
$ towhee create -t pyop -d <path/to/your/dir> <repo-name>
```
Create operator and initialize it with the 'nnop' operator type template, also specify the framework.

```bash
$ towhee create -t nnop -f <my-framework> <repo-name>
```
Only create operator in hub without initializing the files.

```bash
$ towhee create <repo-name> --plain
```

- **Create Operator in local**

When you add `--local` to the command, it will not create the operator in the hub, but just initializes the file structure in local.

```bash
$ towhee create-op <repo-name> --local
```

### 3. Setup your Operator

- **Setup develop**

When you are developing the code, you can run setup under your operator folder.

> It will generate symbolic link to the python package of towheeoperator.

```bash
$ towhee install --develop
```

Setup the operator under your repository path with a specific namespace.

```bash
$ towhee install -n <namespace> -p <path/to/your/op> --develop
```

- **Setup install**

When you have finished the code, you can run it to setup python package under your operator folder.

> It will setup towheeoperator.[repo] python package.

```bash
$ towhee install
```

Install the operator under your repository path with a specific namespace.

```bash
$ towhee install -n <namespace> -p <path/to/your/op>
```

### 4. Run

You can run the pipeline with your specific yaml file.

```bash
$ towhee run <path/to/your/yaml/file>
```

Also you can run pipeline in hub.

```bash
$ towhee run towhee/image-embedding-resnet50
```

