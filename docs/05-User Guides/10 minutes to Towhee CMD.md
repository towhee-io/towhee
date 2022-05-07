# 10 minutes to Towhee CMD

The Towhee command line tool provides some [Towhee Hub](https://towhee.io/) related actions, and can also help you set up your Operators and run your Pipelines, you can [learn more about it](https://github.com/towhee-io/towhee/tree/main/towhee/command). The following show you the common examples of Towhee CMD, so that you can quickly understand.

## Manage your account

First, you need to create an account on the [towhee hub website](https://towhee.io/), then you can log in so that you can use this account do actions such as creating and pushing repositories locally. 

### Login your account

As shown in the command line below, you need to provide your account name and password. Here is an example with the **towhee** account.

```Bash
$ towhee login
Username: towhee
Password: 
Successfully logged in.
```

### Check out who is logged in

```Bash
$ towhee whoami
Username: towhee
```

### Logout your account

If you don't need to log out, please skip this action.

```Bash
$ towhee logout
Done.
```

> If you want to switch to another account in Mac, in addition to running `$towhee logout`, you also need to remove towhee's keychain access. The step-by-step operation is to press `command` + `space` bar to search for `keychain`, then open this application, and find **towhee.io** in the search box, finally delete if there is a record.

## Create operator and develop

Here to create and develop an Operator, such as `test-add`. Not only can you create the operator in the hub with your account, but you can also create it locally.

### Create operator in hub

The following commands will create `test-add` Operator in the hub and initialize the file structure based on the operator name.

- `-t` means the type of Operator, which is divided into “pyop” and “nnop”, “pyop” is the operator of the python function, and "nnop" is the operator of the neural network, and the `test-add` op is belongs to the "pyop".
- `-d` specify the directory to save the repository file, defaults to ".", we specify `test` directory.
- `test-add` is the name of the operator.

```Bash
$ towhee create-op -t pyop -d test test-add

creating repo with username: towhee, and repo name: test-add

Successfully create Operator in hub: https://towhee.io/<your-account>/test-add

Initializing the repo file structure...
```

If you just want to create the operator in the Towhee hub without initializing the file, you can run the above command with `--plain` argument, e.g. `towhee create-op -t pyop -d test test-add --plain` .

### Create operator locally

If you only want to develop locally without creating the Operator on the Towhee hub, you can run with the `--local` argument, and it will not require your Towhee account.

```Bash
$ towhee create-op -t pyop -d test test-add --local

Initializing the repo file structure...
```

### Develop your code

After initializing the operator file, you will see the following files: **test_add.py**, **__init__.py**, **requirements.txt**, which are required by the Towhee operator, more details in "Create your first operator"(TBD). Then modify the code in **test_add.py** as follws:

```Python
import logging
from towhee.operator import PyOperator
from towhee import register

log = logging.getLogger()

@register(output_schema=['result'])
class TestAdd(PyOperator):
    """
    A one line summary of this class.

    Args:
        factor (`int`):
            This argument is summand.
    """
    def __init__(self, factor: int):
        super().__init__()
        self._factor = factor

    def __call__(self, num: int):
        """
        A one line summary of this function.

        Args:
            num (`int`):
                This argument is addend.

        Returns:
            (`int`)
                The sum of two numbers.
        """
        if not isinstance(num, int):
            log.error('ValueError: the addend must be int instead of %s', type(num))
        result = self._factor + num
        return result
```

Here are some developer notes:

- Use `logging` to log information.

- Modify `__init__` and `__call__` functions in the <repo_name> python file, note that please update the Docstrings.

- Modify the `register.output_schema` in the <repo_name>  python file with your own output.

- Update requirements.txt.

- Please update README.md if you want to publish your operator.

## Setup your operator and test

Once you've developed your operator, you'll definitely want to test it. Then the next step is to set up your operator and test it. The setup is divided into two modes: `develop` and `install`. The `install` mode is usually used to setup operators that will not be modified/debugged, and `develop` mode is used to develop code and have the changes take effect immediately.

### Setup with develop mode

- `-p` specify the directory to the operator file, defaults to ".", and its directory contains **<operator_name>.py**, **__init__.py** and **requirements.txt**.
- `--develop` means the setup mode is `develop`.

```Bash
$ towhee install -n towhee -p test/test-add --develop
```

If you want to setup your operator with `install` mode, you can run with `$ towhee install -n towhee -p test/test-add`.

### Test your operator

Then you can run `towhee.ops` to test your operator.

```Shell
$ python
>>> from towhee import ops
>>> op = ops.towhee.test_add(factor=1)
>>> op(1)
2
```

## Create pipeline and modify

Creating a pipeline is the same as creating an operator. It can be created on the hub or locally, and then modify the yaml file of the pipeline.

### Create pipeline in hub

The following commands will create `test-add-pipeline` Pipeline in the hub and initialize the file structure based on the pipeline name.

- `-d` specify the directory to save the repository file, defaults to ".", we specify `test` directory.
- `test-add-pipeline` is the name of the pipeline.

```Bash
$ towhee create-pipeline -d test test-add-pipeline

creating repo with username: towhee, and repo name: test-add-pipeline

Successfully create Operator in hub: https://towhee.io/<your-account>/test-add-pipeline

Initializing the repo file structure...
```

If you just want to create the pipeline in the Towhee hub without initializing the file, you can run the above command with `--plain` argument, e.g. `towhee create-pipeline -t pyop -d test test-add --plain` .

### Create pipeline locally

If you only want to develop locally without creating the pipeline on the Towhee hub, you can run with the `--local` argument, and it will not require your Towhee account.

```Bash
$ towhee create-pipeline -d test test-add-pipeline --local

Initializing the repo file structure...
```

### Modify the YAML

Then you need to modify the **test_add_pipeline.yaml** file as follows, more details in "Create your first pipeline"(TBD).

```YAML
name: 'test-add-pipeline'
type: 'test'
operators:
    -
        name: '_start_op'
        function: '_start_op'
        init_args:
        inputs:
            -
                df: '_start_df'
                name: 'num'
                col: 0
        outputs:
            -
                df: 'addend'
        iter_info:
            type: map
    -
        name: 'test-add'
        function: 'towhee/test-add'
        init_args:
            factor: 1
        inputs:
            -
                df: 'addend'
                name: 'num'
                col: 0
        outputs:
            -
                df: 'sum'
        iter_info:
            type: map
    -
        name: '_end_op'
        function: '_end_op'
        init_args:
        inputs:
            -
                df: 'sum'
                name: 'result'
                col: 0
        outputs:
            -
                df: '_end_df'
        iter_info:
            type: map
dataframes:
    -
        name: '_start_df'
        columns:
            -
                name: 'num'
                vtype: 'int'
    -
        name: 'addend'
        columns:
            -
                name: 'num'
                vtype: 'int'
    -
        name: 'sum'
        columns:
            -
                name: 'result'
                vtype: 'int'
    -
        name: '_end_df'
        columns:
            -
                name: 'result'
                vtype: 'int'
```

## Run your pipeline

You can run your pipeline with the path to the yaml file.

- `-i` specify the input data.
- `test/test-add-pipeline/test_add_pipeline.yaml` is the path to the pipeline.

```Bash
$ towhee run -i 1 test/test-add-pipeline/test_add_pipeline.yaml
2
```