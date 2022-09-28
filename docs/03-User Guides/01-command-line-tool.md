# 10 minutes to Towhee CMD

The Towhee command line tool provides some [Towhee Hub](https://towhee.io/) related actions, and can also help you set up your Operators, you can [learn more about it](https://github.com/towhee-io/towhee/tree/main/towhee/command). The following shows you the common examples of towhee cmd, so that you can quickly understand.

## Manage your account

First, you can create an account on the [towhee hub website](https://towhee.io/), then you can log in so that you can use this account to do actions such as creating and pushing repositories locally. 

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

> If you want to switch to another account on Mac, in addition to running `$towhee logout`, you also need to remove towhee's keychain access. 
>
> The step-by-step operation is to press `command` + `space` bar to search for `keychain`, then open this application, and find **towhee.io** in the search box, finally delete if there is a record.

## Create operator and develop

Here to create and develop an Operator, such as `test-add`. Not only can you create the operator in the hub with your account, but you can also create it locally.

### Create operator in hub

The following commands will create `test-add` Operator in the hub and initialize the file structure based on the operator name.

- `-t` means the type of Operator, which is divided into “pyop” and “nnop”, “pyop” is the operator of the python function, "nnop" is the operator of the neural network, and the `test-add` op belongs to the "pyop". 

- `-d` specifies the directory to save the repository file, defaults to ".", we specify `test` directory. 

- `test-add` is the name of the operator. 

```Bash
$ towhee create -t pyop -d test test-add
creating repo with username: towhee, and repo name: test-add

Successfully create Operator in hub: https://towhee.io/<your-account>/test-add

Initializing the repo file structure...
```

If you just want to create the operator in the towhee hub without initializing the file, you can run the above command with `--plain` argument, e.g. `towhee create-op -t pyop -d test test-add --plain`.

### Create operator locally

If you only want to develop locally without creating the Operator on the Towhee hub, you can run with the `--local` argument.

```Bash
$ towhee create -t pyop -d test test-add --local
 
Initializing the repo file structure...
```

### Develop your code

After initializing the operator file, you will see the following files: **test_add.py**, **__init__.py**, **requirements.txt**, which are required by the towhee operator, and more details in "How to develop my first Operator"(TBD). Then modify the code in **test_add.py** as follows:

```Python
from towhee.operator import PyOperator, SharedType


class TestAdd(PyOperator):
    """
    Simple addition.

    Args:
        factor (`int`):
            This argument is summand.
    """
    def __init__(self, factor: int):
        super().__init__()
        self._factor = factor

    def __call__(self, num: int):
        """
        Add num and factor.

        Args:
            num (`int`):
                This argument is added.

        Returns:
            (`int`)
                The sum of two numbers.
        """
        result = self._factor + num
        return result

    @property
    def shared_type(self):
        return SharedType.Shareable

    def input_schema(self):
        return [(int, (1,))]

    def output_schema(self):
        return [(int, (1,))]
```

Here are some developer notes:

- Modify `__init__` and `__call__` functions in the **<repo_name> python file(test_add.py)**, note that please update the Docstring. For the DataType and DataShape of the input and output schema, please refer to the following format:

  | **DataType**                       | **DataShape**  |
  | ---------------------------------- | -------------- |
  | int                                | (1, )          |
  | str                                | (1, )          |
  | float                              | (1, )          |
  | Image                              | Shape of image |
  | AudioFrame                         | (1024, )       |
  | ImageFrame                         | Shape of image |
  | Numpy type: np.float, np.int32 ... | Shape of numpy |

- Update **requirements.txt**. 

- Please update **README.md** if you want to publish your operator. 

## Setup your operator and test

Once you've developed your operator, you'll definitely want to test it. The next step is to set up your operator and test it. The setup is divided into two modes: `develop` and `install`. The `install` mode is usually used to setup operators that will not be modified/debugged, and `develop` mode is used to develop code and have the changes take effect immediately.

### Setup with develop mode

- `-p` specifies the directory to the operator file, defaults to ".", and its directory contains **<operator_name>.py**, **__init__.py**, and **requirements.txt**. 

- `--develop` means the setup mode is `develop`. 

```Bash
$ towhee install -n towhee -p test/test-add --develop
```

If you want to setup your operator with `install` mode, you can run with `$ towhee install -n towhee -p test/test-add`.

### Test your operator

Then you can run `towhee.ops` to test this operator.

```Shell
$ python
>>> from towhee import ops
>>> op = ops.towhee.test_add(factor=1)
>>> op(1)
2
```

And you can also run with `towhee.dc`:

```shell
>>> import towhee
>>> towhee.dc([1,3]).towhee.test_add(factor=1).to_list()
[2, 4]
```
