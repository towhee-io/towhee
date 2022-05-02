# Installation

### Installation requirements

Towhee has been tested with Python 3.6+ on macOS 10, macOS 11 (Intel CPU only), Ubuntu 18.04, Ubuntu 20.04, and Windows 10. Each individual operator may have its own specific requirements for Pytorch and Tensorflow, but in general, Pytorch 1.2+ and Tensorflow 2.0+ should work for the majority of model-based operators.

### Installing Towhee with Conda

If you're a conda user, you can install Towhee with the following lines:

```shell
$ conda create -n towhee_env python=3.6  # create a conda environment for Towhee
$ conda activate towhee_env  # activate your newly created conda environment
$ conda install -c towhee-io towhee   # install Towhee via the `towhee-io` channel
```

### Install Towhee with pip

As with `conda`, we highly recommend first activating a [virtual environment](https://docs.python.org/3/library/venv.html) to avoid system-level dependency issues. If you are unfamiliar with virtual environments, check out this [guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) first.

```shell
$ python3 -m venv towhee_env  # create a virtual environment
$ source towhee_env/bin/activate  # activate your newly created virtual environment
$ pip3 install -U pip  # we recommend updating pip first before installing Towhee
$ pip3 install -U towhee  # install Towhee
```

Alternatively, if you are on Windows, install Python 3.6+ first, either through the Windows Store or through one of the [official releases](https://www.python.org/downloads/windows). You can then install Towhee as follows:

```console
c:\> python -m venv towhee_env
c:\> source towhee_env/Scripts/Activate
c:\> pip3 install -U towhee
```

### Install Towhee from source

As with conda and pip, we again strongly recommend first activating a virtual environment:

```shell
$ python3 -m venv towhee_env
$ source towhee_env/bin/activate
```

After activating the virtual enviroment, run the following commands to install Towhee.

```shell
$ git clone https://github.com/towhee-io/towhee.git
$ cd towhee
$ python setup.py install
```

Please note that the `main` branch is still being tested and may have residual bugs and/or unimplemented features. We can always use a helping hand; if you encounter any issues, please [let us know](https://github.com/towhee-io/towhee/issues/new/choose) on Github. Contributions are always welcome!
