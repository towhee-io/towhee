---
id: cmdline
title: Command Line Tool
---

### About command-line tool

Towhee provides a command-line tool called `towhee`, which helps users and developers to develop and test a pipeline or operator.

### For developers

#### create an operator/pipeline

To start a new operator, you can prepare initial code with the following command:

```shell
$ towhee template --type operator --name new_operator # or -t operator -n new_operator
```

This command creates an empty operator called `new_operator` in the directory with the same name.

#### debugging development

In order to test the new operator, we need to create an `egg-link` file in the `site-packages` directory. The file helps the python interrupter to find our code when importing `new_operator`.

```shell
$ towhee develop --develop ./new_operator
```
where `./new_operator` is the path for the operator's  source code.

#### install an operator/pipeline

Operators/pipelines can be installed into python environment with the following command:
```shell 
$ towhee develop -i ./new_operator
```

#### packaging 

```shell
$ towhee develop -p ./new_operator
```

### For Users

To run a pipeline from command line

```shell
$ towhee execute -n pipeline_name -D variable=value -D variable=value

```