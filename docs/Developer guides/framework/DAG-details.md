---
id: DAG-details
title: Architecture part I - DAG
---

In Towhee, each pipeline is stored as a DAG (directed acyclic graph) in which each node represents an operator. Towhee's engine will execute the operators according to the DAG. Towhee builds a series of representations to illustrate the pipeline.

### Representations

**BaseRepr**

All the representations are inherited from a base representation class called `BaseRepr`, which covers some mutual functions such as load and verification.

Towhee provides several ways of loading the representations. Users can either load the components from a YAML file(both local file and remote file) or a YAML file pre-loaded as a string.

Towhee will automatically check the validity of the given information. All the components have some must-have information. If users want to contribute their own pipelines or operators, please refer to [YAML example](https://hub.towhee.io/towhee/image-embedding-pipeline-template/src/branch/main/image_embedding_pipeline_template.yaml).

**GraphRepr**

`GraphRepr` is the representation of a DAG(pipeline) in Towhee. A graph can be represented by its nodes(operators) and edges(dataframes). Therefore a `GraphRepr` has two attributes `_operators`(a dict of `OperatorRepr`)and `_dataframes`(a dict of `DataFrameRepr`).

When generating a `GraphRepr` from a YAML, Towhee will do some static checks to check the validity of the YAML file, such as loop detection and isolation detection. Note that we do not allow loops or isolations in the graph.

Generating a `GraphrRepr` requires loading the operator and dataframe information from the given YAML file and putting into `_operators` and `_dataframes`.

**OperatorRepr**

An operator is the unit to process the input data in a Towhee pipeline. When creating an

`OperatorRepr`, we need the following information:

- Name: the name of the operator.

- Function: in the form of 'author/operator_name', so Towhee's engine can find the corresponding functional unit either from local cache or Towhee hub.

- Tag: optional, the version of operator to use, since each operator repo might have several branches or tags in Towhee's hub, users can specify the tag to use, otherwise Towhee use 'main' in default.

- Init_args: the arguments to initialize the operator.

- Inputs/outputs: the input and output dataframes, to illustrate how the data flows in the pipeline, note that we need to specify the column and type of input data.

- Iter_info: the way of processing input data.

- Framework: optional, if the operator is neural network related and based on some machine learning framworks, users can specify which framework to use. Towhee uses pytorch by default.

**DataFrameRepr**

The `DataFrameRepr` is the representation of the data that flows in the pipeline. Each dataframe has several columns. Therefore `DataFrameRepr` has one vital attribute `_columns`, which is a list. `_columns`contains the column information including data type. When running a pipeline, users have to make sure all the operators receive and generate the exact type of data as described in the `DataFrameRepr`.

**YAML**

The YAML is the file to illustrate and generate the pipeline. In the current stage of development, pipeline contributors have to write their own YAML files according to our template. However in the coming future, Towhee will provides APIs for users to generate the YAML.

Basically a correct YAML file consists of pipeline name, operators, and dataframes.

- Name: the name of the pipeline, in Towhee, pipelines are distinguished by their authors and names, which means one author cannot create two pipelines with the same name.

- Operators: the information of the operators in this pipeline, as mentioned in `OperatorRepr`. There are two points to pay attention to:

  - All the pipelines should have two operators `_start_op` and `_end_op`. They do not perform any functionality, but are the sign of start and end of the pipeline.

  - Make sure the operators exist either in the local cache or Towhee's hub.

- Dataframes: the information of dataframes in this pipeline, including name, column name, column type.

**YAML Example**

Here is an example of image embedding:

```YAML

name: 'simple_pipeline'
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
                df: 'input_df'
        iter_info:
            type: map
    -
        name: 'add_op1'
        function: 'local/add_operator'
        init_args:
            factor: 1
        inputs:
            -
                df: 'input_df'
                name: 'num'
                col: 0
        outputs:
            -
                df: 'internal_df'
        iter_info:
            type: map
    -
        name: 'add_op2'
        function: 'local/add_operator'
        init_args:
            factor: 2
        inputs:
            -
                df: 'internal_df'
                name: 'num'
                col: 0
        outputs:
            -
                df: 'output_df'
        iter_info:
            type: map
    -
        name: '_end_op'
        function: '_end_op'
        init_args:
        inputs:
            -
                df: 'output_df'
                name: 'sum'
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
        name: 'input_df'
        columns:
            -
                name: 'num'
                vtype: 'int'
    -
        name: 'internal_df'
        columns:
            -
                name: 'sum'
                vtype: 'int'
    -
        name: 'output_df'
        columns:
            -
                name: 'sum'
                vtype: 'int'
    -
        name: '_end_df'
        columns:
            -
                name: 'sum'
                vtype: 'int'
```

![avatar](./dag.png)
