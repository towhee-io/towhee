---
id: pipeline-overview
title: Pipeline overview
---

### Pipeline overview

_Note: if you have not done so, we highly recommend you read our [architecture overview](./architecture-overview.md) section first_

All Towhee Pipelines are represented through a `YAML` file. This file defines how individual Operators are chained together to generate a usable result. Within Towhee, YAML files are loaded and converted into a [graph representation](./DAG-details.md), which is essentially a Python object instance that is easily readable by a Towhee Engine.

Here's an example of a Pipeline YAML, pulled directly from the [Towhee hub](https://towhee.io/towhee/image-embedding-resnet50/src/branch/main/image_embedding_resnet50.yaml).

```
name: 'image-embedding-resnet50'
type: 'image-embedding'
operators:
    -
        name: '_start_op'
        function: '_start_op'
        init_args:
        inputs:
            -
                df: '_start_df'
                name: 'img_path'
                col: 0
        outputs:
            -
                df: 'img_str'
        iter_info:
            type: map
    -
        name: 'image_decoder'
        function: 'towhee/image-decoder'
        tag: 'main'
        init_args:
        inputs:
            -
                df: 'img_str'
                name: 'image_path'
                col: 0
        outputs:
            -
                df: 'image'
        iter_info:
            type: map
    -
        name: 'embedding_model'
        function: 'towhee/resnet-image-embedding'
        tag: 'main'
        init_args:
            model_name: 'resnet50'
        inputs:
            -
                df: 'image'
                name: 'image'
                col: 0
        outputs:
            -
                df: 'embedding'
        iter_info:
            type: map
    -
        name: '_end_op'
        function: '_end_op'
        init_args:
        inputs:
            -
                df: 'embedding'
                name: 'feature_vector'
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
                name: 'img_path'
                vtype: 'str'
    -
        name: 'img_str'
        columns:
            -
                name: 'img_path'
                vtype: 'str'
    -
        name: 'image'
        columns:
            -
                name: 'image'
                vtype: 'towhee.types.Image'
    -
        name: 'embedding'
        columns:
            -
                name: 'feature_vector'
                vtype: 'numpy.ndarray'
    -
        name: '_end_df'
        columns:
            -
                name: 'feature_vector'
                vtype: 'numpy.ndarray'
```

Let's go over this Pipeline file line-by-line.

**Pipeline name**

The first line of a Towhee Pipeline YAML should be the actual name of the Pipeline:

```
name: 'image-embedding-resnet50'
```

If you're Pipeline developer, be sure this Pipeline name does not conflict with other repository names under your username - the Towhee hub tool will automatically use this name as the repository name if you choose to upload your Pipeline to the Towhee hub.

**Pipeline type**

Most Pipelines will associated with an optional `type` (also known as a "category" on the Towhee hub):

```
type: 'image-embedding'
```

In the example above, we have defined the `image-embedding-resnet50` Pipeline to be an `image-embedding` Pipeline. All Pipeline types have pre-defined input and output formats; `image-embedding`'s are `str` (image path) and `numpy.ndarray`, respectively.

While the `type` parameter is optional, we do not recommend leaving it blank, as this will leave the Pipeline's inputs and outputs unconstrained.

**Operator list**

All Operators are classes which inherit Towhee's `Operator` class (neural network Operators should inherit `NNOperator`). Operators within a Pipeline can be specified as follows:

```
operators:
-
    name: 'embedding_model'
    function: 'towhee/resnet-image-embedding'
    tag: 'main'
    init_args:
        model_name: 'resnet50'
```

The corresponding Operator for the above snippet can be found [here](https://towhee.io/towhee/resnet-image-embedding/).

- `name` denotes the name of the Operator - this can be anything so long as it does not conflict with other Operator names in the same Pipeline. We recommend using something descriptive.
- `function` and `tag` specify the local `git` repository and repository tag under `$CACHE_DIR` to load the Operator from; if this repository+tag combination is not found, the engine will automatically attempt to download it from the Towhee hub.
- `init_args` are initialization arguments passed to the Operator upon instantiation. When the Operator is loaded by the engine, these initialization arguments are passed into the `__init__` function of the Operator.

In addition to the Operator name, repository, and initialization parameters, Operators must also receive input data from one or many DataFrames within the Pipeline. The below snippet is an example of a single Operator input:

```
    inputs:
        -
            df: 'image'
            name: 'image'
            col: 0
```

Here, `df` specifies the input DataFrame, `col` is the column index within the DataFrame to extract data frame, and `name` is the Operator's parameter name that `df.col` should map to. For more information on DataFrames, please see the next section.

While an Operator can have any number of inputs, only one output DataFrame is allowed:

```
    outputs:
        -
            df: 'embedding'
```

To maintain compatibility with the `inputs` parameter, the `outputs` parameter is formatted as a YAML list as well (the list should always have only one element). The columns of the DataFrame specified by `outputs[0]['df']` must match the outputs of the Operator.

Note that `_start_op` and `_end_op` are reserved keywords that denote special Operators used by Towhee engines to signify the start and end of a Pipeline; they should not be used anywhere else.

**Dataframe list**

Dataframes are simply data containers in tabular format. Each row of a DataFrame corresponds to a single "line" of data, while columns represent the field:

```
dataframes:
-
    name: 'image'
    columns:
        -
            name: 'image'
            vtype: 'towhee.types.Image'
```

Here, `name` denotes the name of the DataFrame, while `columns` lists each DataFrame field in standard array indexing order. Within `column`, `vtype` represents the field/variable type. This is required by the Engine so that it can perform static type checks prior to running the Pipelines.
