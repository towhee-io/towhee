---
id: pipeline-overview
title: Pipeline overview
---

### Pipeline overview

All Towhee pipelines are represented through a `YAML` file. This file defines how individual operators are chained together to generate a usable result. Within Towhee, YAML files are loaded and converted into a [graph representation](./DAG-details.md), which is essentially a Python object instance that is easily readable by Towhee engines.

Here's an example of a pipeline YAML, pulled directly from the [Towhee hub](https://towhee.io/towhee/image-embedding-resnet50/src/branch/main/image_embedding_resnet50.yaml).

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

Let's go over this pipeline file line-by-line.

**Pipeline name**

The first line of a Towhee pipeline YAML should be the actual name of the pipeline:

```
name: 'image-embedding-resnet50'
```

If you're Towhee pipeline developer, be sure this pipeline name does not conflict with other repository names under your username - the Towhee hub tool will automatically use this name as the repository name if you choose to upload your pipeline to the Towhee hub.

**Pipeline type**

All pipelines should be associated with a type (also known as a "category" on the Towhee hub):

```
type: 'image-embedding'
```

In the example above, we have defined the `image-embedding-resnet50` pipeline to be an `image-embedding` pipeline. All pipeline types have pre-defined input and output formats; `image-embedding`'s are `str` (image path) and `numpy.ndarray`, respectively.

**Operator list**

Operators are not limited to neural networks; traditional ML models, image processing algorithms, or even simple Python scripts can all be packaged as Towhee operators and provided to users via the Towhee hub. In this sense, operators are simply transformations on a set of input data.

All operators are classes which inherit Towhee's `Operator` class (neural network operators should inherit `NNOperator`). Operators within a pipeline can be specified as follows:

```
operators:
-
    name: 'embedding_model'
    function: 'towhee/resnet-image-embedding'
    tag: 'main'
    init_args:
        model_name: 'resnet50'
```

The corresponding operator for the above snippet can be found [here](https://towhee.io/towhee/resnet-image-embedding/).

- `name` denotes the name of the operator - this can be anything so long as it does not conflict with other operator names in the same pipeline. We recommend using something descriptive.
- `function` and `tag` specify the local `git` repository and repository tag under `$CACHE_DIR` to load the operator from; if this repository+tag combination is not found, the engine will automatically attempt to download it from the Towhee hub.
- `init_args` are initialization arguments passed to the operator upon instantiation. When the operator is loaded by the engine, these initialization arguments are passed into the `__init__` function of the operator.

In addition to the operator name, repository, and initialization parameters, operators must also receive input data from one or many dataframes within the pipeline. The below snippet is an example of a single operator input:

```
    inputs:
        -
            df: 'image'
            name: 'image'
            col: 0
```

Here, `df` specifies the input dataframe, `col` is the column index within the dataframe to extract data frame, and `name` is the operator's parameter name that `df.col` should map to. For more information on dataframes, please see the next section.

While an operator can have any number of inputs, only one output dataframe is allowed:

```
    outputs:
        -
            df: 'embedding'
```

To maintain compatibility with the `inputs` parameter, the `outputs` parameter is formatted as a YAML list as well (the list should always have only one element). The columns of the dataframe specified by `outputs[0]['df']` must match the outputs of the operator.

Note that `_start_op` and `_end_op` are reserved keywords that denote special operators used by Towhee engines to signify the start and end of a pipeline; they should not be used anywhere else.

**Dataframe list**

Dataframes are simply data containers in tabular format. Each row of a dataframe corresponds to a single "line" of data, while columns represent the field:

```
dataframes:
-
    name: 'image'
    columns:
        -
            name: 'image'
            vtype: 'towhee.types.Image'
```

Here, `name` denotes the name of the dataframe, while `columns` lists each dataframe field in standard array indexing order. Within `column`, `vtype` represents the field/variable type. This is required by the Engine so that it can perform static type checks prior to running the pipelines.
