---
id: architectural-overview
title: Architecture overview
---

### Basic components

##### Operator

The Operator forms the basic building block for Towhee. Each operator is considered a single _transformation_, or unit of work, on the input data.

##### Pipeline

Towhee uses the YAML file to define a pipeline, the pipeline is a DAG. The DAG only can have one input and one output, we use internal operators(`_start_op` and `_end_op`). The DAG can branch and merge. **DataFrame** The `DataFrame` is the edge in a DAG. It is used to transport the data between operators. **Operator** Calculation module of a pipeline. Consume previous dataframe's data, calculate new data, and put it into the next dataframe.

##### DAG

##### Engine
