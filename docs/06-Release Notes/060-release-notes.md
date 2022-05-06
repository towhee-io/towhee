# v0.6.0 Release Notes

#### Highlights

- A new programming API, `DataCollection`, is released in this version. The users can build their own unstructured data processing pipeline and application with a pythonic,  method-chaining style DSL, based on `DataCollection`. Please check details in our [API document](https://towhee.readthedocs.io/en/main/data_collection/get_started.html);
- Towhee now provides a decorator `towhee.register` to help the users register their own  code as an operator, and cooperate their code with the operators from towhee hub. The decorator also simplifies operator development. Please check our examples in the API document: [towhee ‒ Towhee v0.6.0 documentation](https://towhee.readthedocs.io/en/main/towhee/towhee.html);
- `towhee.train` now integrate captum to explain model;

#### `towhee`

- The DataCollection API:
  - Core API #659: #727
  - Parallel execution: #676 #705 #837
  - DAG execution: #1024
  - Experimental ray backend: #933 #947
  - Mixins: #750 #834 #929 #961 #970 #1026
- The decorator for register operator: #897 #920
- Support for entity class: #799 #868 #998 #1035

#### `towhee.models`

towhee.models now is a separate python package , which makes it much easier to be upgraded independently.
- plot utils during training: #841 #842
- integrate captum to explain model: #854
- fix some training bugs: #780 #940
