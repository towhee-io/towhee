---
id: column-based-benchmark-report
title: column based benchmark report
authors:
  - name: Kaiyuan Hu
    url: https://github.com/Chiiizzzy
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Column-based Benchmark Report

In this Section we will report benchmarks we ran for row-based computing and column-based computing, so as to explain the advantage and necessity of introducing column-based storage.

## Speed

- First we compare the speed of comparing values in two columns (with 100000 rows) using row-based DataCollection operator and column-based pyarrow built-in operator separately.

```{code-cell} ipython3
---
tags: [hide-cell]
---
import time
import matplotlib.pyplot as plt
import pyarrow.compute as pc
from towhee import Entity, DataFrame

e = [Entity(a=a, b=b) for a,b in zip(range(100000),range(100000))]
df = DataFrame(e)
time_1 = time.process_time()
df.runas_op[('a', 'b'), 'c'](func = lambda x, y : x == y)
time_2 = time.process_time()
df.to_column()
time_3 = time.process_time()
df._iterable.append_column('d', pc.equal(df._iterable['a'], df._iterable['b']))
time_4 = time.process_time()
x = ['row_computing', 'convert_to_col', 'col_computing']
y = [time_2 - time_1, time_3 - time_2, time_4 - time_3]
```

```{code-cell} ipython3
plt.title('Column Comparing Result Analysis')
plt.plot(x, y)
plt.xlabel('action')
plt.ylabel('time(s)')
plt.show()
```

- Secondly we compare matrix matmul, with matrix size (10000, 2, 5), (100000, 2, 5), (1000000, 2, 5). In this step we want to investigate how different chunk sizes affect computational speed so we run matmul in different chunk sizes.

```{code-cell} ipython3
---
tags: [hide-cell]
---
import time
import matplotlib.pyplot as plt
import numpy as np
import towhee
from towhee.types.tensor_array import TensorArray

x = [100000, 1000000, 10000000]
chunk_s = []
chunk_l = []
unchunk = []
row_dc = []
for i in [100000, 1000000, 10000000]:
	arr = TensorArray.from_numpy(np.arange(i).reshape([-1,2,5]))
	dc = towhee.dc([x for x in np.arange(i).reshape([-1,2,5])])
	trans = np.random.random([5,2])

	time_1 = time.process_time()
	[np.matmul(a, trans) for a in arr.chunks(20)]
	time_2 = time.process_time()
	[np.matmul(a, trans) for a in arr.chunks(50)]
	time_3 = time.process_time()
	[np.matmul(a, trans) for a in arr]
	time_4 = time.process_time()
	[np.matmul(a, trans) for a in dc]
	time_5 = time.process_time()
	chunk_s.append(time_2 - time_1)
	chunk_l.append(time_3 - time_2)
	unchunk.append(time_4 - time_3)
	row_dc.append(time_5 - time_4)
```

```{code-cell} ipython3
plt.title('Matrix matmul Result Analysis')
plt.xscale('log')
plt.yscale('log')
plt.plot(x, chunk_s, color='green', label='chunk size 20')
plt.plot(x, chunk_l, color='red', label='chunk size 50')
plt.plot(x, unchunk,  color='skyblue', label='unchunk')
plt.plot(x, row_dc, color='blue', label='row dc')
plt.legend()
plt.xlabel('size')
plt.ylabel('time(s)')
plt.show()
```

- Then we will test some tensor related operators, including tensor reshape and tensor matmul, with row-based computing operator and col-based computing operator.

```{code-cell} ipython3
---
tags: [hide-cell]
---
import time
import matplotlib.pyplot as plt
import numpy as np
from towhee import DataFrame, Entity

df = DataFrame([Entity(a = np.ones([1, 2])) for _ in range(100000)])
time_1 = time.process_time()
df = df.tensor_reshape['a', 'b'](shape = [2, 1])
time_2 = time.process_time()
df.to_column()
time_3 = time.process_time()
df = df.tensor_reshape['a', 'c'](shape = [2, 1])
time_4 = time.process_time()

x = ['row_computing', 'convert_to_col', 'col_computing']
y = [time_2 - time_1, time_3 - time_2, time_4 - time_3]
```

```{code-cell} ipython3
plt.title('Tensor Reshape Result Analysis')
plt.plot(x, y)
plt.xlabel('action')
plt.ylabel('time(s)')
plt.show()
```

```{code-cell} ipython3
---
tags: [hide-cell]
---
import time
import matplotlib.pyplot as plt
import numpy as np
from towhee import DataFrame, Entity
from towhee.types.tensor_array import TensorArray

df = DataFrame([Entity(a = np.ones([2, 1]), b = np.ones([1, 2])) for _ in range(100000)])
time_1 = time.process_time()
df = df.tensor_matmul[('a', 'b'), 'c']()
time_2 = time.process_time()
df.to_column()
time_3 = time.process_time()
df = df.tensor_matmul[('a', 'b'), 'd']()
time_4 = time.process_time()

x = ['row_computing', 'convert_to_col', 'col_computing']
y = [time_2 - time_1, time_3 - time_2, time_4 - time_3]
```

```{code-cell} ipython3
plt.title('Tensor Matmul Result Analysis')
plt.plot(x, y)
plt.xlabel('action')
plt.ylabel('time(s)')
plt.show()
```