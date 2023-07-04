# serving
Through Towhee's `api_service` interface, you can quickly convert the defined pipelines into **http/gRPC** services. Let's take a simple example to see how to use the api_service module
## Define Service

```python
import typing as T
from pydantic import BaseModel
from towhee import AutoPipes, api_service
from towhee.serve.io import JSON, TEXT

# Create a service object
# The / interface will be generated and the string passed in desc will be returned
service = api_service.APIService(desc="Welcome")

# Create sentence_embedding pipeline
stn = AutoPipes.pipeline('sentence_embedding')

# Define a /embedding interface，
# the inputs is TEXT(str)
# the output is ndarray
@service.api(path='/embedding', input_model=TEXT(), output_model=NDARRAY())
def embedding(text: str):
    return stn(text).to_list()

# JSON IO supports defining some complex data structures by using pydantic
class Item(BaseModel):
    url: str
    ids: T.List[int]

@service.api(path='/echo', input_model = JSON(Item), output_model=JSON(Item))
def echo(item: Item):
    return item

```

## HTTP Server
### start server
```python
from towhee.serve.http.server import HTTPServer
HTTPServer(service).run('0.0.0.0', 8000)
```
### Access Server
**Access by Python code**
```python
import requests
from towhee.utils.serializer import from_json

requests.get('http://127.0.0.1:8000').json() 
# Welcome

# The http server converts the ndarray type into a json string,
# It can be converted back through towhee's from_json interface.
from_json(requests.post('http://127.0.0.1:8000/embedding', json='hello world').content)
# ndarray

requests.post('http://127.0.0.1:8000/echo', json={'url': 1, 'ids': [1, 2]}).json()
# {'url': '1', 'ids': [1, 2]}
```
**Access by Curl cmd**
```shell
curl http://127.0.0.1:8000
# Welcome

curl -X POST -H "Content-Type: application/json" -d "hello, towhee" http://127.0.0.1:8000/embedding
# json_str 

curl -X POST -H "Content-Type: application/json" -d '{"url": "http://towhee.io", "ids": [1,2]}' http://127.0.0.1:8000/echo
# {"url": "http://towhee.io", "ids": [1, 2]}
```

## gRPC Server
### Start Server
```python
from towhee.serve.grpc.server import GRPCServer
GRPCServer(service).run('0.0.0.0', 8000)
```
**Access by gRPC Client**
```python
from towhee.serve.grpc.client import Client
c = Client('0.0.0.0', 8000)

# response
#  code: int 0 success
#  msg: str
#  content: Any
respones = c('/echo', {'url': 1, 'ids': [1, 2]})
respones = c('/embedding', 'hello')
c.close()

# or
with Client('0.0.0.0', 8000) as c:
	respones = c('/echo', {'url': 1, 'ids': [1, 2]})
	respones = c('/embedding', 'hello')

```

### Start HTTP and gRPC Server
```python
from towhee.serve.http.server import HTTPServer
from towhee.serve.grpc.server import GRPCServer
s = GRPCServer(service)
s.start('0.0.0.0', 8002)
HTTPServer(service).run('0.0.0.0', 8000)
```

## IO

Currently supports **TEXT**, **BYTES**, **JSON** and **NDARRAY**

#### **TEXT/BYTES**

**TEXT** and **BYTES** correspond to the str and bytes types in Python respectively.

#### **JSON**

The JSON IO of the HTTPServer can support the ndarray type, and the client can use it with the from_json function.

The gRPC Server can only handle some basic types and cannot handle ndarray. You can use the to_json function provided by towhee to convert it into json_str.

Can be used with `pydantic`

#### **NDARRAY**
Returned in Json form in HTTP Server, converted to ndarray by from_json. 
 
gRPC client can return ndarray directly.

Only limited dtypes are supported:

|   |
|---|
| np.object  |  
| np.float32  | 
| np.float64 | 
| np.bool_ | 
| np.int32 | 
| np.int64 | 
| np.uint32 | 
| np.uint64 | 
    
    
    
    
    

