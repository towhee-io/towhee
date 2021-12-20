# Towhee Style Guide

## **Coding Style**
Towhee's coding style is based on PEP8 and PEP257.

### **Maximum Line Length**
- The limit for docstring is 88 characters for each line.
- The limit for code and comment is 150 characters for each line.

### **Indentation and Split**
Use 4 whitespaces per indentation level.
```python
def foo(
	arg_1,
	arg_2,
	arg_3,
	arg_4,
	...
):
	...

bar = (
    part_1
    + part_2
    + (part_3 - part4)
    - part_5
    - part_6
)
```
*Notes*:
- Add a level of indent to clearly distinguish hanging indents.
- Split before the first arguemnt in hanging indents.
- Start a new line for the closing parenthesis, brackets, braces.
- Dedent the closing parenthesis, brackets, braces.
- Split a long line before operators and after commas.

### **Blank Lines**
- Leave two blank lines between:
	- top-level functions and classes.
- Leave One blank line between
	- functions defined inside classes.
	- logical sections inside a function.
- Leave one blank line at the end of a python file
```python
class Foo():
    def foo_1():
        ...

    def foo_2():
        ...


def foo():
    # This is the first logical part.
    ...

    # This is the second logical part.
    ...


def bar():
    ...

```

### **Imports**
Place Imports at the top of the file, right after module comments and docstrings, before globals and constants.
```python
import std_lib_A
import std_lib_B

import third_party_lib_A
import third_party_lib_B

from std_lib_C import object_a, object_b
from std_lib_D import (
    object_c,
	object_d,
	object_e,
	object_f,
    object_g,
	object_h,
	object_i,
	object_j
)
```
*Notes*:
- Organize the imports in following order:
	1. Standard library imports
	2. Related third party imports
	3. Local application/library specific imports
- Import one library in per line.
- It's ok to import several objects from one library in one line, but when the number of objects is more the 7, use hanging indents.  
- Leave one blank line between different groups.
- Use absolute imports, avoid wildcard imports:
```python
# It's best to use absolute imports.
import some_module.some_object
from some_module import some_object

# Avoid explicit relative imports if possible.
from . import sibling_module
from .sibling_module import sibling_obejct

# Avoid using wildcard imports.
from some_module import *
```

### **Whitespace**
#### **Avoid unnecessary whitespace in the following cases:**
- Immediately after or before parentheses, brackets or braces:
```python
# Correct:
example(list[1], {key: 2})
dct['key'] = lst[index]

# Wrong:
example ( list[ 1 ], { key: 2 } )
dct [ 'key' ] = lst [ index ]
```
- Between a trailing comma and a following close parenthesis:
```python
# Correct:
foo = (0,)

# Wrong:
bar = (0, )
```
- Immediately before a comma, semicolon, or colon, but leave a space after them unless followed by parentheses, brackets or braces:
```python
# Correct:
if x == 4:
    print x, y; x, y = y, x

# Wrong:
if x == 4 :
    print x , y ;x , y = y , x
```
- The end of a line.

#### **Do use whitespace in following cases:**
- Surround following operators with a single space on either side:
	- assignment (=),
	- augmented assignment (+=, -=, etc.)
	- comparisons (==, <, >, !=, <>, <=, >=, in, not in, is, is not)
	- booleans (and, or, not).
- Surround operators with lowest priority with a single space on either side:
```python
# Correct:
i = i + 1

i += 1

x = x*2 - 1

hypot2 = x*x + y*y

c = (a+b) * (a-b)

if a and b:
    ...

if a == b:
    ...

for i in range(10):
    ...
```
- Function annotations, after : and surround -> :
```python
# Correct:
def foo(bar: str) -> int:
    ...
```

### **Annotation**
Add annotation for all functions.
```python
from Typing import Set, Dict, List, Tuple, Any

def foo(a: Set(str), b: List[int], c: Dict[str, Any]) -> Tuple(int, str):
    ...
    return return_int, return_str
```
*Notes*:
- Use Set, Dict, List, Tuple instead of set, dict, list, tuple in annotation, also specify the data type inside.

*Special case*:
- If we want to return a Foo object inside of the Foo class, use string to annotate the return type:
```python
Class Foo():
	...

    @classmethod
    def initializera(args) -> 'Foo':
        ...
        return Foo_instance
```

### **Comments and Docstring**
#### **Comments**
- All comments and docstrings should be composed of sentences, not phrases:
```python
# Correct:   
# My code does this.

# Wrong:  
# my code does this
```
- Use block comments rather than inline comments. Block comments should apply to some (or all) code that follows them, and are supposed to indented to the same level as the code they are applied.
```python
# Correct:
# The Following code does this.
...

#Worng:
... # The code does this
```
**Docstring**
- Write dostring for every function and class according to the following template, unless the function or the class is:
	- Externally invisible
	- Very short
	- Obvious

Template：
```python
"""
Brief introduction within one line.

Detailed description, paragraph 1...

Detailed description, paragraph 2 ...

Args:
    arg0 (`int`):
        arg0 description.
    arg1 (`Union[float, str]`):
        arg1 description.

Returns:
    (`Tuple[bool, int]`)
        Return value description.

Raises:
    (`xxxError`)
        Raise xxxError when ....
"""
```
*Notes*:
- Start a new line for a brief introduction, do not add an introduction right after the opening """.
- Use `` (the symbol in the left of 1), not '' when declare the types.
- If needed, add some detailed description after the introduction separated by one blank line.
- If there is more than one paragraph in the detailed description, separate them with one blank line.
- If a function does not have Args, Returns or Raises, do not add them in the docstring.
- In Args, add a colon (':') after (\`type\`), leave whitespace before (\`type\`) .
- In Returns, Raises, only list the return type and error types, as (\`return/error type\`), no colon(':') needed.
- Put the docstrings for a class's `__init__` function at the beginning of the class definition:
```python
class Foo():
    """
    Introduction to the Foo class...

	Detailed description...

    Args:
        args_0 (`type`):
        ...
    """
    def __init__(args):
        ...
```

A detailed exmaple:
```python
def __init__(
    arg_1: int, arg_2: list, arg_3: dict,
    arg_4: float, arg_5: bool, arg_6: tuple
) -> Tuple[int, float, str]:
"""
A one line summary.

This is a complicated function so we need multi-line docstring, this function does something.

Functions that do not have return values, raise errors, or need examples can omit the sections below.

Args:
    arg_1 (`str`):
        ...
    arg_2 (`List[float]`):
        ...
    arg_3 (`Dict[int, str]`):
        ...
    ...

Returns:
    (`Tuple[int, float, str]`)
        A tuple with three values, first being int...

Raises:
    (`IOError`)
        Throw an IOError when...
"""
```

*Special case*：
- Return a Foo object inside Foo class:
```python
Class Foo():
    @classmethod
    def generate(args) -> 'Foo':
        """
        Short intro.

        Detailed Description.

        Args:
            ...

        Returns:
            (`path/to/Foo`)
                Returns a Foo instance.
        """
        return Foo_instance
```

### **Naming Conventions**
- Variable and function names should be lowercase and connected with underscore if necessary:
```python
foo = 1
def foo_bar() -> None:
	...
```
- Uppercase for Constants:
```python
MY_CONSTANT
```
- CapWords for class name:
```python
class MyClass(object):
	...
```
- Use one leading underscore only for non-public methods and instance variables:
```python
def _private_function() -> None:
	...
```
