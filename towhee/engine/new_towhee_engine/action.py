from towhee.dataframe.array.array import Array

class Action:
    def __init__(self, outline):
        """Create an action.

        Actions are one step of the dag. Each action contains:
        1. An iterator that can read from one or many action outputs.
        2. One operator which is the task being performed
        3. One or many towhee.Arrays to store output for each schema output.
        """ 
        self._outline = outline
        self._iterator = Iterator(outline['iterator']) # select the type of iterator, and which array it reads
        self._op = towhee.op(outline['op']) # current op dispatching
        self._arrays = self.array_creation() # multiple arrays for multiple schema output

    def array_creation():
        temp = {}
        for x in self._outline['outputs']:
            temp[x['name']] = Array(readers=len(x['readers']))
    
    def run(self):
        """
        NEEDS WORK
    
        Iterate through the iterator and perform operation on values. When value is returned, append it to the towhee.array.
        This then triggers all the iterators of that array to yield their values for the next step. I believe this logic works, 
        but might need more work.
        

        """
        for x in self._iterator:
            self._array.append(self.op(x)):
            self._array.call_yields()
