# Architecture part III - Dataframe

Similar to Spark and Pandas DataFrames, a Towhee DataFrame is a two dimensional table of data with a varying number of rows and fixed number of columns.

**Initialization**

Dataframes are created by calls from within a GraphContext. To create a dataframe, the GraphContext needs to pass in a name and the columns for that dataframe. Name comes in the form of a string and the columns come in the form of (str(name), str(type)). In the dataframes current state, there is no type checking of the values. But that will be added in the near future.

Once that info is passed in, the dataframe proceeds to build up empty Towhee Arrays for each column. In addition to the passed in columns, a `_Frame` column is appended as the last column. The df also initializes other variables, including necessary locks and dicts for tracking iterator information and growth.

**Arrays**

Because dataframes are column-based, we use arrays to store the data. Each column is represented by a seperate array. We decided to make our own Array implementation due to the need for garbage collection. We did this by creating an array that keeps tracks of the current offset which can then be used to caclulate the correct position in a list that has been cut shorter due to garbage collection. Our current implementation just remakes a list without the gc'ed values, but we are looking into better/faster implementations.

**Frames**

All rows within Towhee include a `_Frame` column. This column is used track which row the dataframe is on (`row_id`), the row's timestamp (useful when dealing with video data), and other info that helps figure out where the data came from/what it contains. In addition to this, the `_Frame` also has the ability to keep track of the rows previous `row_id`, which is used for keeping order when using multiple runners (for now only filter and map runners are supported).


**Iterators**

Due to the fact that we ultimately want to Towhee to be multiprocessed , we decided to aim for a decoupled iterator-dataframe system early on, hoping that it would be easier to transition to when towhee will be running on different python instances. With this idea, we want to limit the dataframe to the most basic opreations, allowing the development of many different iterators. The iterators currently are in charge of incrementing their location and deciding what data to grab. So far we have included iterators for time_window, map, filter, batch. Time window is used to grab all the values that fall within a range of of row_ids/timestamps, and is incremented by a given step. Map and batch iterators read data one by one from the dataframe, with map being a an instance of batch with a step and count of 1.  


**Insertion**
In this implementation of the dataframe, insertions are limited to dicts and tuples, with multiple inserts being a list of dict/tuples. When adding data, the _frame of the row is first checked to see if it exists, if not, it is added. If the _frame has a prev_id, that means that ordering is required. If the current index does not match the prev_id, then the row is cached for later when the right position opens up. Otherwise, the data is inserted into the end of each column array. Once added in, the len of the dataframe is incremented, and dataframe checks if any blocked iterators had their condition met (waiting for 5 rows, waiting for timestamp > x) or if any of the cached rows can be added in.

**Retrieval**

Data from the dataframe is currently accesible in two ways, get() and get_window().

***get(self, offset, count = 1, iter_id = None)***

get() is used for accessing data at a specific offset. From this offset you can grab however many values you need (used in the case of batch iterators). The iterator_id is used to update and double check trackers on the dataframe side of things.

***get_window(self, start, end, step, comparator, iter_id)***

get_window() is used for accesing windows of data, either by the data's `row_id` or `timestamp`. The start of the window is included, and the end is not included, meaning a call of get_window(0, 5, 5, 'row_id') would return the rows 0, 1, 2, 3, 4.

**Codes**

In order to simplify communication between dataframes and iterators, we decided to use some enum codes to signal the different resulting conditions of accessing data; for example: no more data, block for new data, etc.

**Garbage Collection**

The current renditon of garbage collection works by keeping track of all the offsets of the iterators. With these offsets, we then grab the minimum offset and garbage collect all the data up to that offset.
