"""
Big Data Processing (For Electron Microscopy)
=============================================

This example is completely hyperspy/pyxem independent and is meant to show ``dask`` is used/ can be used.

What is Lazy Processing?
------------------------
A good way to think of lazy processing is: "let's try to build an effective plan to process my
dataset from start to finish".... Without actually processing my data over and over again.

To do that you just make a nice list or plan of things to do:

Or Better yet have the computer make that list for you!  This is what dask calls a
"Task Graph" and you can see the graphs (if they are small... Hint: they will not be) by using:

```python
s.data.visualize()
```

Let's start by looking at a comparison between how ``dask`` and ``numpy`` work.
"""

import numpy as np
import dask.array as da
from dask import optimize
from dask_image.ndfilters import gaussian_filter

# make both an in (and out of memory
in_memory_array = np.ones((10, 10, 10, 10))  # small array that fits in memory
out_memory_array = da.ones(
    (50, 100, 100, 100)
)  # large array that does not fit in memory

print(in_memory_array)

out_memory_array

# %%
# Visualizing the Task Graph
# --------------------------
# Dask also gives you the ability to visualize the task graph, which is a great way to see
# how the computation will be performed.  This can be done using the ``visualize`` method.
# Here each rectangle represents an underlying chunk of data, the arrows represent the dependencies
# between tasks and the circles represent the functions themselves. Here the function ``ones_like``
# from numpy is used to create an array of ones.

out_memory_array.visualize()

# %%
# An Embarrassingly Parallel Graph
# --------------------------------
# This is an embarrassingly parallel task graph.  This doesn't mean you should be embarrassed
# it just means that the computer should be embarrassed if it isn't processing this in parallel.
# When you have straight lines everything is easy, nothing overlaps, and you can give each separate
# task to one core, and it is happy!
#
# Let's try to slice this array and see what happens.  This is a common operation when you might want
# to look at a subset of your data.

slice_arr = out_memory_array[:, :, 10, 10]
slice_arr.visualize()

# %%
# What you can see is a lot of the functions terminate early. Under the hood, dask is smart enough to
# know that it doesn't need to compute the entire array to get the slice. Before you do any computation
# with the array dask will call the ``optimize`` function to prune the task graph.
slice_arr_opt = optimize(slice_arr)[0]
slice_arr_opt.visualize()

# %%
# Now the task graph is much smaller, and it only contains the tasks that are necessary to compute the slice.
# Dask is also smart enough to reuse existing chunks and computations. Let's say we wanted to add our
# slice to the original array...

addition = out_memory_array + slice_arr[:, :, np.newaxis, np.newaxis]
addition.visualize()

# %%
# Now the task graph is much larger. The two task graphs for out_memory_array and slice_arr are also
# combined into one. One key point is that there are still only 8 total calls to the ``ones_like`` function
# and the two chunks that are needed to slice the array are reused.
#
# Chunking a dataset
# -------------------
# Chunking is a very important concept in dask. It allows you to break up your dataset into smaller
# pieces that can be processed in parallel.
#
# Let's say you have a 4D STEM dataset that is 100 x 100 probe positions and 128 x 128 pixels
# at each probe position. In this case we have chunked the first two dimensions
# into 25 x 25 chunks of probe positions and the last two dimensions are left as -1, which means
# that they will be spanned by the entire chunk. This is a common chunking strategy for 4D STEM data

data_4D = da.ones((100, 100, 128, 128), chunks=(25, 25, -1, -1))

# %%
# One thing you might want to do is to create a pacbed image. This is usually done by summing
# over the probe positions.  This is a common operation in 4D STEM data processing.

pacbed = data_4D.sum(axis=(0, 1))
pacbed.visualize()

# %%
# Now we have a task graph that sums over the probe positions.  This is a very simple operation
# and dask is able to optimize it quite well. But what we can see is that the task graph now
# is not embarrassingly parallel anymore. Because we have multiple chunks in the first two dimensions
# we have to sum over those chunks, pass the results to a separate worker etc.
#
# In contrast, if we try to sum over the last two dimensions, to make something like a virtual image,

virtual_image = data_4D.sum(axis=(2, 3))
virtual_image.visualize()

# %%
# Now the task graph is embarrassingly parallel again. This is because the last two dimensions are
# chunked into the same size as the original data. This operation is going to be slightly faster
# than the previous one because it can be done in parallel without any communication between workers.
#
# There is never any free lunch though. Let's say we just want to sum over the central zero beam. Because
# of the "culling" that dask does, it works most efficiently when the data is chunked in a way that
# dask can ignore chunks that are not needed for the computation.

data_4D_new_chunks = da.ones((100, 100, 128, 128), chunks=(25, 25, 8, 8))
virtual_bright_field = data_4D_new_chunks[:, :, 56:72, 56:72].sum(axis=(2, 3))

# %%
# Playing around with it
# ----------------------
# A good rule of thumb is to chunk your data in a way that the last two dimensions are
# chunked into the same size as the original data. It might not be the most efficient way to do
# things like making virtual images, but often it is much more efficient for other operations which
# often operate on each diffraction pattern individually.
#
# Let's try something a little more complicated as well. Let's say we wanted to filter the data
# using a gaussian filter.

gaussian_filter(data_4D, sigma=(0, 0, 1, 1)).visualize()

# %%
# That works quite nicely and we have a nice embarrassingly Parallel workflow. If we try the
# same thing in the (real space) probe dimension.

gaussian_filter(data_4D, sigma=(1, 1, 0, 0)).visualize()


# %%
# Now it is much more complicated.  It will still run, but you might make things more efficient in
# the second case by increasing your chunk size in real space.

rechunked_4d = data_4D.rechunk((50, 50, -1, -1))
gaussian_filter(rechunked_4d, sigma=(1, 1, 0, 0)).visualize()

# %%
# The next example will go over how we can load data lazily.  But the concept is almost identical,
# instead of the ``ones_like`` function at the bottom there will be some function which loads 1
# chunk of the data from the disk.
