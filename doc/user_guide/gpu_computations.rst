GPU Computing
-------------

The GPU is a powerful tool for accelerating many types of computations.  The GPU is particularly well-suited
to problems that can be expressed as data-parallel computations additionally GPU's often have higher memory
bandwidth than CPU's.  This means that they can offer much faster processing.

GPU support in pyxem/hyperspy is a point of active development.  Currently, the GPU support is limited and
remains in the `beta` stage. This means that the API is not yet stable and may change as development continues
to occur.  We are actively seeking feedback from users to help guide the development of the GPU support!


Operations that are supported on the GPU:
------------------------------------------
 - Basic Operations
 - Azimuthal Integration
 - Template Matching


You can transfer data to the GPU using the `to_gpu` method.  This method will transfer the data to the GPU
or use dask to perform the operation in parallel.  You can transfer the data back to the CPU using the `from_gpu`.

```python
import pyxem as pxm
s = pxm.data.pdcusi(lazy=True)
with dask.config.set(scheduler='synchronous'): # This is required to ensure that only one thread is used --> 1 GPU
    s.to_gpu() # Creates a plan to transfer the data to GPU
    az = s.get_azimuthal_integral2d(inplace=False) # automatically uses GPU method
    az.from_gpu() # Creates a plan to transfer the data back to the CPU
```

Maybe more useful is the `dask-cuda` package which allows you to use multiple GPU's or will handle the
scheduling of the GPU operations for you without the context managing shown above.

```python

from dask_cuda import LocalCUDACluster
from dask.distributed import Client

cluster = LocalCUDACluster()
client = Client(cluster)

import pyxem as pxm
s = pxm.data.pdcusi(lazy=True)
s.to_gpu() # Creates a plan to transfer the data to GPU
az = s.get_azimuthal_integral2d(inplace=False) # automatically uses GPU method
az.from_gpu() # Creates a plan to transfer the data back to the CPU
```

The GPU support is currently limited to NVIDIA GPUs and requires the `cupy` package to be installed. If
you are interested in increasing GPU support to other vendors, please let us know!
