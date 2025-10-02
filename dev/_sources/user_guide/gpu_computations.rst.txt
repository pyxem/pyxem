GPU Computing
-------------

The GPU is a powerful tool for accelerating many types of computations.  The GPU is particularly well-suited
to problems that can be expressed as data-parallel computations additionally GPU's often have higher memory
bandwidth than CPU's.  This means that they can offer much faster processing.

GPU support in pyxem/hyperspy is a point of active development.  Currently, the GPU support is limited and
remains in the `beta` stage. This means that the API is not yet stable and may change as development continues
to occur.  We are actively seeking feedback from users to help guide the development of the GPU support!

The GPU support is currently limited to NVIDIA GPUs and requires the `cupy <https://cupy.dev>`_ package to be installed.
If you are interested in increasing GPU support to other vendors, please let us know!

Just a note that cuda can be a bit difficult to install depending on your hardware etc.  If you are having
trouble, please let us know by raising an issue and we will try to help you out.


Supported Operations
^^^^^^^^^^^^^^^^^^^^

Operations that are supported on the GPU:

- Generic Operations
- 2D Azimuthal Integration
- Template Matching


Transfer Data To and From the GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can transfer data to the GPU using the :meth:`~.signals.CommonDiffraction.to_device` method.  This method
will transfer the data to the GPU or use dask to perform the operation in parallel.  You can transfer the data
back to the CPU using the :meth:`~.signals.CommonDiffraction.to_host` method.

Note that this will be limited by the number of GPU's you have available.

.. code-block::

    import pyxem as pxm
    s = pxm.data.pdnip_glass(lazy=True)
    s.to_device() # Creates a plan to transfer the data to GPU
    az = s.get_azimuthal_integral2d(inplace=False) # automatically uses GPU method
    az.to_host() # Creates a plan to transfer the data back to the CPU

Working with Multiple GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^

Maybe more useful is the `dask-cuda <https://docs.rapids.ai/api/dask-cuda/stable>`_ (linux only) package
which allows you to use multiple GPU's or will handle the scheduling of the GPU operations for you without the
context managing shown above.

.. code-block::

    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client

    cluster = LocalCUDACluster()
    client = Client(cluster)

    import pyxem as pxm
    s = pxm.data.pdnip_glass(lazy=True)
    s.to_device() # Creates a plan to transfer the data to GPU
    az = s.get_azimuthal_integral2d(inplace=False) # automatically uses GPU method
    az.to_host() # Creates a plan to transfer the data back to the CPU
    az.compute() # This will 1 transfer the data to the GPU in blocks operate and then transfer the data back to CPU
