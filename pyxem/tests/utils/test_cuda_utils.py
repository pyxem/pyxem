from pyxem.utils import cuda_utils as cutls
import dask.array as da
import numpy as np
import pytest
from unittest.mock import Mock


try:
    import cupy as cp

    CUPY_INSTALLED = True
except ImportError:
    CUPY_INSTALLED = False
    cp = np


skip_cupy = pytest.mark.skipif(not CUPY_INSTALLED, reason="cupy is required")


@skip_cupy
def test_dask_array_to_gpu():
    cutls.dask_array_to_gpu(da.array([1, 2, 3, 4]))


@skip_cupy
def test_dask_array_from_gpu():
    cutls.dask_array_from_gpu(da.array(cp.array([1, 2, 3, 4])))
