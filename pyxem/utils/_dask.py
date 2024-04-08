# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

"""Utils for using dask."""

import numpy as np
import dask.array as da
import scipy.ndimage as ndi
from skimage import morphology
from hyperspy.misc.utils import isiterable


def _align_single_frame(image, shifts, **kwargs):
    temp_image = ndi.shift(image, shifts[::-1], **kwargs)
    return temp_image


def _get_signal_dimension_chunk_slice_list(chunks):
    """Convenience function for getting the signal chunks as slices

    The slices are assumed to be used on a HyperSpy signal object.
    Thus the input will be in the Dask chunk order (y, x), while the
    output will be in the HyperSpy order (x, y).

    """
    chunk_slice_raw_list = da.core.slices_from_chunks(chunks[-2:])
    chunk_slice_list = []
    for chunk_slice_raw in chunk_slice_raw_list:
        chunk_slice_list.append((chunk_slice_raw[1], chunk_slice_raw[0]))
    return chunk_slice_list


def _get_signal_dimension_host_chunk_slice(x, y, chunks):
    chunk_slice_list = _get_signal_dimension_chunk_slice_list(chunks)
    for chunk_slice in chunk_slice_list:
        x_slice, y_slice = chunk_slice
        if y_slice.start <= y < y_slice.stop:
            if x_slice.start <= x < x_slice.stop:
                return chunk_slice
    return False


def _intensity_peaks_image_single_frame(frame, peaks, disk_r):
    """Intensity of the peaks is calculated by taking the mean value
    of the pixel values inside radius disk_r where the centers are the
    peak positions. If the peak position plus disk_r exceed the detector
    edges, then the intensity for that peak will be put to zero.

    Parameters
    ----------
    frame : NumPy 2D array
    peaks: Numpy 2D array with x and y coordinates of peaks
    disk : NumPy 2D array
        Must be smaller than frame
        peaks: NumPy Object
        can have multiple peaks per image

    Returns
    -------
    intensity_array : NumPy array with
        peak coordinates and intensity of peaks

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.data.dummy_data.dummy_data.get_cbed_signal()
    >>> peaks = np.array(([50,50],[25,50]))
    >>> intensity = dt._intensity_peaks_image_single_frame(
    ...     s.data[0,0,:,:], peaks, 5)

    """
    array_shape = peaks.shape
    mask = morphology.disk(disk_r)
    size = np.shape(frame)
    intensity_array = np.zeros((array_shape[0], 3), dtype="float64")
    for i in range(array_shape[0]):
        cx = int(peaks[i, 0])
        cy = int(peaks[i, 1])
        intensity_array[i, 0] = peaks[i, 0]
        intensity_array[i, 1] = peaks[i, 1]
        if (
            (cx - disk_r < 0)
            | (cx + disk_r + 1 >= size[0])
            | (cy - disk_r < 0)
            | (cy + disk_r + 1 >= size[1])
        ):
            intensity_array[i, 2] = 0
        else:
            subframe = frame[
                cx - disk_r : cx + disk_r + 1, cy - disk_r : cy + disk_r + 1
            ]
            intensity_array[i, 2] = np.mean(mask * subframe)

    return intensity_array


def _get_chunking(signal, chunk_shape=None, chunk_bytes=None):
    """Get chunk tuple based on the size of the dataset.

    The signal dimensions will be within one chunk, and the navigation
    dimensions will be chunked based on either chunk_shape, or
    be optimized based on the chunk_bytes.

    Parameters
    ----------
    signal : hyperspy or pyxem signal
    chunk_shape : int, optional
        Size of the navigation chunk, of None (the default), the chunk
        size will be set automatically.
    chunk_bytes : int or string, optional
        Number of bytes in each chunk. For example '60MiB'. If None (the default),
        the limit will be '30MiB'. Will not be used if chunk_shape is None.

    Returns
    -------
    chunks : tuple
    """
    if chunk_bytes is None:
        chunk_bytes = "30MiB"
    nav_dim = signal.axes_manager.navigation_dimension
    sig_dim = signal.axes_manager.signal_dimension
    if chunk_shape is not None:
        if not isiterable(chunk_shape):
            chunk_shape = [chunk_shape] * nav_dim

    chunks_dict = {}
    for i in range(nav_dim):
        if chunk_shape is None:
            chunks_dict[i] = "auto"
        else:
            chunks_dict[i] = chunk_shape[i]
    for i in range(nav_dim, nav_dim + sig_dim):
        chunks_dict[i] = -1

    chunks = da.core.normalize_chunks(
        chunks=chunks_dict,
        shape=signal.data.shape,
        limit=chunk_bytes,
        dtype=signal.data.dtype,
    )
    return chunks


def _get_dask_array(signal, chunk_shape=None, chunk_bytes=None):
    if signal._lazy:
        dask_array = signal.data
    else:
        chunks = _get_chunking(signal, chunk_shape, chunk_bytes)
        dask_array = da.from_array(signal.data, chunks=chunks)
    return dask_array
