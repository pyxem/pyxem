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

import numpy as np
import dask.array as da
from scipy import ndimage
from skimage.feature import match_template, blob_dog, blob_log
import scipy.ndimage as ndi
from skimage import morphology
from hyperspy.misc.utils import isiterable


def align_single_frame(image, shifts, **kwargs):
    temp_image = ndi.shift(image, shifts[::-1], **kwargs)
    return temp_image


def get_signal_dimension_chunk_slice_list(chunks):
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


def get_signal_dimension_host_chunk_slice(x, y, chunks):
    chunk_slice_list = get_signal_dimension_chunk_slice_list(chunks)
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
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
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


def _intensity_peaks_image_chunk(data, peak_array, disk_r):
    """Intensity of the peaks is calculated by taking the mean value
    of the pixel values inside radius disk_r where the centers are the
    peak positions for the entire chunk. If the peak position plus disk_r
    exceed the detector edges, then the intensity for that peak will be
    put to zero.

    Parameters
    ----------
    data : NumPy 4D array
    peak_array: NumPy 2D array
        In the form [[x0, y0], [x1, y1], [x2, y2], ...]
    r__disk : Integer number which represents the radius of the discs

    Returns
    -------
    intensity array : NumPy object with x, y and intensity for every peak

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> peak_array = dt._peak_find_dog_chunk(s.data)
    >>> intensity = dt._intensity_peaks_image_chunk(s.data, peak_array, 5)
    """

    output_array = np.empty(data.shape[:-2], dtype=object)
    if peak_array.ndim < data.ndim:
        dim_dif = data.ndim - peak_array.ndim
        for i in range(dim_dif):
            peak_array = np.expand_dims(peak_array, axis=peak_array.ndim)

    for index in np.ndindex(data.shape[:-2]):
        islice = np.s_[index]
        peaks = peak_array[islice][0, 0]
        output_array[islice] = _intensity_peaks_image_single_frame(
            data[islice], peaks, disk_r
        )
    return output_array


def _intensity_peaks_image(dask_array, peak_array, disk_r):
    """Intensity of the peaks is calculated by taking the mean value
    of the pixel values inside radius disk_r where the centers are the
    peak positions for the full dask array. If the peak position plus disk_r
    exceed the detector edges, then the intensity for that peak will be
    put to zero.

    Parameters
    ----------
    dask_array : Dask array
        The two last dimensions are the signal dimensions. Must have at least
        2 dimensions.
    peak_array: Dask array
        Must have the same shape as dask_array's navigation dimensions,
        ergo dask_array's dimensions, except the two last ones.
        So if dask_array has the shape [10, 15, 50, 50], the peak_array
        must have the shape [10, 15].
        In the form [[x0, y0], [x1, y1], [x2, y2], ...]
    r_disk : Integer number which represents the radius of the discs
        peak_array = Numpy object with x and y coordinates peaks

    Returns
    -------
    intensity_array : dask object array
        Same size as the two last dimensions in data.
        The x, y peak positions and intensities are stored in the
        three columns.

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> import dask.array as da
    >>> s = pxm.dummy_data.dummy_data.get_cbed_signal()
    >>> dask_array = da.from_array(s.data, chunks=(5, 5, 25, 25))
    >>> peak_array = dt._peak_find_dog(dask_array)
    >>> intensity = dt._intensity_peaks_image(dask_array, peak_array, 5)

    """
    if dask_array.shape[:-2] != peak_array.shape:
        raise ValueError(
            "peak_array ({0}) must have the same shape as dask_array "
            "except the two last dimensions ({1})".format(
                peak_array.shape, dask_array.shape[:-2]
            )
        )
    if not hasattr(dask_array, "chunks"):
        raise ValueError("dask_array must be a Dask array")
    if not hasattr(peak_array, "chunks"):
        raise ValueError("peak_array must be a Dask array")

    dask_array_rechunked = _rechunk_signal2d_dim_one_chunk(dask_array)

    peak_array_rechunked = peak_array.rechunk(dask_array_rechunked.chunks[:-2])
    shape = list(peak_array_rechunked.shape)
    shape.extend([1, 1])
    peak_array_rechunked = peak_array_rechunked.reshape(shape)

    drop_axis = (dask_array_rechunked.ndim - 2, dask_array_rechunked.ndim - 1)
    kwargs_intensity_peaks = {"disk_r": disk_r}

    output_array = da.map_blocks(
        _intensity_peaks_image_chunk,
        dask_array_rechunked,
        peak_array_rechunked,
        drop_axis=drop_axis,
        dtype=object,
        **kwargs_intensity_peaks
    )
    return output_array


def _peak_refinement_centre_of_mass_frame(frame, peaks, square_size):
    """Refining the peak positions using the center of mass of the peaks.

    Parameters
    ----------
    frame : Numpy array
    peaks : Numpy array
    square_size : Even Int, this should be larger than the diffraction
        disc diameter. However not to large because then other discs will
        influence the center of mass.

    Returns
    -------
    peak_array : A Numpy array with the x and y positions of the refined peaks

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.dummy_data.get_cbed_signal()
    >>> peak_array = np.array(([48.,48.],[25.,48.]))
    >>> r_p_array = dt._peak_refinement_centre_of_mass_frame(
    ...     s.data[0,0,:,:], peak_array, 20)

    """
    new_peak = np.zeros(peaks.shape, dtype="float64")

    if peaks.size == 0:
        return peaks
    else:
        for i in range(peaks.shape[0]):
            subframe = _center_of_mass_experimental_square(
                frame, np.asarray(peaks[i], dtype=np.uint16), square_size
            )
            # If the peak is outside the subframe, return the same position
            if subframe is None:
                new_peak[i, 0] = peaks[i, 0].astype("float64")
                new_peak[i, 1] = peaks[i, 1].astype("float64")
            else:
                subframe = subframe.astype("float64")
                f_x = (subframe.shape[0]) / 2
                f_y = (subframe.shape[1]) / 2
                cx, cy = _center_of_mass_hs(subframe)
                new_peak[i, 0] = peaks[i, 0].astype("float64") + (cx - f_x)
                new_peak[i, 1] = peaks[i, 1].astype("float64") + (cy - f_y)

        return new_peak


def _peak_refinement_centre_of_mass_chunk(data, peak_array, square_size):
    """Refining the peak positions using the center of mass of the peaks.

    Parameters
    ----------
    data : Numpy array
        The two last dimensions are the signal dimensions. Must have at least
        2 dimensions.
    peak_array : Numpy array
    square_size : Even integer, this should be larger than the diffraction
        disc diameter. However not to large because then other discs will
        influence the center of mass.

    Returns
    -------
    peak_array : A NumPy array with the x and y positions of the refined peaks

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> s = pxm.dummy_data.get_cbed_signal()
    >>> peak_array = dt._peak_find_dog_chunk(s.data)
    >>> r_p_array = dt._peak_refinement_centre_of_mass_chunk(
    ...     s.data, peak_array, 20)


    """
    output_array = np.empty(data.shape[:-2], dtype=object)
    if peak_array.ndim < data.ndim:
        dim_dif = data.ndim - peak_array.ndim
        for i in range(dim_dif):
            peak_array = np.expand_dims(peak_array, axis=peak_array.ndim)
    frame = np.zeros(data.shape[-2:])
    for index in np.ndindex(data.shape[:-2]):
        islice = np.s_[index]
        frame[:] = data[islice]
        peaks = peak_array[islice][0, 0]
        output_array[islice] = _peak_refinement_centre_of_mass_frame(
            frame, peaks, square_size
        )
    return output_array


def _peak_refinement_centre_of_mass(dask_array, peak_array, square_size):
    """Refining the peak positions using the center of mass of the peaks

    Parameters
    ----------
    dask_array : Dask array
        The two last dimensions are the signal dimensions. Must have at least
        2 dimensions.
    peak_array : Dask array
    square_size : Even integer, this should be larger than the diffraction
        disc diameter. However not to large because then other discs will
        influence the center of mass.

    Returns
    -------
    peak_array : A dask array with the x and y positions of the refined peaks

    Examples
    --------
    >>> import pyxem.utils.dask_tools as dt
    >>> import dask.array as da
    >>> s = pxm.dummy_data.get_cbed_signal()
    >>> dask_array = da.from_array(s.data, chunks=(5, 5, 25, 25))
    >>> peak_array = dt._peak_find_dog(dask_array)
    >>> r_p_array = dt._peak_refinement_centre_of_mass(
    ...     dask_array, peak_array, 20)
    >>> ref_peak_array = r_p_array.compute()
    """
    if dask_array.shape[:-2] != peak_array.shape:
        raise ValueError(
            "peak_array ({0}) must have the same shape as dask_array "
            "except the two last dimensions ({1})".format(
                peak_array.shape, dask_array.shape[:-2]
            )
        )
    if not hasattr(dask_array, "chunks"):
        raise ValueError("dask_array must be a Dask array")
    if not hasattr(peak_array, "chunks"):
        raise ValueError("peak_array must be a Dask array")

    dask_array_rechunked = _rechunk_signal2d_dim_one_chunk(dask_array)

    peak_array_rechunked = peak_array.rechunk(dask_array_rechunked.chunks[:-2])
    shape = list(peak_array_rechunked.shape)
    shape.extend([1, 1])
    peak_array_rechunked = peak_array_rechunked.reshape(shape)

    drop_axis = (dask_array_rechunked.ndim - 2, dask_array_rechunked.ndim - 1)
    kwargs_refinement = {"square_size": square_size}

    output_array = da.map_blocks(
        _peak_refinement_centre_of_mass_chunk,
        dask_array_rechunked,
        peak_array_rechunked,
        drop_axis=drop_axis,
        dtype=object,
        **kwargs_refinement
    )
    return output_array


def _center_of_mass_hs(z):
    """Return the center of mass of an array with coordinates in the
    hyperspy convention.

    Parameters
    ----------
    z : 2D NumPy array

    Returns
    -------
    cx: The x location of the center of mass
    cy: The y location of the center of mass
    """
    s = np.sum(z)
    if s != 0:
        z *= 1 / s

    dx = np.sum(z, axis=0)
    dy = np.sum(z, axis=1)
    h, w = z.shape
    cx = np.sum(dx * np.arange(w))
    cy = np.sum(dy * np.arange(h))

    return cy, cx


def _center_of_mass_experimental_square(z, vector, square_size):
    """Wrapper for _get_experimental_square that makes the non-zero
    elements symmetrical around the 'unsubpixeled' peak by zeroing a
    'spare' row and column (top and left).

    Parameters
    ----------
    z : np.array
    vector : np.array([x,y])
    square_size : int (even)

    Returns
    -------
    z_adpt : np.array
        z, but with row and column zero set to 0
    """

    z_adpt = np.copy(
        _get_experimental_square(z, vector=vector, square_size=square_size)
    )

    if z_adpt.size == 0:
        return None
    else:
        z_adpt[:, 0] = 0
        z_adpt[0, :] = 0
        return z_adpt


def _get_experimental_square(z, vector, square_size):
    """Defines a square region around a given diffraction vector and returns an
    upsampled copy.

    Parameters
    ----------
    z : np.array()
        Single diffraction pattern
    vector : np.array()
        Single vector in pixels (int) [x,y] with top left as [0,0]
    square_size : int
        The length of one side of the bounding square (must be even)

    Returns
    -------
    _z : np.array()
        Of size (L,L) where L = square_size

    Examples
    --------
    >>> d = pxm.dummy_data.get_cbed_signal()
    >>> import pyxem.utils.dask_tools as dt
    >>> sub_d = dt._get_experimental_square(d.data[0,0,:,:], [50,50], 30)

    """
    if square_size % 2 != 0:
        raise ValueError("'square_size' must be an even number")

    cx, cy, half_ss = (vector[1]), (vector[0]), int(square_size / 2)
    _z = z[cy - half_ss : cy + half_ss, cx - half_ss : cx + half_ss]
    return _z
