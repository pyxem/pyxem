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

# For 0.20.0 we should update the code to call these functions from utils._dask, where they should be made private

"""Utils for Dask."""


from pyxem.utils._deprecated import deprecated
import scipy.ndimage as ndi


@deprecated(since="0.18.0", removal="0.20.0")
def align_single_frame(image, shifts, **kwargs):
    temp_image = ndi.shift(image, shifts[::-1], **kwargs)
    return temp_image


@deprecated(since="0.18.0", removal="0.20.0")
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


@deprecated(since="0.18.0", removal="0.20.0")
def get_signal_dimension_host_chunk_slice(x, y, chunks):
    chunk_slice_list = get_signal_dimension_chunk_slice_list(chunks)
    for chunk_slice in chunk_slice_list:
        x_slice, y_slice = chunk_slice
        if y_slice.start <= y < y_slice.stop:
            if x_slice.start <= x < x_slice.stop:
                return chunk_slice
    return False
