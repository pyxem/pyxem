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

"""Utils for slicing on vectors signals."""

import numpy as np
from hyperspy.signal import BaseSignal


def slice_signal(arr, col_slice, row_slice):
    if (
        hasattr(row_slice, "ndim") and row_slice.ndim == 0
    ):  # Dealing with possible 0 d array
        row_slice = [
            row_slice,
        ]
    if arr.ndim == 1:
        return arr[col_slice]
    else:
        return arr[row_slice, col_slice]


# Note this functionality should be upstreamed to hyperspy and replaced by equivalent functionality
# added to the ``isig`` attribute of hyperspy signals.
class Slicer:
    def __init__(self, signal):
        self.signal = signal

    def __getitem__(self, item):
        if isinstance(item, tuple):  # multiple dimensions
            if len(item) == 0 or len(item) > 2:
                raise ValueError(
                    "Only column and row slicing 2-D arrays is currently supported"
                )
            col_slice = self.str2slice(item[0])
            if len(item) == 2:
                row_slice = item[1]
            else:
                row_slice = slice(None)
        else:
            col_slice = self.str2slice(item)
            row_slice = slice(None)
        if self.signal.ragged:
            kwargs = dict(output_signal_size=(), output_dtype=object)
            if not isinstance(row_slice, slice):
                if not isinstance(row_slice, BaseSignal) or not row_slice.ragged:
                    raise ValueError(
                        "Only ragged boolean indexing is currently supported for ragged signals"
                    )
        else:
            kwargs = dict()

        slic = self.signal.map(
            slice_signal,
            col_slice=col_slice,
            row_slice=row_slice,
            inplace=False,
            ragged=self.signal.ragged,
            **kwargs
        )
        if self.signal.scales is not None:
            slic.scales = np.array(self.signal.scales)[col_slice]
        if self.signal.offsets is not None:
            slic.offsets = np.array(self.signal.offsets)[col_slice]
        if self.signal.column_names is not None:
            from pyxem.signals import DiffractionVectors1D, DiffractionVectors2D

            if isinstance(slic, DiffractionVectors1D) and isinstance(
                self.signal, DiffractionVectors2D
            ):
                name = np.array(self.signal.column_names)[col_slice]
                slic.axes_manager.signal_axes[0].name = "" if name is None else name
                slic.column_names = None
            else:
                slic.column_names = np.array(self.signal.column_names)[col_slice]
        if self.signal.units is not None:
            from pyxem.signals import DiffractionVectors1D, DiffractionVectors2D

            if isinstance(slic, DiffractionVectors1D) and isinstance(
                self.signal, DiffractionVectors2D
            ):
                unit = np.array(self.signal.units)[col_slice]
                slic.axes_manager.signal_axes[0].units = "" if unit is None else unit
            else:
                slic.column_names = np.array(self.signal.column_names)[col_slice]
        return slic

    def str2slice(self, item):
        if isinstance(item, str):
            item = list(self.signal.column_names).index(item)
        elif isinstance(item, (np.ndarray, list)):
            item = np.array([self.str2slice(i) for i in item])
        elif isinstance(item, (slice, int)):
            pass
        else:
            raise ValueError(
                "item must be a string or an int or an array of strings or ints"
            )
        return item
