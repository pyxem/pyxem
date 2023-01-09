# -*- coding: utf-8 -*-
# Copyright 2016-2022 The pyXem developers
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

from hyperspy.signals import Signal2D, BaseSignal
from pyxem.signals import Diffraction2D
import numpy as np
from hyperspy.roi import CircleROI

import dask.array as da
from dask.graph_manipulation import clone

from pyxem.utils.dask_tools import _get_dask_array
from pyxem.utils.insitu_utils import _register_drift_5d, get_drift_vectors

class InSituDiffraction2D(Diffraction2D):
    """Signal class for in-situ 4D-STEM data."""

    _signal_type = "insitu_diffraction"

    def __init__(self, *args, **kwargs):
        """
        Create a InsituDiffraction2D object from a numpy.ndarray or dask.array.

        Parameters
        ----------
        *args :
            Passed to the __init__ of Signal2D. The first arg should be
            a numpy.ndarray or dask.array
        **kwargs :
            Passed to the __init__ of Signal2D.
        """
        super().__init__(*args, **kwargs)

    def roll_time_axis(self, time_axis):
        """Roll time axis to default index (2)"""
        return self.rollaxis(time_axis, 2)

    def get_time_series(self, roi=None, time_axis=2):
        """
        Create a intensity time series from virtual aperture defined by roi.

        Parameters
        ----------
        roi: :obj:`hyperspy.roi.BaseInteractiveROI`
            Roi for virtual detector. If None, an ADF mask is created
        time_axis: int
            Index of time axis. Default is 2

        Returns
        ---------
        virtual_series: Signal2D
            Time series of virtual detector images
        """
        out_axes = [0, 1, 2]
        out_axes.remove(time_axis)

        if roi is None:
            cx = (self.axes_manager.signal_extent[0] + self.axes_manager.signal_extent[1]) / 2
            cy = (self.axes_manager.signal_extent[2] + self.axes_manager.signal_extent[3]) / 2
            r = min(cx, cy)
            r_inner = r / 2
            roi = CircleROI(cx=cx, cy=cy, r=r, r_inner=r_inner)
        virtual_series = self.get_integrated_intensity(roi, out_signal_axes=out_axes)
        virtual_series.metadata.General.title = "Integrated intensity time series"

        return virtual_series

    def correct_real_space_drift(self, xdrift=None, ydrift=None, time_axis=2, lazy_result=True):
        """
        Perform real space drift registration on the dataset.

        Parameters
        ----------
        xdrift: np.array
            Real space drift vectors in x direction. If None, drift is calculated from data instead.
        ydrift: np.array
            Real space drift vectors in x direction. If None, drift is calculated from data instead.
        time_axis: int
            Index of time axis. Default is 2
        lazy_result: bool, default True
            Whether to return lazy result.

        Returns
        ---------
        registered_data: InSituDiffraction2D
            Real space drift corrected version of the original dataset
        """
        if xdrift is None or ydrift is None:
            ref = self.get_time_series(time_axis=time_axis)
            xdrift, ydrift = get_drift_vectors(ref)

        if time_axis != 2:
            dask_data = _get_dask_array(self.roll_time_axis(time_axis))
        else:
            dask_data = _get_dask_array(self)

        time_chunks = self.get_chunk_size()[0][0]
        xdrift_dask = da.from_array(xdrift[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
                                    chunks=(time_chunks, 1, 1, 1, 1))
        ydrift_dask = da.from_array(ydrift[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
                                    chunks=(time_chunks, 1, 1, 1, 1))
        depthx = np.ceil(np.max(np.abs(xdrift))).astype(int)
        depthy = np.ceil(np.max(np.abs(ydrift))).astype(int)
        overlapped_depth = {0: 0, 1: depthy, 2: depthx, 3: 0, 4: 0}

        data_overlapped = da.overlap.overlap(dask_data,
                                             depth=overlapped_depth,
                                             boundary='none')

        # Clone original overlap dask array to work around memory release issue in map_overlap
        data_clones = da.concatenate(
            [clone(b, omit=data_overlapped) for b in data_overlapped.blocks]
        )

        mapped = data_clones.map_blocks(_register_drift_5d,
                                        shifts1=ydrift_dask,
                                        shifts2=xdrift_dask,
                                        dtype='float32')

        registered_data = InSituDiffraction2D(
            da.overlap.trim_internal(mapped, overlapped_depth)
        ).as_lazy()

        # Set axes info for registered signal
        for i, axis in enumerate(registered_data.axes_manager.navigation_axes):
            axis.name = self.axes_manager.navigation_axes[i].name
            axis.scale = self.axes_manager.navigation_axes[i].scale
            axis.offset = self.axes_manager.navigation_axes[i].offset
            axis.unit = self.axes_manager.navigation_axes[i].unit
        for i, axis in enumerate(registered_data.axes_manager.signal_axes):
            axis.name = self.axes_manager.siganl_axes[i].name
            axis.scale = self.axes_manager.signal_axes[i].scale
            axis.offset = self.axes_manager.signal_axes[i].offset
            axis.unit = self.axes_manager.signal_axes[i].unit

        if not lazy_result:
            registered_data.compute()

        return registered_data
