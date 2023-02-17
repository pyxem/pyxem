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

from hyperspy.signals import Signal1D
from pyxem.signals import Diffraction2D
import numpy as np
from hyperspy.roi import RectangularROI

import dask.array as da
from dask.graph_manipulation import clone

from pyxem.utils.dask_tools import _get_dask_array, _get_chunking
from pyxem.utils.insitu_utils import _register_drift_5d, _g2_2d
import pyxem.utils.pixelated_stem_tools as pst


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
            roi = RectangularROI(self.axes_manager.signal_extent[0],
                                 self.axes_manager.signal_extent[1],
                                 self.axes_manager.signal_extent[2],
                                 self.axes_manager.signal_extent[3])

        virtual_series = self.get_integrated_intensity(roi, out_signal_axes=out_axes)
        virtual_series.metadata.General.title = "Integrated intensity time series"

        return virtual_series

    def get_drift_vectors(self, time_axis=2, **kwargs):
        """
        Calculate real space drift vectors from time series of images

         Parameters
        ----------
        s: Signal2D
            Time series of reconstructed images
        **kwargs:
            Passed to the hs.signals.Signal2D.estimate_shift2D() function

        Returns
        -------
        shift_vectors

        """
        roi = kwargs.pop("roi", None)
        ref = self.get_time_series(roi=roi, time_axis=time_axis)

        shift_reference = kwargs.get("reference", "stat")
        sub_pixel = kwargs.get("sub_pixel_factor", 10)
        s = ref.estimate_shift2D(reference=shift_reference,
                                 sub_pixel_factor=sub_pixel,
                                 **kwargs)
        shift_vectors = Signal1D(s)

        pst._copy_axes_object_metadata(self.axes_manager.navigation_axes[time_axis],
                                       shift_vectors.axes_manager.navigation_axes[0])

        return shift_vectors

    def correct_real_space_drift(self, shifts=None, time_axis=2, lazy_result=True):
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
        if shifts is None:
            shifts = self.get_drift_vectors(time_axis=time_axis)

        if time_axis != 2:
            dask_data = _get_dask_array(self.roll_time_axis(time_axis))
        else:
            dask_data = _get_dask_array(self)

        if self._lazy:
            time_chunks = self.get_chunk_size()[0][0]
        else:
            time_chunks = _get_chunking(self)[0][0]
        xdrift = shifts.data[:, 0]
        ydrift = shifts.data[:, 1]
        xdrift_dask = da.from_array(xdrift[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
                                    chunks=(time_chunks, 1, 1, 1, 1))
        ydrift_dask = da.from_array(ydrift[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
                                    chunks=(time_chunks, 1, 1, 1, 1))
        depthx = np.ceil(np.max(np.abs(xdrift))).astype(int)
        depthy = np.ceil(np.max(np.abs(ydrift))).astype(int)
        overlapped_depth = {0: 0, 1: depthy, 2: depthx, 3: 0, 4: 0}

        data_overlapped = da.overlap.overlap(dask_data,
                                             depth=overlapped_depth,
                                             boundary={a: 'none' for a in range(5)})

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
        for nav_axis_old, nav_axis_new in zip(
                self.axes_manager.navigation_axes, registered_data.axes_manager.navigation_axes
        ):
            pst._copy_axes_object_metadata(nav_axis_old, nav_axis_new)
        for sig_axis_old, sig_axis_new in zip(
                self.axes_manager.signal_axes, registered_data.axes_manager.signal_axes
        ):
            pst._copy_axes_object_metadata(sig_axis_old, sig_axis_new)

        if not lazy_result:
            registered_data.compute()

        return registered_data

    def get_g2_2d_kresolved(self, time_axis=2, normalization='split', kbin=1, tbin=1):
        """
        Calculate k resolved g2 from in situ diffraction signal

        Parameters
        ----------
        time_axis: int
            Index of time axis. Default is 2
        normalization: string
            Normalization format for time autocorrelation, 'split' or 'self'
        kbin: int
            Binning factor for both k axes
        tbin: int
            Binning factor for t axis

        Returns
        ---------
        g2kt: Signal2D or Correlation2D?
            k resolved time correlation signal
        """
        if time_axis != 2:
            transposed_signal = self.roll_time_axis(time_axis).transpose(navigation_axes=[0, 1])
        else:
            transposed_signal = self.transpose(navigation_axes=[0, 1])

        g2kt = transposed_signal.map(_g2_2d,
                                     normalization=normalization,
                                     kbin=kbin,
                                     tbin=tbin,
                                     inplace=False)

        g2kt.set_signal_type('correlation')

        return g2kt
