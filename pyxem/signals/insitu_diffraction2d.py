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

from hyperspy.signals import Signal1D
from pyxem.signals import Diffraction2D
from hyperspy._signals.lazy import LazySignal

import numpy as np
from hyperspy.roi import RectangularROI

import dask.array as da
from dask.graph_manipulation import clone

from pyxem.utils._dask import _get_dask_array, _get_chunking
from pyxem.utils._insitu import (
    _register_drift_5d,
    _register_drift_2d,
    _g2_2d,
    _interpolate_g2_2d,
    _get_resample_time,
)
import pyxem.utils._pixelated_stem_tools as pst


class InSituDiffraction2D(Diffraction2D):
    """Signal class for in-situ 4D-STEM data.

    Parameters
    ----------
    *args:
        See :class:`hyperspy.api.signals.Signal2D`.
    **kwargs:
        See :class:`hyperspy.api.signals.Signal2D`
    """

    _signal_type = "insitu_diffraction"

    def roll_time_axis(self, time_axis):
        """Roll time axis to default index (2)"""
        return self.rollaxis(time_axis, 2)

    def get_time_series(self, roi=None, time_axis=2):
        """Create a intensity time series from virtual aperture defined by roi.

        Parameters
        ----------
        roi: :obj:`~hyperspy.roi.BaseInteractiveROI`
            Roi for virtual detector. If None, full roi of diffraction plane is used
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
            roi = RectangularROI(
                self.axes_manager.signal_extent[0],
                self.axes_manager.signal_extent[2],
                self.axes_manager.signal_extent[1],
                self.axes_manager.signal_extent[3],
            )

        virtual_series = self.get_integrated_intensity(roi, out_signal_axes=out_axes)
        virtual_series.metadata.General.title = "Integrated intensity time series"

        return virtual_series

    def get_drift_vectors(
        self, time_axis=2, reference="cascade", sub_pixel_factor=10, **kwargs
    ):
        """Calculate real space drift vectors from time series of images

        Parameters
        ----------
        s: :class:`hyperspy.api.signals.Signal2D`
            Time series of reconstructed images
        reference: 'current', 'cascade', or 'stat'
            reference argument passed to :meth:`~hyperspy.api.signals.Signal2D.estimate_shift2D`
            function. Default is 'cascade'
        sub_pixel_factor: float
            sub_pixel_factor passed to :meth:`~hyperspy.api.signals.Signal2D.estimate_shift2D`
            function. Default is 10
        **kwargs:
            Passed to the :meth:`~pyxem.signals.InSituDiffraction2D.get_time_series` function

        Returns
        -------
        shift_vectors
        """
        roi = kwargs.pop("roi", None)
        ref = self.get_time_series(roi=roi, time_axis=time_axis)

        s = ref.estimate_shift2D(
            reference=reference, sub_pixel_factor=sub_pixel_factor, **kwargs
        )
        shift_vectors = Signal1D(s)

        pst._copy_axes_object_metadata(
            self.axes_manager.navigation_axes[time_axis],
            shift_vectors.axes_manager.navigation_axes[0],
        )

        return shift_vectors

    def correct_real_space_drift(
        self, shifts=None, time_axis=2, order=1, lazy_result=True
    ):
        """
        Perform real space drift registration on the dataset.

        Parameters
        ----------
        shifts: Signal1D
            shift vectors to register, must be in the shape of <N_time | 2>.
            If None, shift vectors will be calculated automatically
        time_axis: int
            Index of time axis. Default is 2
        lazy_result: bool, default True
            Whether to return lazy result.
        order: int
           The order of the spline interpolation for registration. Default is 1

        Returns
        ---------
        registered_data: InSituDiffraction2D
            Real space drift corrected version of the original dataset
        """
        if shifts is None:
            shifts = self.get_drift_vectors(time_axis=time_axis)

        if time_axis != 2:
            s_ = self.roll_time_axis(time_axis)
        else:
            s_ = self
        dask_data = _get_dask_array(s_)

        if self._lazy:
            time_chunks = s_.get_chunk_size()[0][0]
        else:
            time_chunks = _get_chunking(s_)[0][0]
        xdrift = shifts.data[:, 0]
        ydrift = shifts.data[:, 1]
        xdrift_dask = da.from_array(
            xdrift[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
            chunks=(time_chunks, 1, 1, 1, 1),
        )
        ydrift_dask = da.from_array(
            ydrift[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
            chunks=(time_chunks, 1, 1, 1, 1),
        )
        depthx = np.ceil(np.max(np.abs(xdrift))).astype(int)
        depthy = np.ceil(np.max(np.abs(ydrift))).astype(int)
        overlapped_depth = {0: 0, 1: depthx, 2: depthy, 3: 0, 4: 0}

        data_overlapped = da.overlap.overlap(
            dask_data, depth=overlapped_depth, boundary={a: "none" for a in range(5)}
        )

        # Clone original overlap dask array to work around memory release issue in map_overlap
        data_clones = da.concatenate(
            [clone(b, omit=data_overlapped) for b in data_overlapped.blocks]
        )

        mapped = data_clones.map_blocks(
            _register_drift_5d,
            shifts1=xdrift_dask,
            shifts2=ydrift_dask,
            order=order,
            dtype="float32",
        )

        registered_data = InSituDiffraction2D(
            da.overlap.trim_internal(mapped, overlapped_depth)
        ).as_lazy()

        # Set axes info for registered signal
        for nav_axis_old, nav_axis_new in zip(
            s_.axes_manager.navigation_axes,
            registered_data.axes_manager.navigation_axes,
        ):
            pst._copy_axes_object_metadata(nav_axis_old, nav_axis_new)
        for sig_axis_old, sig_axis_new in zip(
            s_.axes_manager.signal_axes, registered_data.axes_manager.signal_axes
        ):
            pst._copy_axes_object_metadata(sig_axis_old, sig_axis_new)

        if not lazy_result:
            registered_data.compute()

        return registered_data

    def correct_real_space_drift_fast(
        self, shifts=None, time_axis=2, order=1, **kwargs
    ):
        """
        Perform real space drift registration on the dataset with fast performance
        over spatial axes. If signal is lazy, spatial axes must not be chunked

        Parameters
        ----------
        shifts: Signal1D
            shift vectors to register, must be in the shape of <N_time | 2>.
            If None, shift vectors will be calculated automatically
        time_axis: int
            Index of time axis. Default is 2
        order: int
           The order of the spline interpolation for registration. Default is 1
        **kwargs:
            Passed to :meth:`~hyperspy.signal.BaseSignal.map`

        Returns
        ---------
        registered_data: InSituDiffraction2D
            Real space drift corrected version of the original dataset
        """
        if self._lazy:
            nav_axes = [0, 1, 2]
            nav_axes.remove(2 - time_axis)
            chunkings = self.get_chunk_size()
            if len(chunkings[nav_axes[0]]) != 1 or len(chunkings[nav_axes[1]]) != 1:
                raise Exception(
                    "Spatial axes are chunked. Please rechunk signal or use 'correct_real_space_drift' "
                    "instead"
                )

        if shifts is None:
            shifts = self.get_drift_vectors(time_axis=time_axis)

        if time_axis != 2:
            s_ = self.roll_time_axis(time_axis=time_axis)
        else:
            s_ = self
        s_transposed = s_.transpose(signal_axes=(0, 1))

        xdrift = shifts.data[:, 0]
        ydrift = shifts.data[:, 1]
        xs = Signal1D(
            np.repeat(
                np.repeat(
                    xdrift[:, np.newaxis, np.newaxis],
                    repeats=s_transposed.axes_manager.navigation_axes[0].size,
                    axis=-1,
                ),
                repeats=s_transposed.axes_manager.navigation_axes[1].size,
                axis=1,
            )[:, :, :, np.newaxis]
        )

        ys = Signal1D(
            np.repeat(
                np.repeat(
                    ydrift[:, np.newaxis, np.newaxis],
                    repeats=s_transposed.axes_manager.navigation_axes[0].size,
                    axis=-1,
                ),
                repeats=s_transposed.axes_manager.navigation_axes[1].size,
                axis=1,
            )[:, :, :, np.newaxis]
        )

        registered_data = s_transposed.map(
            _register_drift_2d,
            shift1=xs,
            shift2=ys,
            order=order,
            inplace=False,
            **kwargs
        )

        registered_data_t = registered_data.transpose(navigation_axes=[-2, -1, -3])
        registered_data_t.set_signal_type("insitu_diffraction")

        return registered_data_t

    def get_g2_2d_kresolved(
        self,
        time_axis=2,
        normalization="split",
        k1bin=1,
        k2bin=1,
        tbin=1,
        resample_time=None,
    ):
        """
        Calculate k resolved g2 from in situ diffraction signal

        Parameters
        ----------
        time_axis: int
            Index of time axis. Default is 2
        normalization: string, Default is 'split'
            Normalization format for time autocorrelation, 'split' or 'self'
        k1bin: int
            Binning factor for k1 axis
        k2bin: int
            Binning factor for k2 axis
        tbin: int
            Binning factor for t axis
        resample_time: int or np.array, Default is None
            If int, time is resample into log linear with resample_time as
            number of sampling. If array, it is used as resampled time axis
            instead. No resampling is performed if None

        Returns
        ---------
        g2kt: Signal2D or Correlation2D
            k resolved time correlation signal
        """
        if time_axis != 2:
            transposed_signal = self.roll_time_axis(time_axis).transpose(
                navigation_axes=[0, 1]
            )
        else:
            transposed_signal = self.transpose(navigation_axes=[0, 1])

        g2kt = transposed_signal.map(
            _g2_2d,
            normalization=normalization,
            k1bin=k1bin,
            k2bin=k2bin,
            tbin=tbin,
            inplace=False,
        )

        if resample_time is not None:
            if isinstance(resample_time, int):
                trs = _get_resample_time(
                    t_size=transposed_signal.axes_manager.signal_axes[-1].size / tbin,
                    dt=transposed_signal.axes_manager.signal_axes[-1].scale * tbin,
                    t_rs_size=resample_time,
                )
                g2rs = g2kt.map(
                    _interpolate_g2_2d,
                    t_rs=trs,
                    dt=transposed_signal.axes_manager.signal_axes[-1].scale * tbin,
                    inplace=False,
                )
                g2rs.set_signal_type("correlation")
                return g2rs
            if (
                isinstance(resample_time, (list, tuple, np.ndarray))
                and len(np.shape(resample_time)) == 1
            ):
                g2rs = g2kt.map(
                    _interpolate_g2_2d,
                    t_rs=resample_time / tbin,
                    dt=transposed_signal.axes_manager.signal_axes[-1].scale * tbin,
                    inplace=False,
                )
                g2rs.set_signal_type("correlation")
                return g2rs
            else:
                raise TypeError("'resample_time' must be int or 1d array")

        g2kt.set_signal_type("correlation")

        return g2kt


class LazyInSituDiffraction2D(LazySignal, InSituDiffraction2D):
    pass
