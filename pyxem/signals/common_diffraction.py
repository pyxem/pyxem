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

from hyperspy.api import interactive
from hyperspy.misc.utils import isiterable
import hyperspy.api as hs

from traits.trait_base import Undefined
from pyxem import CUPY_INSTALLED
from hyperspy.misc.utils import _get_block_pattern, add_scalar_axis
from tlz import concat

if CUPY_INSTALLED:
    import cupy as cp
from pyxem.utils.virtual_images_utils import normalize_virtual_images


OUT_SIGNAL_AXES_DOCSTRING = """out_signal_axes : None, iterable of int or string
            Specify which navigation axes to use as signal axes in the virtual
            image. If None, the two first navigation axis are used.
        """


class CommonDiffraction:
    """Common functions for all Diffraction Signals classes"""

    def to_device(self):  # pragma: no cover
        """Transfer the data to the GPU.

        Notes
        -----
        This code is currently in an experimental stage. Please report and issues you have on the pyxem github
        """
        if not CUPY_INSTALLED:
            raise ImportError(
                "The cupy package is required to use this method. "
                "Please install it using `conda install cupy`."
            )
        if not self._gpu:
            if self._lazy:
                self.data = self.data.map_blocks(
                    cp.asarray
                )  # pass chunk-wise the data to GPU
            else:
                self.data = cp.asarray(self.data)  # pass all the data to the GPU

    def to_host(self):  # pragma: no cover
        """Transfer the data from the GPU to the CPU.

        Notes
        -----
        This code is currently in an experimental stage. Please report and issues you have on the pyxem github
        """
        if not CUPY_INSTALLED:
            raise ImportError(
                "The cupy package is required to use this method. "
                "Please install it using `conda install cupy`."
            )
        if self._gpu:
            if self._lazy:
                self.data = self.data.map_blocks(cp.asnumpy)
            else:
                self.data = cp.asnumpy(self.data)

    @property
    def _gpu(self):
        if not CUPY_INSTALLED:
            return False
        else:
            if self._lazy:
                return isinstance(self.data._meta, cp.ndarray)
            else:
                return isinstance(self.data, cp.ndarray)

    @property
    def unit(self):
        if self.axes_manager.signal_axes[0].units is Undefined:
            print("The unit hasn't been set yet")
            return
        else:
            return self.axes_manager.signal_axes[0].units

    @unit.setter
    def unit(self, unit):
        """Set the units

        Parameters
        ----------
        unit : "q_nm^-1", "q_A^-1","k_nm^-1","k_A^-1","2th_deg", "2th_rad"
            The diffraction units
        """
        acceptable = ["q_nm^-1", "q_A^-1", "k_nm^-1", "k_A^-1", "2th_deg", "2th_rad"]
        if unit in acceptable:
            for axes in self.axes_manager.signal_axes:
                axes.units = unit
        else:
            print(
                'The unit must be "q_nm^-1", "q_A^-1","k_nm^-1",'
                '"k_A^-1","2th_deg", "2th_rad"'
            )

    @staticmethod
    def _get_sum_signal(signal, out_signal_axes=None):
        out = signal.nansum(signal.axes_manager.signal_axes)
        if out_signal_axes is None:
            out_signal_axes = list(
                np.arange(min(signal.axes_manager.navigation_dimension, 2))
            )
        if len(out_signal_axes) > signal.axes_manager.navigation_dimension:
            raise ValueError(
                "The length of 'out_signal_axes' can't be longer"
                "than the navigation dimension of the signal."
            )
        # Reset signal to default Signal1D or Signal2D
        out.set_signal_type("")
        return out.transpose(out_signal_axes)

    def plot_integrated_intensity(self, roi, out_signal_axes=None, **kwargs):
        """Interactively plots the integrated intensity over the scattering
        range defined by the roi.

        Parameters
        ----------
        roi : float
            Any interactive ROI detailed in HyperSpy.
        out_signal_axes : None, iterable of int or string
            Specify which navigation axes to use as signal axes in the virtual
            image. If None, the two first navigation axis are used.
        **kwargs:
            Keyword arguments to be passed to the `plot` method of the virtual
            image.

        Examples
        --------
        .. code-block:: python

            >>> # For 1D diffraction signal, we can use a SpanROI
            >>> roi = hs.roi.SpanROI(left=1., right=2.)
            >>> dp.plot_integrated_intensity(roi)

        .. code-block:: python

            >>> # For 2D diffraction signal,we can use a CircleROI
            >>> roi = hs.roi.CircleROI(3, 3, 5)
            >>> dp.plot_integrated_intensity(roi)

        """
        # Plot signal when necessary
        if self._plot is None or not self._plot.is_active:
            self.plot()

        # Get the sliced signal from the roi
        sliced_signal = roi.interactive(self, axes=self.axes_manager.signal_axes)

        # Create an output signal for the virtual dark-field calculation.
        out = self._get_sum_signal(self, out_signal_axes)
        out.metadata.General.title = "Integrated intensity"

        # Create the interactive signal
        interactive(
            sliced_signal.nansum,
            axis=sliced_signal.axes_manager.signal_axes,
            event=roi.events.changed,
            recompute_out_event=None,
            out=out,
        )

        # Plot the result
        out.plot(**kwargs)

    def get_virtual_image(self, rois, new_axis_dict=None, normalize=False):
        """Get a virtual images from a set of rois

        Parameters
        ----------
        rois : iterable of :obj:`hyperspy.roi.BaseInteractiveROI`
            Any interactive ROI detailed in HyperSpy.
        new_axis_dict : dict, optional
            A dictionary with the properties of the new axis. If None, a default
            axis is created.
        normalize : bool, optional
            If True, the virtual images are normalized to the maximum value.
        """
        if not isiterable(rois):
            rois = [
                rois,
            ]
        if new_axis_dict is None:
            new_axis_dict = {
                "name": "Virtual Dark Field",
                "offset": 0,
                "scale": 1,
                "units": "a.u.",
                "size": len(rois),
            }

        vdfs = [self.get_integrated_intensity(roi) for roi in rois]

        vdfim = hs.stack(
            vdfs, new_axis_name=new_axis_dict["name"], show_progressbar=False
        )

        vdfim.set_signal_type("virtual_dark_field")

        if vdfim.metadata.has_item("Diffraction.integrated_range"):
            del vdfim.metadata.Diffraction.integrated_range
        vdfim.metadata.set_item("Diffraction.roi_list", [f"{roi}" for roi in rois])

        # Set new axis properties
        if len(rois) > 1:
            new_axis = vdfim.axes_manager[new_axis_dict["name"]]
            for k, v in new_axis_dict.items():
                setattr(new_axis, k, v)

        if normalize:
            vdfim.map(normalize_virtual_images, show_progressbar=False)
        return vdfim

    def get_integrated_intensity(self, roi, out_signal_axes=None):
        """Obtains the intensity integrated over the scattering range as
        defined by the roi.

        Parameters
        ----------
        roi : :obj:`hyperspy.roi.BaseInteractiveROI`
            Any interactive ROI detailed in HyperSpy.
        %s

        Returns
        -------
        integrated_intensity : :obj:`hyperspy.signals.Signal2D` or :obj:`hyperspy.signals.Signal1D`
            The intensity integrated over the scattering range as defined by
            the roi.

        Examples
        --------
        .. code-block:: python

            >>> # For 1D diffraction signal, we can use a SpanROI
            >>> roi = hs.roi.SpanROI(left=1., right=2.)
            >>> virtual_image = dp.get_integrated_intensity(roi)

        .. code-block:: python

            >>> # For 2D diffraction signal,we can use a CircleROI
            >>> roi = hs.roi.CircleROI(3, 3, 5)
            >>> virtual_image = dp.get_integrated_intensity(roi)

        """
        dark_field = roi(self, axes=self.axes_manager.signal_axes)
        dark_field_sum = self._get_sum_signal(dark_field, out_signal_axes)
        dark_field_sum.metadata.General.title = "Integrated intensity"
        roi_info = f"{roi}"
        if self.metadata.get_item("General.title") not in ("", None):
            roi_info += f" of {self.metadata.General.title}"
        dark_field_sum.metadata.set_item("Diffraction.integrated_range", roi_info)

        return dark_field_sum

    get_integrated_intensity.__doc__ %= OUT_SIGNAL_AXES_DOCSTRING

    def add_navigation_signal(self, data, name="nav1", unit=None, nav_plot=False):
        """Adds in a navigation signal to the metadata.  Any type of navigation signal is acceptable.

        Parameters
        -------------------
        data: np.array
            The data for the navigation signal.  Should be the same size as the navigation axis.
        name: str
            The name of the axis.
        unit: str
            The units for the intensity of the plot. e.g 'nm' for thickness.
        """
        dict_signal = {}
        dict_signal[name] = {
            "data": data,
            "unit": unit,
            "use_as_navigation_plot": nav_plot,
        }
        if not self.metadata.has_item("Navigation_signals"):
            self.metadata.add_node("Navigation_signals")
        self.metadata.Navigation_signals.add_dictionary(dict_signal)

    def _map_blocks_prepare(self, navigation_chunks="auto"):
        """Prepare the signal for mapping blocks.  This function will check if the signal
        is chunked correctly and rechunk if necessary.
        """
        if not self._lazy:
            s_input = self.as_lazy()
            s_input.rechunk(nav_chunks=navigation_chunks)
        else:
            s_input = self
        chunk_span = np.equal(s_input.data.chunksize, s_input.data.shape)
        chunk_span = [
            chunk_span[i] for i in s_input.axes_manager.signal_indices_in_array
        ]

        if not all(chunk_span):
            old_sig = s_input.rechunk(inplace=False, nav_chunks=None)
        else:
            old_sig = s_input
        return old_sig

    def _blockwise(
        self,
        func,
        signal_shape=None,
        navigation_shape=None,
        dtype=None,
        inplace=False,
        lazy_output=None,
        ragged=False,
        num_workers=None,
        meta=None,
        **kwargs,
    ):
        """Apply a function to each block of the signal. This function might change in the
        future.

        Parameters
        ----------
        func : callable
            The function to apply to each block.
        *args : list
            The arguments to pass to the function.
        signal_shape : tuple
            The shape of the signal.
        dtype: np.dtype
            The type of the signal.
        **kwargs : dict
            The keyword arguments to pass to the function.

        Returns
        -------
        Diffraction2D
            The signal with the function applied to each block.
        """
        if dtype is None:
            dtype = self.data.dtype
        if signal_shape is None:
            signal_shape = self.axes_manager._signal_shape_in_array
        if navigation_shape is None:
            navigation_shape = self.axes_manager._navigation_shape_in_array
        if lazy_output is None:
            lazy_output = self._lazy
        new_shape = navigation_shape + signal_shape
        old_sig = self._map_blocks_prepare()
        if meta is None:
            meta = old_sig.data._meta

        arg_pairs, adjust_chunks, new_axis, output_pattern = _get_block_pattern(
            (old_sig.data,), new_shape
        )
        axes_changed = len(new_axis) != 0 or len(adjust_chunks) != 0

        mapped = da.blockwise(
            func,
            output_pattern,
            *concat(arg_pairs),
            adjust_chunks=adjust_chunks,
            new_axes=new_axis,
            align_arrays=False,
            dtype=dtype,
            concatenate=True,
            meta=meta,
            **kwargs,
        )
        data_stored = False
        if inplace:
            if (
                not self._lazy
                and not lazy_output
                and (mapped.shape == self.data.shape)
                and (mapped.dtype == self.data.dtype)
            ):
                # da.store is used to avoid unnecessary amount of memory usage.
                # By using it here, the contents in mapped is written directly to
                # the existing NumPy array, avoiding a potential doubling of memory use.
                da.store(
                    mapped,
                    self.data,
                    dtype=mapped.dtype,
                    compute=True,
                    num_workers=num_workers,
                )
                data_stored = True
            else:
                self.data = mapped
            sig = self
        else:
            sig = self._deepcopy_with_new_data(mapped)

        am = sig.axes_manager
        sig._lazy = lazy_output

        if ragged:
            axes_dicts = self.axes_manager._get_navigation_axes_dicts()
            sig.axes_manager.__init__(axes_dicts)
            sig.axes_manager._ragged = True
        elif axes_changed:
            am.remove(am.signal_axes[len(signal_shape) :])
            for ind in range(len(signal_shape) - am.signal_dimension, 0, -1):
                am._append_axis(size=signal_shape[-ind], navigate=False)

        if not ragged:
            sig.axes_manager._ragged = False
            if signal_shape == () and am.navigation_dimension == 0:
                add_scalar_axis(sig)
            sig.get_dimensions_from_data()
        sig._assign_subclass()

        if not lazy_output and not data_stored:
            sig.data = sig.data.compute(num_workers=num_workers)
        return sig
