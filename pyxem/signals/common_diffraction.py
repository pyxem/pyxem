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

from hyperspy.api import interactive
from hyperspy.misc.utils import isiterable
import hyperspy.api as hs

from traits.trait_base import Undefined
from pyxem import CUPY_INSTALLED
import dask.array as da


if CUPY_INSTALLED:
    import cupy as cp
from pyxem.utils.virtual_images_utils import normalize_virtual_images
from importlib import import_module


OUT_SIGNAL_AXES_DOCSTRING = """out_signal_axes : None, iterable of int or string
            Specify which navigation axes to use as signal axes in the virtual
            image. If None, the two first navigation axis are used.
        """


class CommonDiffraction:
    """Common functions for all Diffraction Signals classes"""

    def to_device(self):  # pragma: no cover
        """Transfer the data to the GPU.

        Returns
        -------
        Diffraction2D
            The data on the GPU.
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
        self._gpu = True

    def to_host(self):  # pragma: no cover
        """Transfer the data from the GPU to the CPU."""
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
        self._gpu = False

    @property
    def _gpu(self):
        if not CUPY_INSTALLED:
            return False
        else:
            return isinstance(self.data._meta, cp.ndarray)

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

    def map(
        self,
        function,
        show_progressbar=None,
        num_workers=None,
        inplace=True,
        ragged=None,
        navigation_chunks=None,
        output_signal_size=None,
        output_dtype=None,
        lazy_output=None,
        apply_blockwise=False,
        **kwargs,
    ):
        """Apply a function to the signal data at all the navigation
        coordinates.

        The function must operate on numpy arrays. It is applied to the data at
        each navigation coordinate pixel-py-pixel. Any extra keyword arguments
        are passed to the function. The keywords can take different values at
        different coordinates. If the function takes an `axis` or `axes`
        argument, the function is assumed to be vectorized and the signal axes
        are assigned to `axis` or `axes`.  Otherwise, the signal is iterated
        over the navigation axes and a progress bar is displayed to monitor the
        progress.

        In general, only navigation axes (order, calibration, and number) are
        guaranteed to be preserved.

        Parameters
        ----------

        function : :std:term:`function`
            Any function that can be applied to the signal. This function should
            not alter any mutable input arguments or input data. So do not do
            operations which alter the input, without copying it first.
            For example, instead of doing `image *= mask`, rather do
            `image = image * mask`. Likewise, do not do `image[5, 5] = 10`
            directly on the input data or arguments, but make a copy of it
            first. For example via `image = copy.deepcopy(image)`.
        %s
        %s
        inplace : bool, default True
            If ``True``, the data is replaced by the result. Otherwise
            a new Signal with the results is returned.
        ragged : None or bool, default None
            Indicates if the results for each navigation pixel are of identical
            shape (and/or numpy arrays to begin with). If ``None``,
            the output signal will be ragged only if the original signal is ragged.
        navigation_chunks : str, None, int or tuple of int, default ``None``
            Set the navigation_chunks argument to a tuple of integers to split
            the navigation axes into chunks. This can be useful to enable
            using multiple cores with signals which are less that 100 MB.
            This argument is passed to :meth:`~._signals.lazy.LazySignal.rechunk`.
        output_signal_size : None, tuple
            Since the size and dtype of the signal dimension of the output
            signal can be different from the input signal, this output signal
            size must be calculated somehow. If both ``output_signal_size``
            and ``output_dtype`` is ``None``, this is automatically determined.
            However, if for some reason this is not working correctly, this
            can be specified via ``output_signal_size`` and ``output_dtype``.
            The most common reason for this failing is due to the signal size
            being different for different navigation positions. If this is the
            case, use ragged=True. None is default.
        output_dtype : None, numpy.dtype
            See docstring for output_signal_size for more information.
            Default None.
        %s
        **kwargs : dict
            All extra keyword arguments are passed to the provided function

        Notes
        -----
        If the function results do not have identical shapes, the result is an
        array of navigation shape, where each element corresponds to the result
        of the function (of arbitrary object type), called a "ragged array". As
        such, most functions are not able to operate on the result and the data
        should be used directly.

        This method is similar to Python's :func:`python:map` that can
        also be utilized with a :class:`~hyperspy.signal.BaseSignal`
        instance for similar purposes. However, this method has the advantage of
        being faster because it iterates the underlying numpy data array
        instead of the :class:`~hyperspy.signal.BaseSignal`.

        Currently requires a uniform axis.

        Examples
        --------
        Apply a Gaussian filter to all the images in the dataset. The sigma
        parameter is constant:

        >>> import scipy.ndimage
        >>> im = hs.signals.Signal2D(np.random.random((10, 64, 64)))
        >>> im.map(scipy.ndimage.gaussian_filter, sigma=2.5)

        Apply a Gaussian filter to all the images in the dataset. The signal
        parameter is variable:

        >>> im = hs.signals.Signal2D(np.random.random((10, 64, 64)))
        >>> sigmas = hs.signals.BaseSignal(np.linspace(2, 5, 10)).T
        >>> im.map(scipy.ndimage.gaussian_filter, sigma=sigmas)

        Rotate the two signal dimensions, with different amount as a function
        of navigation index. Delay the calculation by getting the output
        lazily. The calculation is then done using the compute method.

        >>> from scipy.ndimage import rotate
        >>> s = hs.signals.Signal2D(np.random.random((5, 4, 40, 40)))
        >>> s_angle = hs.signals.BaseSignal(np.linspace(0, 90, 20).reshape(5, 4)).T
        >>> s.map(rotate, angle=s_angle, reshape=False, lazy_output=True)
        >>> s.compute()

        Rotate the two signal dimensions, with different amount as a function
        of navigation index. In addition, the output is returned as a new
        signal, instead of replacing the old signal.

        >>> s = hs.signals.Signal2D(np.random.random((5, 4, 40, 40)))
        >>> s_angle = hs.signals.BaseSignal(np.linspace(0, 90, 20).reshape(5, 4)).T
        >>> s_rot = s.map(rotate, angle=s_angle, reshape=False, inplace=False)

        If you want some more control over computing a signal that isn't lazy
        you can always set lazy_output to True and then compute the signal setting
        the scheduler to 'threading', 'processes', 'single-threaded' or 'distributed'.

        Additionally, you can set the navigation_chunks argument to a tuple of
        integers to split the navigation axes into chunks. This can be useful if your
        signal is less that 100 mb but you still want to use multiple cores.

        >>> s = hs.signals.Signal2D(np.random.random((5, 4, 40, 40)))
        >>> s_angle = hs.signals.BaseSignal(np.linspace(0, 90, 20).reshape(5, 4)).T
        >>> s.map(
        ...    rotate, angle=s_angle, reshape=False, lazy_output=True,
        ...    inplace=True, navigation_chunks=(2,2)
        ... )
        >>> s.compute()

        """
        if lazy_output is None:
            lazy_output = self._lazy
        if ragged is None:
            ragged = self.ragged

        # Separate arguments to pass to the mapping function:
        # ndkwargs dictionary contains iterating arguments which must be signals.
        # kwargs dictionary contains non-iterating arguments
        self_nav_shape = self.axes_manager.navigation_shape
        ndkwargs = {}
        ndkeys = [key for key in kwargs if isinstance(kwargs[key], BaseSignal)]
        for key in ndkeys:
            nd_nav_shape = kwargs[key].axes_manager.navigation_shape
            if nd_nav_shape == self_nav_shape:
                ndkwargs[key] = kwargs.pop(key)
            elif nd_nav_shape == () or nd_nav_shape == (1,):
                # This really isn't an iterating signal.
                kwargs[key] = np.squeeze(kwargs[key].data)
            else:
                raise ValueError(
                    f"The size of the navigation_shape for the kwarg {key} "
                    f"(<{nd_nav_shape}> must be consistent "
                    f"with the size of the mapped signal "
                    f"<{self_nav_shape}>"
                )
        # TODO: Consider support for non-uniform signal axis
        if any([not ax.is_uniform for ax in self.axes_manager.signal_axes]):
            _logger.warning(
                "At least one axis of the signal is non-uniform. Can your "
                "`function` operate on non-uniform axes?"
            )
        else:
            # Check if the signal axes have inhomogeneous scales and/or units and
            # display in warning if yes.
            scale = set()
            units = set()
            for i in range(len(self.axes_manager.signal_axes)):
                scale.add(self.axes_manager.signal_axes[i].scale)
                units.add(self.axes_manager.signal_axes[i].units)
            if len(units) != 1 or len(scale) != 1:
                _logger.warning(
                    "The function you applied does not take into account "
                    "the difference of units and of scales in-between axes."
                )
        # If the function has an axis argument and the signal dimension is 1,
        # we suppose that it can operate on the full array and we don't
        # iterate over the coordinates.
        fargs = []
        try:
            # numpy ufunc operate element-wise on the inputs and we don't
            # except them to have an axis argument
            if not isinstance(function, np.ufunc):
                fargs = inspect.signature(function).parameters.keys()
            else:
                _logger.warning(
                    f"The function `{function.__name__}` can directly operate "
                    "on hyperspy signals and it is not necessary to use `map`."
                )
        except TypeError as error:
            # This is probably a Cython function that is not supported by
            # inspect.
            _logger.warning(error)

        # If the function has an `axes` or `axis` argument
        # we suppose that it can operate on the full array and we don't
        # iterate over the coordinates.
        # We use _map_all only when the user doesn't specify axis/axes
        if (
            not ndkwargs
            and not lazy_output
            and self.axes_manager.signal_dimension == 1
            and "axis" in fargs
            and "axis" not in kwargs.keys()
        ):
            kwargs["axis"] = self.axes_manager.signal_axes[-1].index_in_array
            result = self._map_all(function, inplace=inplace, **kwargs)
        elif (
            not ndkwargs
            and not lazy_output
            and "axes" in fargs
            and "axes" not in kwargs.keys()
        ):
            kwargs["axes"] = tuple(
                [axis.index_in_array for axis in self.axes_manager.signal_axes]
            )
            result = self._map_all(function, inplace=inplace, **kwargs)
        else:
            if show_progressbar is None:
                from hyperspy.defaults_parser import preferences

                show_progressbar = preferences.General.show_progressbar
            # Iteration over coordinates.
            result = self._map_iterate(
                function,
                iterating_kwargs=ndkwargs,  # function argument(s) (iterating)
                show_progressbar=show_progressbar,
                ragged=ragged,
                inplace=inplace,
                lazy_output=lazy_output,
                num_workers=num_workers,
                output_dtype=output_dtype,
                output_signal_size=output_signal_size,
                navigation_chunks=navigation_chunks,
                **kwargs,  # function argument(s) (non-iterating)
            )
        if not inplace:
            return result
        else:
            self.events.data_changed.trigger(obj=self)

    map.__doc__ %= (SHOW_PROGRESSBAR_ARG, LAZY_OUTPUT_ARG, NUM_WORKERS_ARG)

    def _map_all(self, function, inplace=True, **kwargs):
        """
        The function has to have either 'axis' or 'axes' keyword argument,
        and hence support operating on the full dataset efficiently and remove
        the signal axes.
        """
        old_shape = self.data.shape
        newdata = function(self.data, **kwargs)
        if inplace:
            self.data = newdata
            if self.data.shape != old_shape:
                self.axes_manager.remove(self.axes_manager.signal_axes)
            self._lazy = False
            self._assign_subclass()
            return None
        else:
            sig = self._deepcopy_with_new_data(newdata)
            if sig.data.shape != old_shape:
                sig.axes_manager.remove(sig.axes_manager.signal_axes)
            sig._lazy = False
            sig._assign_subclass()
            return sig

    def _map_iterate(
        self,
        function,
        iterating_kwargs=None,
        show_progressbar=None,
        ragged=False,
        inplace=True,
        output_signal_size=None,
        output_dtype=None,
        lazy_output=None,
        num_workers=None,
        navigation_chunks="auto",
        **kwargs,
    ):
        if lazy_output is None:
            lazy_output = self._lazy

        if not self._lazy:
            s_input = self.as_lazy()
            s_input.rechunk(nav_chunks=navigation_chunks)
        else:
            s_input = self

        # unpacking keyword arguments
        if iterating_kwargs is None:
            iterating_kwargs = {}
        elif isinstance(iterating_kwargs, (tuple, list)):
            iterating_kwargs = dict((k, v) for k, v in iterating_kwargs)

        nav_indexes = s_input.axes_manager.navigation_indices_in_array
        chunk_span = np.equal(s_input.data.chunksize, s_input.data.shape)
        chunk_span = [
            chunk_span[i] for i in s_input.axes_manager.signal_indices_in_array
        ]

        if not all(chunk_span):
            _logger.info(
                "The chunk size needs to span the full signal size, rechunking..."
            )

            old_sig = s_input.rechunk(inplace=False, nav_chunks=None)
        else:
            old_sig = s_input

        os_am = old_sig.axes_manager

        autodetermine = (
            output_signal_size is None or output_dtype is None
        )  # try to guess output dtype and sig size?
        if autodetermine and is_cupy_array(self.data):
            raise ValueError(
                "Autodetermination of `output_signal_size` and "
                "`output_dtype` is not supported for cupy array."
            )

        args, arg_keys = old_sig._get_iterating_kwargs(iterating_kwargs)

        if autodetermine:  # trying to guess the output d-type and size from one signal
            testing_kwargs = {}
            for ikey, key in enumerate(arg_keys):
                test_ind = (0,) * len(os_am.navigation_axes)
                # For discussion on if squeeze is necessary, see
                # https://github.com/hyperspy/hyperspy/pull/2981
                testing_kwargs[key] = np.squeeze(args[ikey][test_ind].compute())[()]
            testing_kwargs = {**kwargs, **testing_kwargs}
            test_data = np.array(
                old_sig.inav[(0,) * len(os_am.navigation_shape)].data.compute()
            )
            temp_output_signal_size, temp_output_dtype = guess_output_signal_size(
                test_data=test_data,
                function=function,
                ragged=ragged,
                **testing_kwargs,
            )
            if output_signal_size is None:
                output_signal_size = temp_output_signal_size
            if output_dtype is None:
                output_dtype = temp_output_dtype
        output_shape = self.axes_manager._navigation_shape_in_array + output_signal_size
        arg_pairs, adjust_chunks, new_axis, output_pattern = _get_block_pattern(
            (old_sig.data,) + args, output_shape
        )

        axes_changed = len(new_axis) != 0 or len(adjust_chunks) != 0

        if show_progressbar:
            pbar = ProgressBar()
            pbar.register()
        mapped = da.blockwise(
            process_function_blockwise,
            output_pattern,
            *concat(arg_pairs),
            adjust_chunks=adjust_chunks,
            new_axes=new_axis,
            align_arrays=False,
            dtype=output_dtype,
            concatenate=True,
            arg_keys=arg_keys,
            function=function,
            output_dtype=output_dtype,
            nav_indexes=nav_indexes,
            output_signal_size=output_signal_size,
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
            sig = s_input._deepcopy_with_new_data(mapped)

        am = sig.axes_manager
        sig._lazy = lazy_output

        if ragged:
            axes_dicts = self.axes_manager._get_navigation_axes_dicts()
            sig.axes_manager.__init__(axes_dicts)
            sig.axes_manager._ragged = True
        elif axes_changed:
            am.remove(am.signal_axes[len(output_signal_size) :])
            for ind in range(len(output_signal_size) - am.signal_dimension, 0, -1):
                am._append_axis(size=output_signal_size[-ind], navigate=False)

        if not ragged:
            sig.axes_manager._ragged = False
            if output_signal_size == () and am.navigation_dimension == 0:
                add_scalar_axis(sig)
            sig.get_dimensions_from_data()
        sig._assign_subclass()

        if not lazy_output and not data_stored:
            sig.data = sig.data.compute(num_workers=num_workers)

        if show_progressbar:
            pbar.unregister()

        return sig
