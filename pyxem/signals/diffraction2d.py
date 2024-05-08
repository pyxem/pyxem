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
from scipy.ndimage import rotate
from skimage import morphology
import dask.array as da
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import warnings

import hyperspy.api as hs
from hyperspy.signals import Signal2D, BaseSignal
from hyperspy._signals.lazy import LazySignal
from hyperspy.misc.utils import isiterable
from importlib import import_module
from hyperspy.axes import UniformDataAxis

from pyxem.signals import (
    CommonDiffraction,
)
from pyxem.utils.pyfai_utils import (
    get_azimuthal_integrator,
    _get_radial_extent,
    _get_setup,
)
from pyxem.utils.diffraction import (
    azimuthal_integrate1d,
    azimuthal_integrate2d,
    gain_normalise,
    remove_bad_pixels,
    circular_mask,
    find_beam_offset_cross_correlation,
    normalize_template_match,
    convert_affine_to_transform,
    apply_transformation,
    find_beam_center_blur,
    find_beam_center_interpolate,
    find_hot_pixels,
    integrate_radially,
    medfilt_1d,
    sigma_clip,
    center_of_mass,
)
from pyxem.utils._azimuthal_integrations import (
    _slice_radial_integrate,
    _slice_radial_integrate1d,
)
from pyxem.utils._dask import (
    _get_dask_array,
    _get_signal_dimension_host_chunk_slice,
    _align_single_frame,
)
from pyxem.utils._signals import (
    _select_method_from_method_dict,
    _to_hyperspy_index,
)
import pyxem.utils._pixelated_stem_tools as pst
import pyxem.utils._dask as dt
import pyxem.utils.ransac_ellipse_tools as ret
from pyxem.utils._deprecated import deprecated, deprecated_argument

from pyxem.utils._background_subtraction import (
    _subtract_median,
    _subtract_dog,
    _subtract_hdome,
    _subtract_radial_median,
)
from pyxem.utils.calibration import Calibration

from pyxem import CUPY_INSTALLED

if CUPY_INSTALLED:
    import cupy as cp


class Diffraction2D(CommonDiffraction, Signal2D):
    """Signal class for two-dimensional diffraction data in Cartesian coordinates.

    Parameters
    ----------
    *args
        See :class:`~hyperspy._signals.signal2d.Signal2D`.
    **kwargs
        See :class:`~hyperspy._signals.signal2d.Signal2D`
    """

    _signal_type = "diffraction"

    """ Methods that make geometrical changes to a diffraction pattern """

    def __init__(self, *args, **kwargs):
        """
        Create a Diffraction2D object from numpy.ndarray.

        Parameters
        ----------
        *args :
            Passed to the __init__ of Signal2D. The first arg should be
            numpy.ndarray
        **kwargs :
            Passed to the __init__ of Signal2D
        """
        super().__init__(*args, **kwargs)
        self.calibration = Calibration(self)

    def apply_affine_transformation(
        self, D, order=1, keep_dtype=False, inplace=True, *args, **kwargs
    ):
        """Correct geometric distortion by applying an affine transformation.

        Parameters
        ----------
        D : array or Signal2D of arrays
            3x3 np.array (or Signal2D thereof) specifying the affine transform
            to be applied.
        order : 1,2,3,4 or 5
            The order of interpolation on the transform. Default is 1.
        keep_dtype : bool
            If True dtype of returned ElectronDiffraction2D Signal is that of
            the input, if False, casting to higher precision may occur.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to :meth:`hyperspy.api.signals.BaseSignal.map`.
        **kwargs:
            Keyword arguments to be passed to :meth:`hyperspy.api.signals.BaseSignal.map`.

        Returns
        -------
            ElectronDiffraction2D Signal containing the affine Transformed
            diffraction patterns.


        """

        shape = self.axes_manager.signal_shape
        if isinstance(D, np.ndarray):
            transformation = convert_affine_to_transform(D, shape)
        else:
            transformation = D.map(
                convert_affine_to_transform, shape=shape, inplace=False
            )

        if not keep_dtype:
            out_dtype = float
        else:
            out_dtype = self.data.dtype

        return self.map(
            apply_transformation,
            transformation=transformation,
            output_dtype=out_dtype,
            order=order,
            keep_dtype=keep_dtype,
            inplace=inplace,
            *args,
            **kwargs,
        )

    def shift_diffraction(
        self,
        shift_x,
        shift_y,
        interpolation_order=1,
        inplace=False,
        show_progressbar=True,
    ):
        """Shift the diffraction patterns in a pixelated STEM signal.

        The points outside the boundaries are set to zero.

        Parameters
        ----------
        shift_x, shift_y : int or NumPy array
            If given as int, all the diffraction patterns will have the same
            shifts. Each diffraction pattern can also have different shifts,
            by passing a NumPy array with the same dimensions as the navigation
            axes.
        interpolation_order : int
            When shifting, a spline interpolation is used. This parameter
            sets the order of this spline. Must be between 0 and 5.
            Note that in some low-signal and high noise datasets, using a
            non-zero order might lead to artifacts. See the docstring in
            scipy.ndimage.shift for more information. Default 1.
        inplace : bool
            If True (default), the data is replaced by the result. Useful when
            working with very large datasets, as this avoids doubling the
            amount of memory needed. If False, a new signal with the results
            is returned.
        show_progressbar : bool
            Default True.

        Returns
        -------
        shifted_signal : Diffraction2D signal

        Examples
        --------
        >>> s = pxm.data.dummy_data.get_disk_shift_simple_test_signal()
        >>> s_c = s.center_of_mass(threshold=3., show_progressbar=False)
        >>> s_c -= 25 # To shift the center disk to the middle (25, 25)
        >>> s_shift = s.shift_diffraction(
        ...     s_c.inav[0].data, s_c.inav[1].data,
        ...     show_progressbar=False)
        >>> s_shift.plot()

        Using a different interpolation order

        >>> s_shift = s.shift_diffraction(
        ...     s_c.inav[0].data, s_c.inav[1].data, interpolation_order=3,
        ...     show_progressbar=False)

        """

        if (not isiterable(shift_x)) or (not isiterable(shift_y)):
            shift_x, shift_y = pst._make_centre_array_from_signal(
                self, x=shift_x, y=shift_y
            )
        s_shift_x = BaseSignal(shift_x).T
        s_shift_y = BaseSignal(shift_y).T

        s_shift = self.map(
            pst._shift_single_frame,
            inplace=inplace,
            ragged=False,
            show_progressbar=show_progressbar,
            interpolation_order=interpolation_order,
            shift_x=s_shift_x,
            shift_y=s_shift_y,
        )
        if not inplace:
            return s_shift

    def rotate_diffraction(self, angle, show_progressbar=True):
        """
        Rotate the diffraction dimensions.

        Parameters
        ----------
        angle : scalar
            Clockwise rotation in degrees.
        show_progressbar : bool
            Default True

        Returns
        -------
        rotated_signal : Diffraction2D class

        Examples
        --------
        >>> s = pxm.data.dummy_data.get_holz_simple_test_signal()
        >>> s_rot = s.rotate_diffraction(30, show_progressbar=False)

        """
        s_rotated = self.map(
            rotate,
            ragged=False,
            angle=-angle,
            reshape=False,
            inplace=False,
            show_progressbar=show_progressbar,
        )
        if self._lazy:
            s_rotated.compute(show_progressbar=show_progressbar)
        return s_rotated

    def flip_diffraction_x(self):
        """Flip the dataset along the diffraction x-axis.

        The function returns a new signal, but the data itself
        is a view of the original signal. So changing the returned signal
        will also change the original signal (and visa versa). To avoid
        changing the original signal, use the deepcopy method afterwards,
        but note that this requires double the amount of memory.
        See below for an example of this.

        Returns
        -------
        flipped_signal : Diffraction2D signal

        Example
        -------
        >>> s = pxm.data.dummy_data.get_holz_simple_test_signal()
        >>> s_flip = s.flip_diffraction_x()

        To avoid changing the original object afterwards

        >>> s_flip = s.flip_diffraction_x().deepcopy()

        """
        s_out = self.copy()
        s_out.axes_manager = self.axes_manager.deepcopy()
        s_out.data = np.flip(self.data, axis=-1)
        return s_out

    def flip_diffraction_y(self):
        """Flip the dataset along the diffraction y-axis.

        The function returns a new signal, but the data itself
        is a view of the original signal. So changing the returned signal
        will also change the original signal (and visa versa). To avoid
        changing the original signal, use the deepcopy method afterwards,
        but note that this requires double the amount of memory.
        See below for an example of this.


        Returns
        -------
        flipped_signal : Diffraction2D signal

        Example
        -------
        >>> s = pxm.data.dummy_data.get_holz_simple_test_signal()
        >>> s_flip = s.flip_diffraction_y()

        To avoid changing the original object afterwards

        >>> s_flip = s.flip_diffraction_y().deepcopy()

        """
        s_out = self.copy()
        s_out.axes_manager = self.axes_manager.deepcopy()
        s_out.data = np.flip(self.data, axis=-2)
        return s_out

    """ Masking and other non-geometrical 'correction' to patterns """

    def get_direct_beam_mask(self, radius):
        """Generate a signal mask for the direct beam.

        Parameters
        ----------
        radius : float
            Radius for the circular mask in pixel units.

        Return
        ------
        numpy.ndarray
            The mask of the direct beam
        """
        shape = self.axes_manager.signal_shape
        center = (shape[1] - 1) / 2, (shape[0] - 1) / 2

        signal_mask = Signal2D(circular_mask(shape=shape, radius=radius, center=center))

        return signal_mask

    def apply_gain_normalisation(
        self, dark_reference, bright_reference, inplace=True, *args, **kwargs
    ):
        """Apply gain normalization to experimentally acquired electron
        diffraction patterns.

        Parameters
        ----------
        dark_reference : pyxem.signals.ElectronDiffraction2D
            Dark reference image.
        bright_reference : pyxem.signals.Diffraction2D
            Bright reference image.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to :meth:`hyperspy.api.signals.BaseSignal.map`.
        **kwargs:
            Keyword arguments to be passed to :meth:`hyperspy.api.signal.BaseSignal.map`.

        """
        return self.map(
            gain_normalise,
            dref=dark_reference,
            bref=bright_reference,
            inplace=inplace,
            *args,
            **kwargs,
        )

    @deprecated_argument(
        name="lazy_result", alternative="lazy_output", since="0.15.0", removal="1.0.0"
    )
    def subtract_diffraction_background(
        self, method="median kernel", inplace=False, **kwargs
    ):
        """Background subtraction of the diffraction data.

        Parameters
        ----------
        method : str, optional
            'difference of gaussians', 'median kernel', 'radial median', 'h-dome'
            Default 'median kernel'.

            For `difference of gaussians` the parameters min_sigma (default:1) and
            max_sigma(default:55) control the size of the gaussian kernels used.

            For `median kernel` the footprint(default:19) parameter detemines the
            footprint used to determine the median.

            For `radial median` the parameters center_x(default:128), center_y(default:128) are
            used to detmine the center of the pattern to use to determine the median.

            For `h-dome` the parameter h detemines the relative height of local peaks that
            are supressed.
        **kwargs :
                To be passed to the method chosen: min_sigma/max_sigma, footprint,
                centre_x,centre_y / h

        Returns
        -------
        s : pyxem.signals.Diffraction2D

        Examples
        --------
        >>> s = pxm.data.dummy_data.get_cbed_signal()
        >>> s_r = s.subtract_diffraction_background(method='median kernel',
        ...     footprint=20, lazy_output=False, show_progressbar=False)
        >>> s_r.plot()
        """
        method_dict = {
            "difference of gaussians": _subtract_dog,
            "median kernel": _subtract_median,
            "radial median": _subtract_radial_median,
            "h-dome": _subtract_hdome,
        }
        if method not in method_dict:
            raise NotImplementedError(
                f"The method specified, '{method}',"
                f" is not implemented.  The different methods are:"
                f" 'difference of gaussians','median kernel',"
                f"'radial median' or 'h-dome'."
            )
        subtraction_function = method_dict[method]

        return self.map(subtraction_function, inplace=inplace, **kwargs)

    @deprecated_argument(
        name="mask_array", since="0.15.0", removal="1.0.0", alternative="mask"
    )
    def find_dead_pixels(
        self,
        dead_pixel_value=0,
        mask=None,
    ):
        """Find dead pixels in the diffraction images.

        Parameters
        ----------
        dead_pixel_value : scalar
            Default 0
        mask_array : Boolean Numpy array
        lazy_result : bool
            If True, return a lazy signal. If False, compute
            the result and return a non-lazy signal. Default False.
        show_progressbar : bool

        Returns
        -------
        s_dead_pixels : HyperSpy 2D signal
            With dead pixels as True, rest as False.

        Examples
        --------
        >>> s = pxm.data.dummy_data.get_dead_pixel_signal()
        >>> s_dead_pixels = s.find_dead_pixels(show_progressbar=False)

        Using a mask array

        >>> import numpy as np
        >>> mask_array = np.zeros((128, 128), dtype=bool)
        >>> mask_array[:, 100:] = True
        >>> s = pxm.data.dummy_data.get_dead_pixel_signal()
        >>> s_dead_pixels = s.find_dead_pixels(
        ...     mask_array=mask_array, show_progressbar=False)

        Getting a lazy signal as output

        >>> s_dead_pixels = s.find_dead_pixels(
        ...     lazy_result=True, show_progressbar=False)

        See Also
        --------
        find_hot_pixels
        correct_bad_pixels

        """
        mean_signal = self.mean(axis=self.axes_manager.navigation_axes)
        dead_pixels = mean_signal == dead_pixel_value
        if mask is not None:
            dead_pixels = dead_pixels * np.invert(mask)
        return dead_pixels

    @deprecated_argument(
        name="mask_array", since="0.15.0", removal="1.0.0", alternative="mask"
    )
    @deprecated_argument(
        name="lazy_result", since="0.15.0", removal="1.0.0", alternative="lazy_output"
    )
    def find_hot_pixels(
        self, threshold_multiplier=500, mask=None, inplace=False, **kwargs
    ):
        """Find hot pixels in the diffraction images.

        Note: this method will be default return a lazy signal, since the
        size of the returned signal is the same shape as the original
        signal. So for large datasets actually calculating computing
        the results can use a lot of memory.

        In addition, this signal is currently not very optimized with
        regards to memory use, so be careful when using this method
        for large datasets.

        Parameters
        ----------
        threshold_multiplier : scalar
            Default 500
        mask_array : Boolean NumPy array
        lazy_result : bool
            If True, return a lazy signal. If False, compute
            the result and return a non-lazy signal. Default True.
        show_progressbar : bool

        Examples
        --------
        >>> s = pxm.data.dummy_data.get_hot_pixel_signal()
        >>> s_hot_pixels = s.find_hot_pixels(show_progressbar=False)

        Using a mask array

        >>> import numpy as np
        >>> mask_array = np.zeros((128, 128), dtype=bool)
        >>> mask_array[:, 100:] = True
        >>> s = pxm.data.dummy_data.get_hot_pixel_signal()
        >>> s_hot_pixels = s.find_hot_pixels(
        ...     mask_array=mask_array, show_progressbar=False)

        Getting a non-lazy signal as output

        >>> s_hot_pixels = s.find_hot_pixels()

        See Also
        --------
        find_dead_pixels
        correct_bad_pixels

        """
        return self.map(
            find_hot_pixels,
            threshold_multiplier=threshold_multiplier,
            mask=mask,
            inplace=inplace,
            **kwargs,
        )

    @deprecated_argument(
        name="lazy_result", since="0.15.0", removal="1.0.0", alternative="lazy_output"
    )
    def correct_bad_pixels(
        self,
        bad_pixel_array,
        **kwargs,
    ):
        """Correct bad (dead/hot) pixels by replacing their values with the mean value of neighbors.

        Parameters
        ----------
        bad_pixel_array : array-like
            List of pixels to correct
        show_progressbar : bool, optional
            Default True
        lazy_output : bool, optional
            When working lazily, determines if the result is computed. Default is True (ie. no .compute)
        inplace : bool, optional
            When working in memory, determines if operation is performed inplace, default is True. When
            working lazily the result will NOT be inplace.
        *args :
            passed to :meth:`hyperspy.api.signals.BaseSignal.map` if working in memory
        **kwargs :
            passed to :meth:`hyperspy.api.signals.BaseSignal.map` if working in memory

        Returns
        -------
        signal_corrected: :class:`pyxem.signals.Diffraction2D` or :class:`pyxem.signals.LazyDiffraction2D`

        Examples
        --------
        >>> s = pxm.data.dummy_data.get_hot_pixel_signal()
        >>> s_hot_pixels = s.find_hot_pixels()
        >>> s_corr = s.correct_bad_pixels(s_hot_pixels)

        See Also
        --------
        find_dead_pixels
        find_hot_pixels

        """
        return self.map(remove_bad_pixels, bad_pixels=bad_pixel_array, **kwargs)

    """ Direct beam and peak finding tools """

    @deprecated_argument(
        name="lazy_result", since="0.14", removal="1.0.0", alternative="lazy_output"
    )
    def get_direct_beam_position(
        self,
        method,
        lazy_output=None,
        signal_slice=None,
        half_square_width=None,
        **kwargs,
    ):
        """Estimate the direct beam position in each experimentally acquired
        electron diffraction pattern. Returns the shifts required to center the
        diffraction pattern.

        Parameters
        ----------
        method : str,
            Must be one of "cross_correlate", "blur", "interpolate" or "center_of_mass".

           "cross_correlate": Center finding using cross-correlation of circles of
                ``radius_start`` to ``radius_finish``.
           "blur": Center finding by blurring each frame with a Gaussian kernel with
                standard deviation ``sigma`` and finding the maximum.
           "interpolate": Finding the center by summing along X/Y and finding the peak
                for each axis independently. Data is blurred first using a Gaussian kernel
                with standard deviation ``sigma``.
           "center_of_mass": The center is found using a calculation of the center of mass.
                Optionally a ``mask`` can be applied to focus on just the center of some
                dataset. A threshold value can also be given to suppress contrast from
                weaker diffraction features.
        lazy_output : optional
            If True, s_shifts will be a lazy signal. If False, a non-lazy signal.
            By default, if the signal is (non-)lazy, the result will also be (non-)lazy.
        signal_slice : None or tuple
            A tuple defining the (low_x,high_x, low_y, high_y) to slice the data before
            finding the direct beam. Equivalent to
            s.isig[low_x:high_x, low_y:high_y].get_direct_beam_position()+[low_x,low_y])
        half_square_width : int
            Half the side length of square that captures the direct beam in all
            scans. Means that the centering algorithm is stable against
            diffracted spots brighter than the direct beam. Crops the diffraction
            pattern to `half_square_width` pixels around th center of the diffraction
            pattern. Only one of `half_square_width` or signal_slice can be defined.
        **kwargs:
            Additional arguments accepted by :func:`pyxem.utils.diffraction.find_beam_center_blur`,
            :func:`pyxem.utils.diffraction.find_beam_center_interpolate`,
            :func:`pyxem.utils.diffraction.find_beam_offset_cross_correlation`,
            and :func:`pyxem.signals.diffraction2d.Diffraction2D.center_of_mass`,

        Returns
        -------
        s_shifts : :class:`pyxem.signals.BeamShift`
            Array containing the shifts for each SED pattern, with the first
            signal index being the x-shift and the second the y-shift.

        """
        if half_square_width is not None and signal_slice is not None:
            raise ValueError(
                "Only one of `signal_slice` or `half_sqare_width` " "can be defined"
            )
        elif half_square_width is not None:
            signal_shape = self.axes_manager.signal_shape
            signal_center = np.array(signal_shape) / 2
            min_x = int(signal_center[0] - half_square_width)
            max_x = int(signal_center[0] + half_square_width)
            min_y = int(signal_center[1] - half_square_width)
            max_y = int(signal_center[1] + half_square_width)
            signal_slice = (min_x, max_x, min_y, max_y)

        if signal_slice is not None:  # Crop the data
            sig_axes = self.axes_manager.signal_axes
            sig_axes = np.repeat(sig_axes, 2)
            low_x, high_x, low_y, high_y = [
                _to_hyperspy_index(ind, ax)
                for ind, ax in zip(
                    signal_slice,
                    sig_axes,
                )
            ]
            signal = self.isig[low_x:high_x, low_y:high_y]
        else:
            signal = self

        if "lazy_result" in kwargs:
            warnings.warn(
                "lazy_result was replaced with lazy_output in version 0.14",
                DeprecationWarning,
            )
            lazy_output = kwargs.pop("lazy_result")

        if lazy_output is None:
            lazy_output = signal._lazy

        signal_shape = signal.axes_manager.signal_shape
        origin_coordinates = np.array(signal_shape) / 2

        method_dict = {
            "cross_correlate": find_beam_offset_cross_correlation,
            "blur": find_beam_center_blur,
            "interpolate": find_beam_center_interpolate,
            "center_of_mass": None,
        }

        method_function = _select_method_from_method_dict(
            method, method_dict, print_help=False, **kwargs
        )

        if method == "cross_correlate":
            shifts = signal.map(
                method_function,
                inplace=False,
                output_signal_size=(2,),
                output_dtype=np.float32,
                lazy_output=lazy_output,
                **kwargs,
            )
        elif method == "blur":
            centers = signal.map(
                method_function,
                inplace=False,
                output_signal_size=(2,),
                output_dtype=np.int16,
                lazy_output=lazy_output,
                **kwargs,
            )
            shifts = -centers + origin_coordinates
        elif method == "interpolate":
            centers = signal.map(
                method_function,
                inplace=False,
                output_signal_size=(2,),
                output_dtype=np.float32,
                lazy_output=lazy_output,
                **kwargs,
            )
            shifts = -centers + origin_coordinates
        elif method == "center_of_mass":
            if "mask" in kwargs and signal_slice is not None:
                x, y, r = kwargs["mask"]
                x = x - signal_slice[0]
                y = y - signal_slice[1]
                kwargs["mask"] = (x, y, r)
            centers = signal.center_of_mass(
                lazy_result=lazy_output,
                show_progressbar=False,
                **kwargs,
            )
            shifts = -centers.T + origin_coordinates

        if signal_slice is not None:
            shifted_center = [(low_x + high_x) / 2, (low_y + high_y) / 2]
            unshifted_center = np.array(self.axes_manager.signal_shape) / 2
            shift = np.subtract(unshifted_center, shifted_center)
            shifts = shifts + shift

        shifts.set_signal_type("beam_shift")

        return shifts

    @deprecated_argument(
        name="lazy_result", since="0.15", removal="1.0.0", alternative="lazy_output"
    )
    def center_direct_beam(
        self,
        method=None,
        shifts=None,
        return_shifts=False,
        subpixel=True,
        lazy_output=None,
        align_kwargs=None,
        inplace=True,
        *args,
        **kwargs,
    ):
        """Estimate the direct beam position in each experimentally acquired
        electron diffraction pattern and translate it to the center of the
        image square.

        Parameters
        ----------
        method : str {'cross_correlate', 'blur', 'interpolate', 'center_of_mass'}
            Method used to estimate the direct beam position. The direct
            beam position can also be passed directly with the shifts parameter.
        shifts : Signal, optional
            The position of the direct beam, which can either be passed with this
            parameter (shifts), or calculated on its own.
            Both shifts and the signal need to have the same navigation shape, and
            shifts needs to have one signal dimension with size 2.
        return_shifts : bool, default False
            If True, the values of applied shifts are returned
        subpixel : bool, optional
            If True, the data will be interpolated, allowing for subpixel shifts of
            the diffraction patterns. This can lead to changes in the total intensity
            of the diffraction images, see Notes for more information. If False, the
            data is not interpolated. Default True.
        lazy_output : optional
            If True, the result will be a lazy signal. If False, a non-lazy signal.
            By default, if the signal is lazy, the result will also be lazy.
            If the signal is non-lazy, the result will be non-lazy.
        align_kwargs : dict
            Parameters passed to the alignment function. See scipy.ndimage.shift
            for more information about the parameters.
        *args, **kwargs :
            Additional arguments accepted by :func:`pyxem.utils.diffraction.find_beam_center_blur`,
            :func:`pyxem.utils.diffraction.find_beam_center_interpolate`,
            :func:`pyxem.utils.diffraction.find_beam_offset_cross_correlation`,
            :func:`pyxem.signals.diffraction2d.Diffraction2D.get_direct_beam_position`,
            and :func:`pyxem.signals.diffraction2d.Diffraction2D.center_of_mass`,

        Example
        -------
        >>> s.center_direct_beam(method='blur', sigma=1)

        Using the shifts parameter

        >>> s_shifts = s.get_direct_beam_position(
        ...    method="interpolate", sigma=1, upsample_factor=2, kind="nearest")
        >>> s.center_direct_beam(shifts=s_shifts)

        Notes
        -----
        If the signal has an integer dtype, and subpixel=True is used (the default)
        the total intensity in the diffraction images will most likely not be preserved.
        This is due to ``subpixel=True`` utilizing interpolation. To keep the total intensity
        use a float dtype, which can be done by ``s.change_dtype('float32', rechunk=False)``.

        """
        if (shifts is None) and (method is None):
            raise ValueError("Either method or shifts parameter must be specified")
        if (shifts is not None) and (method is not None):
            raise ValueError(
                "Only one of the shifts or method parameters should be specified, "
                "not both"
            )
        if align_kwargs is None:
            align_kwargs = {}

        if shifts is None:
            shifts = self.get_direct_beam_position(
                method=method, lazy_output=lazy_output, **kwargs
            )
        if "order" not in align_kwargs:
            if subpixel:
                align_kwargs["order"] = 1
            else:
                align_kwargs["order"] = 0
        aligned = self.map(
            _align_single_frame,
            shifts=shifts,
            inplace=inplace,
            lazy_output=lazy_output,
            output_dtype=self.data.dtype,
            output_signal_size=self.axes_manager.signal_shape[::-1],
            **align_kwargs,
        )

        if return_shifts and inplace:
            return shifts
        elif return_shifts:
            return shifts, aligned
        else:
            return aligned

    def threshold_and_mask(self, threshold=None, mask=None, show_progressbar=True):
        """Get a thresholded and masked of the signal.

        Useful for figuring out optimal settings for the
        :meth:`pyxem.signals.Diffraction2D.center_of_mass` method.

        Parameters
        ----------
        threshold : number, optional
            The thresholding will be done at mean times
            this threshold value.
        mask : tuple (x, y, r)
            Round mask centered on x and y, with radius r.
        show_progressbar : bool
            Default True

        Returns
        -------
        s_out : pyxem.signals.Diffraction2D

        Examples
        --------
        >>> s = pxm.data.dummy_data.get_disk_shift_simple_test_signal()
        >>> mask = (25, 25, 10)
        >>> s_out = s.threshold_and_mask(
        ...     mask=mask, threshold=2, show_progressbar=False)
        >>> s_out.plot()

        See Also
        --------
        center_of_mass

        """
        if self._lazy:
            raise NotImplementedError(
                "threshold_and_mask is currently not implemented for "
                "lazy signals. Use compute() first to turn signal into "
                "a non-lazy signal. Note that this will load the full "
                "dataset into memory, which might crash your computer."
            )
        if mask is not None:
            x, y, r = mask
            im_x, im_y = self.axes_manager.signal_shape
            mask = pst._make_circular_mask(x, y, im_x, im_y, r)
        s_out = self.map(
            function=pst._threshold_and_mask_single_frame,
            ragged=False,
            inplace=False,
            show_progressbar=show_progressbar,
            threshold=threshold,
            mask=mask,
        )
        return s_out

    @deprecated_argument(
        since="0.15.0", name="lazy_result", alternative="lazy_output", removal="1.00.0"
    )
    def center_of_mass(
        self,
        threshold=None,
        mask=None,
        **kwargs,
    ):
        """Get the centre of the STEM diffraction pattern using
        center of mass. Threshold can be set to only use the most
        intense parts of the pattern. A mask can be used to exclude
        parts of the diffraction pattern.

        Parameters
        ----------
        threshold : number, optional
            The thresholding will be done at mean times
            this threshold value.
        mask : tuple (x, y, r), optional
            Round mask centered on x and y, with radius r. These are pixel values rather
            than physical units. Default None which means no mask is used.

        Returns
        -------
        DPCSignal
            DPCSignal with beam shifts along the navigation dimension
            and spatial dimensions as the signal dimension(s).

        Examples
        --------
        With mask centered at x=105, y=120 and 30 pixel radius

        >>> s = pxm.data.dummy_data.get_disk_shift_simple_test_signal()
        >>> mask = (25, 25, 10)
        >>> s_com = s.center_of_mass(mask=mask, show_progressbar=False)
        >>> s_color = s_com.get_color_signal()

        Also threshold

        >>> s_com = s.center_of_mass(threshold=1.5, show_progressbar=False)

        Get a lazy signal, then calculate afterwards

        >>> s_com = s.center_of_mass(lazy_result=True, show_progressbar=False)
        >>> s_com.compute(show_progressbar=False)

        """
        if "inplace" in kwargs and kwargs["inplace"]:
            raise ValueError("Inplace is not allowed for center_of_mass")
        else:
            kwargs["inplace"] = False

        det_shape = self.axes_manager.signal_shape
        if mask is not None:
            x, y, r = mask
            mask = pst._make_circular_mask(x, y, det_shape[0], det_shape[1], r)

        ans = self.map(center_of_mass, threshold=threshold, mask=mask, **kwargs)
        ans = ans.T
        ans.set_signal_type("dpc")
        ans.axes_manager.navigation_axes[0].name = "Beam position"
        return ans

    @deprecated_argument(
        name="lazy_result", alternative="lazy_output", since="0.15.0", removal="1.0.0"
    )
    def template_match_disk(self, disk_r=4, inplace=False, **kwargs):
        """Template match the signal dimensions with a disk.

        Used to find diffraction disks in convergent beam electron
        diffraction data.

        Parameters
        ----------
        disk_r : scalar, optional
            Radius of the disk. Default 4.
        lazy_output : bool, default True
            If True, will return a LazyDiffraction2D object. If False,
            will compute the result and return a Diffraction2D object.
        kwargs :
            Passed to :func:`pyxem.utils.diffraction.normalize_template_match`

        Returns
        -------
        template_match : pyxem.signals.Diffraction2D

        Examples
        --------
        >>> s = pxm.data.dummy_data.get_cbed_signal()
        >>> s_template = s.template_match_disk(
        ...     disk_r=5, show_progressbar=False)
        >>> s.plot()

        See Also
        --------
        pyxem.signals.Diffraction2D.template_match_ring
        pyxem.signals.Diffraction2D.template_match
        pyxem.utils.diffraction.normalize_template_match

        """
        disk = morphology.disk(disk_r, self.data.dtype)
        return self.map(
            normalize_template_match, template=disk, inplace=inplace, **kwargs
        )

    @deprecated_argument(
        name="lazy_result", alternative="lazy_output", since="0.15.0", removal="1.0.0"
    )
    def template_match_ring(self, r_inner=5, r_outer=7, inplace=False, **kwargs):
        """Template match the signal dimensions with a ring.

        Used to find diffraction rings in convergent beam electron
        diffraction data.

        Parameters
        ----------
        r_inner, r_outer : scalar, optional
            Inner and outer radius of the rings.
        inplace : bool, optional
            If True, the data is replaced by the filtered data. If False, a
            new signal is returned. Default False.
        kwargs :
            Passed to :func:`pyxem.utils.diffraction.normalize_template_match`

        Returns
        -------
        pyxem.signals.Diffraction2D

        Examples
        --------
        >>> s = pxm.data.dummy_data.get_cbed_signal()
        >>> s_template = s.template_match_ring(show_progressbar=False)
        >>> s.plot()

        See Also
        --------
        pyxem.signals.Diffraction2D.template_match_disk
        pyxem.signals.Diffraction2D.template_match
        pyxem.utils.diffraction.normalize_template_match
        """
        if r_outer <= r_inner:
            raise ValueError(
                "r_outer ({0}) must be larger than r_inner ({1})".format(
                    r_outer, r_inner
                )
            )
        edge = r_outer - r_inner
        edge_slice = np.s_[edge:-edge, edge:-edge]

        ring_inner = morphology.disk(r_inner, dtype=bool)
        ring = morphology.disk(r_outer, dtype=bool)
        ring[edge_slice] = ring[edge_slice] ^ ring_inner
        return self.map(
            normalize_template_match, template=ring, inplace=inplace, **kwargs
        )

    def filter(self, func, inplace=False, **kwargs):
        """Filters the entire dataset given some function applied to the data.

        The function must take a numpy or dask array as input and return a
        numpy or dask array as output which has the same shape, and axes as
        the input.

        Parameters
        ----------
        func : function
            Function to apply to the data. Must take a numpy or dask array as
            input and return a numpy or dask array as output which has the
            same shape as the input.
        inplace : bool, optional
            If True, the data is replaced by the filtered data. If False, a
            new signal is returned. Default False.
        **kwargs :
            Passed to the function.

        Examples
        --------
        >>> import pyxem as pxm
        >>> from scipy.ndimage import gaussian_filter
        >>> s = pxm.data.dummy_data.get_cbed_signal()
        >>> s_filtered = s.filter(gaussian_filter, sigma=1)

        """
        new_data = func(self.data, **kwargs)

        if new_data.shape != self.data.shape:
            raise ValueError(
                "The function must return an array with " "the same shape as the input."
            )
        if inplace:
            self.data = new_data
            return
        else:
            return self._deepcopy_with_new_data(data=new_data)

    def template_match(self, template, inplace=False, **kwargs):
        """Template match the signal dimensions with a binary image.

        Used to find diffraction disks in convergent beam electron
        diffraction data.

        Might also work with non-binary images, but this haven't been
        extensively tested.

        Parameters
        ----------
        template : numpy.ndarray
            The 2D template to match with the signal.
        inplace : bool, optional
            If True, the data is replaced by the filtered data. If False, a
            new signal is returned. Default False.
        **kwargs :
            Any additional keyword arguments to be passed to
            :func:`pyxem.utils.diffraction.normalize_template_match`

        Returns
        -------
        pyxem.signals.Diffraction2D


        Examples
        --------
        >>> import pyxem as pxm
        >>> s = pxm.data.dummy_data.get_cbed_signal()
        >>> binary_image = np.random.randint(0, 2, (6, 6))
        >>> s_template = s.template_match_with_binary_image(
        ...     binary_image, show_progressbar=False)
        >>> s.plot()

        See Also
        --------
        pyxem.signals.Diffraction2D.template_match_disk
        pyxem.signals.Diffraction2D.template_match_ring
        pyxem.utils.diffraction.normalize_template_match

        """

        return self.map(
            normalize_template_match, template=template, inplace=inplace, **kwargs
        )

    @deprecated(since="0.15.0", removal="1.0.0")
    def template_match_with_binary_image(
        self, binary_image, lazy_result=True, show_progressbar=True, **kwargs
    ):
        """Template match the signal dimensions with a binary image.

        Used to find diffraction disks in convergent beam electron
        diffraction data.

        Might also work with non-binary images, but this haven't been
        extensively tested.

        Parameters
        ----------
        binary_image : numpy.ndarray (2-D NumPy array)
        lazy_result : bool, default True
            If True, will return a LazyDiffraction2D object. If False,
            will compute the result and return a Diffraction2D object.
        show_progressbar : bool, default True

        Returns
        -------
        pyxem.signals.Diffraction2D


        Examples
        --------
        >>> s = pxm.data.dummy_data.get_cbed_signal()
        >>> binary_image = np.random.randint(0, 2, (6, 6))
        >>> s_template = s.template_match_with_binary_image(
        ...     binary_image, show_progressbar=False)
        >>> s.plot()

        See Also
        --------
        pyxem.signals.Diffraction2D.template_match_disk
        pyxem.signals.Diffraction2D.template_match_ring

        """
        return self.template_match(
            template=binary_image,
            lazy_output=lazy_result,
            show_progressbar=show_progressbar,
            **kwargs,
        )

    def get_diffraction_vectors(
        self,
        center=None,
        calibration=None,
        column_names=None,
        units=None,
        get_intensity=True,
        **kwargs,
    ):
        """Find vectors from the diffraction pattern. Wraps `hyperspy.api.signals.Signal2D.find_peaks`

        Parameters
        ----------
        center: None or tuple
            The center of the diffraction pattern, if None, will use the
            center determined by the offsets in the signal axes
        calibration: None or tuple
            The calibration of the diffraction pattern, if None, will use the
            scales in the signal axes
        column_names: None or tuple
            The column names of the vectors, if None, will use the names
            of the signal axes
        units: None or tuple
            The units of the vectors, if None, will use the units of the
            signal axes
        get_intensity: bool
            If True, will return the intensity of the peaks as well as the
            intensity of the peaks. Default True.
        kwargs:
            Passed to the peak finding function.
        """
        from pyxem.signals import DiffractionVectors

        vectors = super().find_peaks(
            interactive=False, get_intensity=get_intensity, **kwargs
        )
        vectors = DiffractionVectors.from_peaks(
            vectors,
            center=center,
            calibration=calibration,
            column_names=column_names,
            units=units,
        )
        return vectors

    def peak_position_refinement_com(
        self, peak_array, square_size=10, lazy_result=True, show_progressbar=True
    ):
        """Refines the peak position using the center of mass.

        Parameters
        ----------
        peak_array : Numpy or Dask array
            Object with x and y coordinates of the peak positions.
            Must have the same dimensions as this signal's navigation
            dimensions.
        square_size : int
            Even integer, sub image from which the center of mass is
            calculated. Default 5.
        lazy_result : bool, default True
            If True, will return a LazyDiffraction2D object. If False,
            will compute the result and return a Diffraction2D object.
        show_progressbar : bool, default True

        Returns
        -------
        output_array : dask 2D object array
            Same size as the two last dimensions in data, in the form
            [[y0, x0], [y1, x1], ...].
            The peak positions themselves are stored in 2D NumPy arrays
            inside each position in peak_array. This is done instead of
            making a 4D NumPy array, since the number of found peaks can
            vary in each position.

        Examples
        --------
        >>> s = pxm.data.dummy_data.get_cbed_signal()
        >>> peak_array = s.find_peaks_lazy()
        >>> refined_peak_array = s.peak_position_refinement_com(peak_array, 20)
        >>> refined_peak_array_com = refined_peak_array.compute(
        ...     show_progressbar=False)
        >>> s.add_peak_array_as_markers(refined_peak_array_com)
        >>> s.plot()

        """
        if square_size % 2 != 0:  # If odd number, raise error
            raise ValueError(
                "square_size must be even number, not {0}".format(square_size)
            )
        dask_array = _get_dask_array(self)

        chunks_peak = dask_array.chunks[:-2]
        if hasattr(peak_array, "chunks"):
            peak_array_dask = da.rechunk(peak_array, chunks=chunks_peak)
        else:
            peak_array_dask = da.from_array(peak_array, chunks=chunks_peak)

        output_array = dt._peak_refinement_centre_of_mass(
            dask_array, peak_array_dask, square_size
        )

        if not lazy_result:
            if show_progressbar:
                pbar = ProgressBar()
                pbar.register()
            output_array = output_array.compute()
            if show_progressbar:
                pbar.unregister()
        return output_array

    def intensity_peaks(
        self, peak_array, disk_r=4, lazy_result=True, show_progressbar=True
    ):
        """Get intensity of a peak in the diffraction data.

        The intensity is calculated by taking the mean of the
        pixel values inside radius disk_r from the peak
        position.

        Parameters
        ----------
        peak_array : Numpy or Dask array
            Must have the same navigation shape as this signal.
        disk_r : int
            Radius of the disc chosen to take the mean value of
        lazy_result : bool, default True
            If True, will return a :class:`pyxem.signals.diffraction2d.LazyDiffraction2D` object. If False,
            will compute the result and return a Diffraction2D object.
        show_progressbar : bool, default True

        Returns
        -------
        intensity_array: Numpy or Dask array
            Same navigation shape as this signal, with peak position in
            x and y coordinates and the mean intensity.

        Examples
        --------
        >>> s = pxm.data.dummy_data.get_cbed_signal()
        >>> peak_array = s.find_peaks_lazy()
        >>> intensity_array = s.intensity_peaks(peak_array, disk_r=6)
        >>> intensity_array_computed = intensity_array.compute()

        """
        dask_array = _get_dask_array(self)

        chunks_peak = dask_array.chunks[:-2]
        if hasattr(peak_array, "chunks"):
            peak_array_dask = da.rechunk(peak_array, chunks=chunks_peak)
        else:
            peak_array_dask = da.from_array(peak_array, chunks=chunks_peak)

        output_array = dt._intensity_peaks_image(dask_array, peak_array_dask, disk_r)

        if not lazy_result:
            if show_progressbar:
                pbar = ProgressBar()
                pbar.register()
            output_array = output_array.compute()
            if show_progressbar:
                pbar.unregister()
        return output_array

    """ Plotting (or plotting adjacent) methods """

    def make_probe_navigation(self, method="fast"):
        nav_dim = self.axes_manager.navigation_dimension
        if (0 == nav_dim) or (nav_dim > 2):
            raise ValueError(
                "Probe navigation can only be made for signals with 1 or 2 "
                "navigation dimensions"
            )
        if method == "fast":
            x = round(self.axes_manager.signal_shape[0] / 2)
            y = round(self.axes_manager.signal_shape[1] / 2)
            if self._lazy:
                isig_slice = _get_signal_dimension_host_chunk_slice(
                    x, y, self.data.chunks
                )
            else:
                isig_slice = np.s_[x, y]
            s = self.isig[isig_slice]
        elif method == "slow":
            s = self
        s_nav = s.T.sum()
        if s_nav._lazy:
            s_nav.compute()
        self._navigator_probe = s_nav

    def plot(self, *args, **kwargs):
        if "navigator" in kwargs:
            super().plot(*args, **kwargs)
        elif self.axes_manager.navigation_dimension > 2:
            super().plot(*args, **kwargs)
        elif self.axes_manager.navigation_dimension == 0:
            super().plot(*args, **kwargs)
        else:
            if hasattr(self, "_navigator_probe"):
                nav_sig_shape = self._navigator_probe.axes_manager.shape
                self_nav_shape = self.axes_manager.navigation_shape
                if nav_sig_shape != self_nav_shape:
                    raise ValueError(
                        "navigation_signal does not have the same shape "
                        "({0}) as the signal's navigation shape "
                        "({1})".format(nav_sig_shape, self_nav_shape)
                    )
            else:
                if self._lazy:
                    method = "fast"
                else:
                    method = "slow"
                self.make_probe_navigation(method=method)
            s_nav = self._navigator_probe
            kwargs["navigator"] = s_nav
            super().plot(*args, **kwargs)

    @deprecated(since="0.16.0", removal="1.0.0")
    def add_peak_array_as_markers(self, peak_array, permanent=True, **kwargs):
        """Add a peak array to the signal as HyperSpy markers.

        Parameters
        ----------
        peak_array : NumPy 4D array
        color : string, optional
            Default 'red'
        size : scalar, optional
            Default 20
        permanent : bool, optional
            Default False, if True the markers will be added to the
            signal permanently.
        **kwargs :
            Passed to :class:`hyperspy.api.plot.markers.Points`

        Examples
        --------
        >>> s, parray = pxm.data.dummy_data.get_simple_ellipse_signal_peak_array()
        >>> s.add_peak_array_as_markers(parray)
        >>> s.plot()

        """
        if isinstance(peak_array, np.ndarray):
            if peak_array.dtype == object:
                markers = hs.plot.markers.Points(peak_array.T, **kwargs)
            else:
                markers = hs.plot.markers.Points(peak_array, **kwargs)
        elif isinstance(peak_array, BaseSignal):
            markers = hs.plot.markers.Points.from_signal(peak_array, **kwargs)
        else:
            raise TypeError("peak_array must be a NumPy array or a HyperSpy signal")
        self.add_marker(markers, permanent=permanent)

    def add_ellipse_array_as_markers(
        self,
        ellipse_array,
        inlier_array=None,
        peak_array=None,
    ):
        """Add a ellipse parameters array to a signal as HyperSpy markers.

        Useful to visualize the ellipse results.

        Parameters
        ----------
        ellipse_array : NumPy array
            Array with ellipse parameters in the form (xc, yc, semi0, semi1, rot)
        inlier_array : NumPy array, optional
            The inlier array is a boolean array returned by the
            :meth:`~pyxem.utils.ransac_ellipse_tools.determine_ellipse` algorithm to indicate which
            points were used to fit the ellipse.
        peak_array : NumPy array, optional
            All of the points used to fit the ellipse.

        Examples
        --------
        >>> s, _ = pxm.data.dummy_data.get_simple_ellipse_signal_peak_array()
        >>> ellipse_array = [128, 128, 20, 20, 0] # (xc, yc, semi0, semi1, rot)
        >>> ellipse_array = [[128, ]]
        >>> s.plot()
        >>> e = s.add_ellipse_array_as_markers(ellipse_array)


        """
        if len(self.data.shape) != 4:
            raise ValueError("Signal must be 4 dims to use this function")

        markers = ret.ellipse_to_markers(
            ellipse_array, inlier=inlier_array, points=peak_array
        )

        self.add_marker(markers)

    def angular_mask(self, angle0, angle1, centre_x_array=None, centre_y_array=None):
        """Get a bool array with True values between angle0 and angle1.
        Will use the (0, 0) point as given by the signal as the centre,
        giving an "angular" slice. Useful for analysing anisotropy in
        diffraction patterns.

        Parameters
        ----------
        angle0, angle1 : numbers
        centre_x_array, centre_y_array : NumPy 2D array, optional
            Has to have the same shape as the navigation axis of
            the signal.

        Returns
        -------
        mask_array : numpy.ndarray
            The True values will be the region between angle0 and angle1.
            The array will have the same dimensions as the signal.

        Examples
        --------
        >>> s = pxm.data.dummy_data.get_holz_simple_test_signal()
        >>> s.axes_manager.signal_axes[0].offset = -25
        >>> s.axes_manager.signal_axes[1].offset = -25
        >>> mask_array = s.angular_mask(0.5*np.pi, np.pi)

        """

        bool_array = pst._get_angle_sector_mask(
            self,
            angle0,
            angle1,
            centre_x_array=centre_x_array,
            centre_y_array=centre_y_array,
        )
        return bool_array

    """ Variance generation methods """

    def get_variance(
        self,
        npt,
        method="Omega",
        dqe=None,
        spatial=False,
        navigation_axes=None,
        **kwargs,
    ):
        """Calculates the variance using one of the methods described in [1]. A shot noise correction
           and specification of axes to operate over are also possible.

        Parameters
        ----------
        npt : int
            The number of points to use in the azimuthal integration
        method : 'Omega' or 'r' or 're' or 'VImage', optional
            The method used to calculate the variance. Details in [1]
        dqe : int, optional
            The detector quantum efficiency or the pixel value for one electron.
        spatial : bool, optional
            Included intermediate spatial variance in output (only available if method=='r')
        navigation_axes : list or none, optional
            The axes to calculate the variance over.  The default is to use the navigation axes.
        **kwargs: dict
            Any keywords accepted for the :func:`pyxem.signals.Diffraction2D.get_azimuthal_integral1d` or
            :func:`pyxem.signals.Diffraction2D.get_azimuthal_integral2d` function

        Returns
        -------
        variance : array-like
            Calculate variance as it's own signal

        References
        ----------
        [1] Daulton, T. L et al, Ultramicroscopy, 110(10), 1279–1289, https://doi.org/10.1016/j.ultramic.2010.05.010
            Nanobeam diffraction fluctuation electron microscopy technique for structural characterization of disordered
            materials-Application to Al88-xY7Fe5Tix metallic glasses.

        See Also
        --------
        pyxem.signals.Diffraction2D.get_azimuthal_integral1d
        pyxem.signals.Diffraction2D.get_azimuthal_integral2d
        """

        if method not in ["Omega", "r", "re", "VImage"]:
            raise ValueError(
                "Method must be one of [Omega, r, re, VImage]."
                "for more information read\n"
                "Daulton, T. L., Bondi, K. S., & Kelton, K. F. (2010)."
                " Nanobeam diffraction fluctuation electron microscopy"
                " technique for structural characterization of disordered"
                " materials-Application to Al88-xY7Fe5Tix metallic glasses."
                " Ultramicroscopy, 110(10), 1279–1289.\n"
                " https://doi.org/10.1016/j.ultramic.2010.05.010"
            )

        if method == "Omega":
            one_d_integration = self.get_azimuthal_integral1d(
                npt=npt, mean=True, **kwargs
            )
            variance = (
                (one_d_integration**2).mean(axis=navigation_axes)
                / one_d_integration.mean(axis=navigation_axes) ** 2
            ) - 1
            if dqe is not None:
                sum_points = self.get_azimuthal_integral1d(npt=npt, **kwargs).mean(
                    axis=navigation_axes
                )
                variance = variance - ((sum_points**-1) * dqe)

        elif method == "r":
            one_d_integration = self.get_azimuthal_integral1d(
                npt=npt, mean=True, **kwargs
            )
            integration_squared = (self**2).get_azimuthal_integral1d(
                mean=True, npt=npt, **kwargs
            )
            # Full variance is the same as the unshifted phi=0 term in angular correlation
            full_variance = (integration_squared / one_d_integration**2) - 1

            if dqe is not None:
                full_variance = full_variance - ((one_d_integration**-1) * dqe)

            variance = full_variance.mean(axis=navigation_axes)

            if spatial:
                return variance, full_variance

        elif method == "re":
            one_d_integration = self.get_azimuthal_integral1d(
                npt=npt, mean=True, **kwargs
            ).mean(axis=navigation_axes)
            integration_squared = (
                (self**2)
                .get_azimuthal_integral1d(npt=npt, mean=True, **kwargs)
                .mean(axis=navigation_axes)
            )
            variance = (integration_squared / one_d_integration**2) - 1

            if dqe is not None:
                sum_int = self.get_azimuthal_integral1d(npt=npt, **kwargs).mean()
                variance = variance - (sum_int**-1) * (1 / dqe)

        elif method == "VImage":
            variance_image = (
                (self**2).mean(axis=navigation_axes)
                / self.mean(axis=navigation_axes) ** 2
            ) - 1
            if dqe is not None:
                variance_image = variance_image - (
                    self.sum(axis=navigation_axes) ** -1
                ) * (1 / dqe)
            variance = variance_image.get_azimuthal_integral1d(
                npt=npt, mean=True, **kwargs
            )
        return variance

    """ Methods associated with radial integration, not pyFAI based """

    @deprecated(
        since="0.17",
        alternative="pyxem.signals.diffraction2d.azimuthal_integral2d",
        removal="1.0.0",
    )
    def angular_slice_radial_average(
        self,
        angleN=20,
        centre_x=None,
        centre_y=None,
        slice_overlap=None,
        show_progressbar=True,
    ):
        """Do radial average of different angular slices.
        Useful for analysing anisotropy in round diffraction features,
        such as diffraction rings from polycrystalline materials or
        higher order Laue zone rings.

        Parameters
        ----------
        angleN : int, default 20
            Number of angular slices. If angleN=4, each slice
            will be 90 degrees. The average will start in the top left
            corner (0, 0) when plotting using s.plot(), and go clockwise.
        centre_x, centre_y : int or NumPy array, optional
            If given as int, all the diffraction patterns will have the same
            centre position. Each diffraction pattern can also have different
            centre position, by passing a NumPy array with the same dimensions
            as the navigation axes.
            Note: in either case both x and y values must be given. If one is
            missing, both will be set from the signal (0., 0.) positions.
            If no values are given, the (0., 0.) positions in the signal will
            be used.
        slice_overlap : float, optional
            Amount of overlap between the slices, given in fractions of
            angle slice (0 to 1). For angleN=4, each slice will be 90
            degrees. If slice_overlap=0.5, each slice will overlap by 45
            degrees on each side. The range of the slices will then be:
            (-45, 135), (45, 225), (135, 315) and (225, 45).
            Default off: meaning there is no overlap between the slices.
        show_progressbar : bool
            Default True

        Returns
        -------
        signal : HyperSpy 1D signal
            With one more navigation dimensions (the angular slices) compared
            to the input signal.

        Examples
        --------
        >>> s = pxm.data.dummy_data.get_holz_simple_test_signal()
        >>> s_com = s.center_of_mass(show_progressbar=False)
        >>> s_ar = s.angular_slice_radial_average(
        ...     angleN=10, centre_x=s_com.inav[0].data,
        ...     centre_y=s_com.inav[1].data, slice_overlap=0.2,
        ...     show_progressbar=False)
        >>> s_ar.plot() # doctest: +SKIP

        """
        signal_list = []
        angle_list = []
        if slice_overlap is None:
            slice_overlap = 0
        else:
            if (slice_overlap < 0) or (slice_overlap > 1):
                raise ValueError(
                    "slice_overlap is {0}. But must be between "
                    "0 and 1".format(slice_overlap)
                )
        angle_step = 2 * np.pi / angleN
        for i in range(angleN):
            angle0 = (angle_step * i) - (angle_step * slice_overlap)
            angle1 = (angle_step * (i + 1)) + (angle_step * slice_overlap)
            angle_list.append((angle0, angle1))
        if (centre_x is None) or (centre_y is None):
            centre_x, centre_y = pst._make_centre_array_from_signal(self)
        elif (not isiterable(centre_x)) or (not isiterable(centre_y)):
            centre_x, centre_y = pst._make_centre_array_from_signal(
                self, x=centre_x, y=centre_y
            )

        for angle in tqdm(angle_list, disable=(not show_progressbar)):
            mask_array = self.angular_mask(
                angle[0], angle[1], centre_x_array=centre_x, centre_y_array=centre_y
            )
            s_r = self.radial_average(
                centre_x=centre_x,
                centre_y=centre_y,
                mask_array=mask_array,
                show_progressbar=show_progressbar,
            )
            signal_list.append(s_r)
        angle_scale = angle_list[1][1] - angle_list[0][1]
        signal = hs.stack(signal_list, new_axis_name="Angle slice")
        signal.axes_manager["Angle slice"].offset = angle_scale / 2
        signal.axes_manager["Angle slice"].scale = angle_scale
        signal.axes_manager["Angle slice"].units = "Radians"
        signal.axes_manager[-1].name = "Scattering angle"
        return signal

    """ Methods associated with radial integration, pyFAI based """

    @property
    def ai(self):
        try:
            return self.metadata.Signal["ai"]
        except AttributeError:
            raise ValueError("ai property is not currently set")

    @deprecated(
        since="0.18",
        removal="1.0.0",
        alternative="pyxem.signals.diffraction2d.calibration",
    )
    def set_ai(
        self, center=None, wavelength=None, affine=None, radial_range=None, **kwargs
    ):
        """This function sets the .ai parameter which stores an ~pyfai.AzimuthalIntegrator object based on
        the current calibration applied to the diffraction pattern.

        Parameters
        --------
        center: (x,y) or None
            The center of the diffraction pattern.  If None, the center is the middle of the image.
        wavelength: float
            The wavelength of the energy in 1/meters.  For proper treatment of Ewald Sphere
        affine: numpy.Array 3x3
            A 3x3 array which describes the affine distortion of the pattern. This is translated
            to a spline interpolation which is used in the pyFAI implementations
        radial_range: (start,stop)
            The start and stop of the radial range in real units

        Returns
        -------
        None :
            The metadata item Signal.ai is set

        """
        if wavelength is None and self.unit not in ["2th_deg", "2th_rad"]:
            raise ValueError(
                "if the unit is not '2th_deg' or '2th_rad' then a wavelength must be given."
            )

        pixel_scale = [
            self.axes_manager.signal_axes[0].scale,
            self.axes_manager.signal_axes[1].scale,
        ]

        sig_shape = self.axes_manager.signal_shape
        setup = _get_setup(wavelength, self.unit, pixel_scale, radial_range)
        detector, dist, radial_range = setup
        ai = get_azimuthal_integrator(
            detector=detector,
            detector_distance=dist,
            shape=sig_shape,
            center=center,
            affine=affine,
            wavelength=wavelength,
            **kwargs,
        )
        self.metadata.set_item("Signal.ai", ai)
        return None

    @deprecated_argument(
        name="lazy_result", since="0.14", removal="1.0.0", alternative="lazy_output"
    )
    def get_azimuthal_integral1d(
        self,
        npt,
        mask=None,
        radial_range=None,
        azimuth_range=None,
        inplace=False,
        method="splitpixel_pyxem",
        sum=False,
        **kwargs,
    ):
        """Creates a polar reprojection using pyFAI's azimuthal integrate 2d. This method is designed
        with 2 cases in mind. (1) the signal has pyxem style units, if a wavelength is not provided
        no account is made for the curvature of the Ewald sphere. (2) the signal has pyFAI style units,
        in which case the detector kwarg must be provided.

        Parameters
        ----------
        npt : int
            The number of radial points to calculate
        mask :  boolean array or BaseSignal
            A boolean mask to apply to the data to exclude some points.
            If mask is a BaseSignal then it is iterated over as well.
        radial_range : None or (float, float)
            The radial range over which to perform the integration. Default is
            the full frame
        azimuth_range : None or (float, float)
            The azimuthal range over which to perform the integration. Default is
            from -pi to pi
        inplace : bool
            If the signal is overwritten or copied to a new signal
        method : str
             Can be “numpy”, “cython”, “BBox” or “splitpixel”, “lut”, “csr”,
             “nosplit_csr”, “full_csr”, “lut_ocl” and “csr_ocl” if you want
             to go on GPU. To Specify the device: “csr_ocl_1,2”
        sum : bool
            If true returns the pixel split sum rather than the azimuthal integration which
            gives the mean.

        Other Parameters
        -------
        dummy : float
            Value for dead/masked pixels
        delta_dummy : float
            Precision value for dead/masked pixels
        correctSolidAngle : bool
            Correct for the solid angle of each pixel if True
        dark : ndarray
            The dark noise image
        flat : ndarray
            The flat field correction image
        safe : bool
            Do some extra checks to ensure LUT/CSR is still valid. False is faster.

        Returns
        -------
        integration : Diffraction1D
            A 1D diffraction signal

        Examples
        --------
        Basic case using "2th_deg" units (no wavelength needed)

        >>> ds.unit = "2th_deg"
        >>> ds.get_azimuthal_integral1d(npt=100)

        Basic case using a curved Ewald Sphere approximation and pyXEM units
        (wavelength needed)

        >>> ds.unit = "k_nm^-1" # setting units
        >>> ds.set_ai(wavelength=2.5e-12) # creating an AzimuthalIntegrator Object
        >>> ds.get_azimuthal_integral1d(npt=100)

        See Also
        --------
        pyxem.signals.Diffraction2D.get_azimuthal_integral2d
        """
        usepyfai = method not in ["splitpixel_pyxem"]
        if not usepyfai:
            # get_slices1d should be sped up in the future by
            # getting rid of shapely and using numba on the for loop
            indexes, facts, factor_slices, radial_range = self.calibration.get_slices1d(
                npt, radial_range=radial_range
            )
            if mask is None:
                mask = self.calibration.mask
            integration = self.map(
                _slice_radial_integrate1d,
                indexes=indexes,
                factors=facts,
                factor_slices=factor_slices,
                inplace=inplace,
                mask=mask,
                **kwargs,
            )
        else:
            if "wavelength" in kwargs:
                warnings.warn(
                    "The wavelength parameter was removed in 0.14. The wavelength "
                    "can be set using the `set_ai` function or using `s.beam_energy`"
                    " for `ElectronDiffraction2D` signals"
                )
                kwargs.pop("wavelength")

            sig_shape = self.axes_manager.signal_shape
            if radial_range is None:
                radial_range = _get_radial_extent(
                    ai=self.ai, shape=sig_shape, unit=self.unit
                )
                radial_range[0] = 0
            integration = self.map(
                azimuthal_integrate1d,
                azimuthal_integrator=self.ai,
                npt_rad=npt,
                azimuth_range=azimuth_range,
                radial_range=radial_range,
                method=method,
                inplace=inplace,
                unit=self.unit,
                mask=mask,
                sum=sum,
                **kwargs,
            )
        s = self if inplace else integration
        k_axis = s.axes_manager.signal_axes[0]
        if not isinstance(k_axis, UniformDataAxis):
            k_axis.convert_to_uniform_axis()
        # Dealing with axis changes
        k_axis.name = "Radius"
        k_axis.scale = (radial_range[1] - radial_range[0]) / npt
        k_axis.offset = radial_range[0]

        return integration

    def get_azimuthal_integral2d(
        self,
        npt,
        npt_azim=360,
        mask=None,
        radial_range=None,
        azimuth_range=None,
        inplace=False,
        method="splitpixel_pyxem",
        sum=False,
        correctSolidAngle=True,
        **kwargs,
    ):
        """Creates a polar reprojection using pyFAI's azimuthal integrate 2d. This method is designed
        with 2 cases in mind. (1) the signal has pyxem style units, if a wavelength is not provided
        no account is made for the curvature of the Ewald sphere. (2) the signal has pyFAI style units,
        in which case the detector kwarg must be provided.

        Parameters
        ----------
        npt: int
            The number of radial points to calculate
        npt_azim: int
            The number of azimuthal points to calculate
        mask:  boolean array or BaseSignal
            A boolean mask to apply to the data to exclude some points.
            If mask is a BaseSignal then it is iterated over as well.
        radial_range: None or (float, float)
            The radial range over which to perform the integration. Default is
            the full frame
        azimuth_range:None or (float, float)
            The azimuthal range over which to perform the integration. Default is
            from -pi to pi
        inplace: bool
            If the signal is overwritten or copied to a new signal
        method: str
            Can be “numpy”, “cython”, “BBox” or “splitpixel”, “lut”, “csr”,
            “nosplit_csr”, “full_csr”, “lut_ocl” and “csr_ocl” if you want
            to go on GPU. To Specify the device: “csr_ocl_1,2”.  For pure
            pyxem based methods use "splitpixel_pyxem".
        sum: bool
            If true the radial integration is returned rather then the Azimuthal Integration.
        correctSolidAngle: bool
            Account for Ewald sphere or not. From PYFAI.

        Other Parameters
        -------
        dummy: float
            Value for dead/masked pixels
        delta_dummy: float
            Percision value for dead/masked pixels
        correctSolidAngle: bool
            Correct for the solid angle of each pixel if True
        dark: ndarray
            The dark noise image
        flat: ndarray
            The flat field correction image
        safe: bool
            Do some extra checks to ensure LUT/CSR is still valid. False is faster.
        show_progressbar: bool
            If True shows a progress bar for the mapping function
        max_workers: int
            The number of streams to initialize.

        Returns
        -------
        polar: PolarDiffraction2D
            A polar diffraction signal, when inplace is False, otherwise None

        Examples
        --------
        Basic case using "2th_deg" units (no wavelength needed)

        >>> ds.unit = "2th_deg"
        >>> ds.get_azimuthal_integral2d(npt_rad=100)

        Basic case using a curved Ewald Sphere approximation and pyXEM units
        (wavelength needed)

        >>> ds.unit = "k_nm^-1" # setting units
        >>> ds.set_ai(wavelength=2.5e-12)
        >>> ds.get_azimuthal_integral2d(npt_rad=100)

        See Also
        --------
        pyxem.signals.Diffraction2D.get_azimuthal_integral1d

        """
        if azimuth_range is None:
            azimuth_range = (-np.pi, np.pi)

        usepyfai = method not in ["splitpixel_pyxem"]
        if not usepyfai:
            # get_slices2d should be sped up in the future by
            # getting rid of shapely and using numba on the for loop
            slices, factors, factors_slice, radial_range = (
                self.calibration.get_slices2d(
                    npt,
                    npt_azim,
                    radial_range=radial_range,
                    azimuthal_range=azimuth_range,
                )
            )
            if self._gpu and CUPY_INSTALLED:  # pragma: no cover
                from pyxem.utils._azimuthal_integrations import (
                    _slice_radial_integrate_cupy,
                )

                slices = cp.asarray(slices)
                factors = cp.asarray(factors)
                factors_slice = cp.asarray(factors_slice)
                integration = self._blockwise(
                    _slice_radial_integrate_cupy,
                    slices=slices,
                    factors=factors,
                    factors_slice=factors_slice,
                    npt=npt,
                    npt_azim=npt_azim,
                    inplace=inplace,
                    signal_shape=(npt, npt_azim),
                    mask=mask,
                    dtype=float,
                    **kwargs,
                )
            else:
                if mask is None:
                    mask = self.calibration.mask
                integration = self.map(
                    _slice_radial_integrate,
                    slices=slices,
                    factors=factors,
                    factors_slice=factors_slice,
                    npt_rad=npt,
                    npt_azim=npt_azim,
                    inplace=inplace,
                    mask=mask,
                    **kwargs,
                )

        else:
            sig_shape = self.axes_manager.signal_shape
            if radial_range is None:
                radial_range = _get_radial_extent(
                    ai=self.ai, shape=sig_shape, unit=self.unit
                )
                radial_range[0] = 0
            integration = self.map(
                azimuthal_integrate2d,
                azimuthal_integrator=self.ai,
                npt_rad=npt,
                npt_azim=npt_azim,
                azimuth_range=azimuth_range,
                radial_range=radial_range,
                method=method,
                inplace=inplace,
                unit=self.unit,
                mask=mask,
                sum=sum,
                correctSolidAngle=correctSolidAngle,
                **kwargs,
            )

        s = self if inplace else integration
        s.set_signal_type("polar_diffraction")

        # Dealing with axis changes
        t_axis = s.axes_manager.signal_axes[0]
        k_axis = s.axes_manager.signal_axes[1]
        if not isinstance(k_axis, UniformDataAxis):
            k_axis.convert_to_uniform_axis()
        if not isinstance(t_axis, UniformDataAxis):
            t_axis.convert_to_uniform_axis()

        t_axis.name = "Radians"
        t_axis.units = "Rad"
        t_axis.scale = (azimuth_range[1] - azimuth_range[0]) / npt_azim
        t_axis.offset = azimuth_range[0]

        k_axis.name = "Radius"
        k_axis.scale = (radial_range[1] - radial_range[0]) / npt
        k_axis.offset = radial_range[0]

        return integration

    def get_radial_integral(
        self,
        npt,
        npt_rad,
        mask=None,
        radial_range=None,
        azimuth_range=None,
        inplace=False,
        method="splitpixel",
        sum=False,
        correctSolidAngle=True,
        **kwargs,
    ):
        """Calculate the radial integrated profile curve as I = f(chi)

        Parameters
        ----------
        npt: int
            The number of radial points to calculate
        npt_rad: int
            number of points in the radial space. Too few points may lead to huge rounding errors.
        mask:  boolean array or BaseSignal
            A boolean mask to apply to the data to exclude some points.
            If mask is a BaseSignal then it is iterated over as well.
        radial_range: None or (float, float)
            The radial range over which to perform the integration. Default is
            the full frame
        azimuth_range:None or (float, float)
            The azimuthal range over which to perform the integration. Default is
            from -pi to pi
        inplace: bool
            If the signal is overwritten or copied to a new signal
        method: str
            Can be “numpy”, “cython”, “BBox” or “splitpixel”, “lut”, “csr”,
            “nosplit_csr”, “full_csr”, “lut_ocl” and “csr_ocl” if you want
            to go on GPU. To Specify the device: “csr_ocl_1,2”
        sum: bool
            If true the radial integration is returned rather then the Azimuthal Integration.
        correctSolidAngle: bool
            Account for Ewald sphere or not. From PYFAI.

        Other Parameters
        -------
        dummy: float
            Value for dead/masked pixels
        delta_dummy: float
            Percision value for dead/masked pixels
        correctSolidAngle: bool
            Correct for the solid angle of each pixel if True
        dark: ndarray
            The dark noise image
        flat: ndarray
            The flat field correction image
        safe: bool
            Do some extra checks to ensure LUT/CSR is still valid. False is faster.
        show_progressbar: bool
            If True shows a progress bar for the mapping function

        Returns
        -------
        polar: PolarDiffraction2D
            A polar diffraction signal

        Examples
        --------
        Basic case using "2th_deg" units (no wavelength needed)

        >>> ds.unit = "2th_deg"
        >>> ds.set_ai()
        >>> ds.get_radial_integral(npt=100, npt_rad=400)

        Basic case using a curved Ewald Sphere approximation and pyXEM units
        (wavelength needed)

        >>> ds.unit = "k_nm^-1" # setting units
        >>> ds.set_ai(wavelength=2.5e-12)
        >>> ds.get_radial_integral(npt=100,npt_rad=400)

        """
        sig_shape = self.axes_manager.signal_shape
        if radial_range is None:
            radial_range = _get_radial_extent(
                ai=self.ai, shape=sig_shape, unit=self.unit
            )
            radial_range[0] = 0
        integration = self.map(
            integrate_radially,
            azimuthal_integrator=self.ai,
            npt=npt,
            npt_rad=npt_rad,
            azimuth_range=azimuth_range,
            radial_range=radial_range,
            method=method,
            inplace=inplace,
            radial_unit=self.unit,
            mask=mask,
            sum=sum,
            correctSolidAngle=correctSolidAngle,
            **kwargs,
        )

        s = self if inplace else integration

        # Dealing with axis changes
        k_axis = s.axes_manager.signal_axes[0]
        k_axis.name = "Radius"
        k_axis.scale = (radial_range[1] - radial_range[0]) / npt
        k_axis.offset = radial_range[0]

        return integration

    def get_medfilt1d(
        self,
        npt_rad=1028,
        npt_azim=512,
        mask=None,
        inplace=False,
        method="splitpixel",
        sum=False,
        correctSolidAngle=True,
        **kwargs,
    ):
        """Calculate the radial integrated profile curve as I = f(chi)

        Parameters
        ----------
        npt_rad: int
             The number of radial points.
        npt_azim: int
             The number of radial points
        mask:  boolean array or BaseSignal
            A boolean mask to apply to the data to exclude some points.
            If mask is a BaseSignal then it is iterated over as well.
        inplace: bool
            If the signal is overwritten or copied to a new signal
        method: str
            Can be “numpy”, “cython”, “BBox” or “splitpixel”, “lut”, “csr”,
            “nosplit_csr”, “full_csr”, “lut_ocl” and “csr_ocl” if you want
            to go on GPU. To Specify the device: “csr_ocl_1,2”
        sum: bool
            If true the radial integration is returned rather then the Azimuthal Integration.
        correctSolidAngle: bool
            Account for Ewald sphere or not. From PYFAI.

        Other Parameters
        -------
        dummy: float
            Value for dead/masked pixels
        delta_dummy: float
            Percision value for dead/masked pixels
        correctSolidAngle: bool
            Correct for the solid angle of each pixel if True
        dark: ndarray
            The dark noise image
        flat: ndarray
            The flat field correction image
        safe: bool
            Do some extra checks to ensure LUT/CSR is still valid. False is faster.
        show_progressbar: bool
            If True shows a progress bar for the mapping function


        Returns
        -------
        polar: PolarDiffraction2D
            A polar diffraction signal

        Examples
        --------
        Basic case using "2th_deg" units (no wavelength needed)

        >>> ds.unit = "2th_deg"
        >>> ds.set_ai()
        >>> ds.get_radial_integral(npt=100, npt_rad=400)

        Basic case using a curved Ewald Sphere approximation and pyXEM units
        (wavelength needed)

        >>> ds.unit = "k_nm^-1" # setting units
        >>> ds.set_ai(wavelength=2.5e-12)
        >>> ds.get_radial_integral(npt=100,npt_rad=400)

        """
        sig_shape = self.axes_manager.signal_shape
        radial_range = _get_radial_extent(ai=self.ai, shape=sig_shape, unit=self.unit)
        radial_range[0] = 0
        integration = self.map(
            medfilt_1d,
            azimuthal_integrator=self.ai,
            npt_rad=npt_rad,
            npt_azim=npt_azim,
            method=method,
            inplace=inplace,
            unit=self.unit,
            mask=mask,
            correctSolidAngle=correctSolidAngle,
            **kwargs,
        )

        s = self if inplace else integration

        # Dealing with axis changes
        k_axis = s.axes_manager.signal_axes[0]
        k_axis.name = "Radius"
        k_axis.scale = (radial_range[1] - radial_range[0]) / npt_rad
        # k_axis.units = unit.unit_symbol
        k_axis.offset = radial_range[0]

        return integration

    def sigma_clip(
        self,
        npt_rad=1028,
        npt_azim=512,
        mask=None,
        thres=3,
        max_iter=5,
        inplace=False,
        method="splitpixel",
        sum=False,
        correctSolidAngle=True,
        **kwargs,
    ):
        """Perform the 2D integration and perform a sigm-clipping
        iterative filter along each row. see the doc of scipy.stats.sigmaclip for the options.

        Parameters
        ----------
        npt_rad: int
             The number of radial points.
        npt_azim: int
             The number of radial points
        mask:  boolean array or BaseSignal
            A boolean mask to apply to the data to exclude some points.
            If mask is a BaseSignal then it is iterated over as well.
        inplace: bool
            If the signal is overwritten or copied to a new signal
        method: str
            Can be “numpy”, “cython”, “BBox” or “splitpixel”, “lut”, “csr”,
            “nosplit_csr”, “full_csr”, “lut_ocl” and “csr_ocl” if you want
            to go on GPU. To Specify the device: “csr_ocl_1,2”
        sum: bool
            If true the radial integration is returned rather then the Azimuthal Integration.
        correctSolidAngle: bool
            Account for Ewald sphere or not. From PYFAI.

        Other Parameters
        -------
        dummy: float
            Value for dead/masked pixels
        delta_dummy: float
            Percision value for dead/masked pixels
        correctSolidAngle: bool
            Correct for the solid angle of each pixel if True
        dark: ndarray
            The dark noise image
        flat: ndarray
            The flat field correction image
        safe: bool
            Do some extra checks to ensure LUT/CSR is still valid. False is faster.
        show_progressbar: bool
            If True shows a progress bar for the mapping function


        Returns
        -------
        polar: PolarDiffraction2D
            A polar diffraction signal

        Examples
        --------
        Basic case using "2th_deg" units (no wavelength needed)

        >>> ds.unit = "2th_deg"
        >>> ds.set_ai()
        >>> ds.get_radial_integral(npt=100, npt_rad=400)

        Basic case using a curved Ewald Sphere approximation and pyXEM units
        (wavelength needed)

        >>> ds.unit = "k_nm^-1" # setting units
        >>> ds.set_ai(wavelength=2.5e-12)
        >>> ds.get_radial_integral(npt=100,npt_rad=400)

        """
        sig_shape = self.axes_manager.signal_shape
        radial_range = _get_radial_extent(ai=self.ai, shape=sig_shape, unit=self.unit)
        radial_range[0] = 0
        integration = self.map(
            sigma_clip,
            azimuthal_integrator=self.ai,
            npt_rad=npt_rad,
            npt_azim=npt_azim,
            method=method,
            max_iter=max_iter,
            thres=thres,
            inplace=inplace,
            unit=self.unit,
            mask=mask,
            correctSolidAngle=correctSolidAngle,
            **kwargs,
        )

        s = self if inplace else integration

        # Dealing with axis changes
        k_axis = s.axes_manager.signal_axes[0]
        k_axis.name = "Radius"
        k_axis.scale = (radial_range[1] - radial_range[0]) / npt_rad
        # k_axis.units = unit.unit_symbol
        k_axis.offset = radial_range[0]

        return integration


class LazyDiffraction2D(LazySignal, Diffraction2D):
    pass
