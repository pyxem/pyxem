# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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

"""Signal class for two-dimensional diffraction data in Cartesian coordinates."""

import numpy as np
from skimage import filters
from skimage.morphology import square
from scipy.ndimage import rotate
from skimage import morphology
import dask.array as da
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import warnings

import hyperspy.api as hs
from hyperspy.signals import Signal2D, BaseSignal
from hyperspy._signals.lazy import LazySignal
from hyperspy._signals.signal1d import LazySignal1D
from hyperspy._signals.signal2d import LazySignal2D
from hyperspy.misc.utils import isiterable

from pyFAI.units import to_unit

from pyxem.signals import (
    CommonDiffraction,
    DPCBaseSignal,
    DPCSignal1D,
    DPCSignal2D,
    LazyDPCBaseSignal,
    LazyDPCSignal1D,
    LazyDPCSignal2D,
)
from pyxem.utils.pyfai_utils import (
    get_azimuthal_integrator,
    _get_radial_extent,
    _get_setup,
)
from pyxem.utils.expt_utils import (
    azimuthal_integrate1d,
    azimuthal_integrate2d,
    gain_normalise,
    remove_dead,
    regional_filter,
    circular_mask,
    find_beam_offset_cross_correlation,
    convert_affine_to_transform,
    apply_transformation,
    find_beam_center_blur,
    find_beam_center_interpolate,
    integrate_radially,
    medfilt_1d,
    sigma_clip,
)
from pyxem.utils.dask_tools import (
    _process_dask_array,
    _get_dask_array,
    get_signal_dimension_host_chunk_slice,
    align_single_frame,
)
from pyxem.utils.signal import (
    select_method_from_method_dict,
    transfer_navigation_axes,
)
import pyxem.utils.pixelated_stem_tools as pst
import pyxem.utils.dask_tools as dt
import pyxem.utils.marker_tools as mt
import pyxem.utils.ransac_ellipse_tools as ret


class Diffraction2D(Signal2D, CommonDiffraction):
    _signal_type = "diffraction"

    """ Methods that make geometrical changes to a diffraction pattern """

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
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().

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

        return self.map(
            apply_transformation,
            transformation=transformation,
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
        parallel=True,
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
        parallel : bool
            If True, run the processing on several cores.
            In most cases this should be True, but for debugging False can be
            useful. Default True
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
        >>> s = pxm.dummy_data.get_disk_shift_simple_test_signal()
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
            parallel=parallel,
            show_progressbar=show_progressbar,
            interpolation_order=interpolation_order,
            shift_x=s_shift_x,
            shift_y=s_shift_y,
        )
        if not inplace:
            return s_shift

    def rotate_diffraction(self, angle, parallel=True, show_progressbar=True):
        """
        Rotate the diffraction dimensions.

        Parameters
        ----------
        angle : scalar
            Clockwise rotation in degrees.
        parallel : bool
            Default True
        show_progressbar : bool
            Default True

        Returns
        -------
        rotated_signal : Diffraction2D class

        Examples
        --------
        >>> s = pxm.dummy_data.get_holz_simple_test_signal()
        >>> s_rot = s.rotate_diffraction(30, show_progressbar=False)

        """
        s_rotated = self.map(
            rotate,
            ragged=False,
            angle=-angle,
            reshape=False,
            parallel=parallel,
            inplace=False,
            show_progressbar=show_progressbar,
        )
        if self._lazy:
            s_rotated.compute(progressbar=show_progressbar)
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
        >>> s = pxm.dummy_data.get_holz_simple_test_signal()
        >>> s_flip = s.flip_diffraction_x()

        To avoid changing the original object afterwards

        >>> s_flip = s.flip_diffraction_x().deepcopy()

        """
        s_out = self.copy()
        s_out.axes_manager = self.axes_manager.deepcopy()
        s_out.metadata = self.metadata.deepcopy()
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
        >>> s = pxm.dummy_data.get_holz_simple_test_signal()
        >>> s_flip = s.flip_diffraction_y()

        To avoid changing the original object afterwards

        >>> s_flip = s.flip_diffraction_y().deepcopy()

        """
        s_out = self.copy()
        s_out.axes_manager = self.axes_manager.deepcopy()
        s_out.metadata = self.metadata.deepcopy()
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
        signal-mask : ndarray
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
        dark_reference : ElectronDiffraction2D
            Dark reference image.
        bright_reference : DiffractionSignal
            Bright reference image.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().

        """
        return self.map(
            gain_normalise,
            dref=dark_reference,
            bref=bright_reference,
            inplace=inplace,
            *args,
            **kwargs,
        )

    def subtract_diffraction_background(
        self, method="median kernel", lazy_result=True, show_progressbar=True, **kwargs
    ):
        """Background subtraction of the diffraction data.

        Parameters
        ----------
        method : str, optional
            'difference of gaussians', 'median kernel', 'radial median', 'h-dome'
            'h-dome' is for non-lazy data only. Default 'median kernel'.
        lazy_result : bool, optional
            If True (default), will return a LazyDiffraction2D object. If False,
            will compute the result and return a Diffraction2D object.
        show_progressbar : bool, optional
            Default True
        **kwargs :
            To be passed to the method chosen: min_sigma/max_sigma, footprint,
            centre_x,centre_y / h

        Returns
        -------
        s : Diffraction2D or LazyDiffraction2D signal

        Examples
        --------
        >>> s = pxm.dummy_data.get_cbed_signal()
        >>> s_r = s.subtract_diffraction_background(method='median kernel',
        ...     footprint=20, lazy_result=False, show_progressbar=False)
        >>> s_r.plot()

        """

        # Ugly, should look into making this lazy compatible
        if method == "h-dome":
            self.data = self.data / np.max(self.data)
            bg_subtracted = self.map(regional_filter, inplace=False, **kwargs)
            bg_subtracted.map(filters.rank.mean, selem=square(3))
            bg_subtracted.data = bg_subtracted.data / np.max(bg_subtracted.data)
            return bg_subtracted

        dask_array = _get_dask_array(self)

        if method == "difference of gaussians":
            output_array = dt._background_removal_dog(dask_array, **kwargs)
        elif method == "median kernel":
            output_array = dt._background_removal_median(dask_array, **kwargs)
        elif method == "radial median":
            output_array = dt._background_removal_radial_median(dask_array, **kwargs)
        else:
            raise NotImplementedError(
                "The method specified, '{}', is not implemented. "
                "The different methods are: 'difference of gaussians',"
                " 'median kernel','radial median' or 'h-dome'.".format(method)
            )

        if not lazy_result:
            if show_progressbar:
                pbar = ProgressBar()
                pbar.register()
            output_array = output_array.compute()
            if show_progressbar:
                pbar.unregister()
            s = Diffraction2D(output_array)
        else:
            s = LazyDiffraction2D(output_array)
        pst._copy_signal_all_axes_metadata(self, s)
        return s

    def find_dead_pixels(
        self,
        dead_pixel_value=0,
        mask_array=None,
        lazy_result=False,
        show_progressbar=True,
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
        >>> s = pxm.dummy_data.get_dead_pixel_signal()
        >>> s_dead_pixels = s.find_dead_pixels(show_progressbar=False)

        Using a mask array

        >>> import numpy as np
        >>> mask_array = np.zeros((128, 128), dtype=bool)
        >>> mask_array[:, 100:] = True
        >>> s = pxm.dummy_data.get_dead_pixel_signal()
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
        dask_array = _get_dask_array(self)

        dead_pixels = dt._find_dead_pixels(
            dask_array, dead_pixel_value=dead_pixel_value, mask_array=mask_array
        )
        s_dead_pixels = LazySignal2D(dead_pixels)
        if not lazy_result:
            s_dead_pixels.compute(progressbar=show_progressbar)
        return s_dead_pixels

    def find_hot_pixels(
        self,
        threshold_multiplier=500,
        mask_array=None,
        lazy_result=True,
        show_progressbar=True,
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
        >>> s = pxm.dummy_data.get_hot_pixel_signal()
        >>> s_hot_pixels = s.find_hot_pixels(show_progressbar=False)

        Using a mask array

        >>> import numpy as np
        >>> mask_array = np.zeros((128, 128), dtype=bool)
        >>> mask_array[:, 100:] = True
        >>> s = pxm.dummy_data.get_hot_pixel_signal()
        >>> s_hot_pixels = s.find_hot_pixels(
        ...     mask_array=mask_array, show_progressbar=False)

        Getting a non-lazy signal as output

        >>> s_hot_pixels = s.find_hot_pixels(
        ...     lazy_result=False, show_progressbar=False)

        See Also
        --------
        find_dead_pixels
        correct_bad_pixels

        """
        dask_array = _get_dask_array(self)

        hot_pixels = dt._find_hot_pixels(
            dask_array, threshold_multiplier=threshold_multiplier, mask_array=mask_array
        )

        s_hot_pixels = LazySignal2D(hot_pixels)
        if not lazy_result:
            s_hot_pixels.compute(progressbar=show_progressbar)
        return s_hot_pixels

    def correct_bad_pixels(
        self,
        bad_pixel_array,
        show_progressbar=True,
        lazy_result=True,
        inplace=True,
        *args,
        **kwargs,
    ):
        """Correct bad (dead/hot) pixels by replacing their values with the mean value of neighbors.

        Parameters
        ----------
        bad_pixel_array : array-like
            List of pixels to correct
        show_progressbar : bool, optional
            Default True
        lazy_result : bool, optional
            When working lazily, determines if the result is computed. Default is True (ie. no .compute)
        inplace : bool, optional
            When working in memory, determines if operation is performed inplace, default is True. When
            working lazily the result will NOT be inplace.
        *args :
            passed to .map() if working in memory
        **kwargs :
            passed to .map() if working in memory

        Returns
        -------
        signal_corrected : Diffraction2D or LazyDiffraction2D

        Examples
        --------
        >>> s = pxm.dummy_data.get_hot_pixel_signal()
        >>> s_hot_pixels = s.find_hot_pixels(
        ...     show_progressbar=False, lazy_result=True)
        >>> s_corr = s.correct_bad_pixels(s_hot_pixels)

        See Also
        --------
        find_dead_pixels
        find_hot_pixels

        """
        dask_array = self.data
        bad_pixel_removed = dt._remove_bad_pixels(dask_array, bad_pixel_array.data)
        s_bad_pixel_removed = LazyDiffraction2D(bad_pixel_removed)
        pst._copy_signal2d_axes_manager_metadata(self, s_bad_pixel_removed)
        if not lazy_result:
            s_bad_pixel_removed.compute(progressbar=show_progressbar)
        return s_bad_pixel_removed

    """ Direct beam and peak finding tools """

    def get_direct_beam_position(self, method, lazy_result=None, **kwargs):
        """Estimate the direct beam position in each experimentally acquired
        electron diffraction pattern.

        Parameters
        ----------
        method : str,
            Must be one of "cross_correlate", "blur" or "interpolate"
        lazy_result : optional
            If True, s_shifts will be a lazy signal. If False, a non-lazy signal.
            By default, if the signal is (non-)lazy, the result will also be (non-)lazy.
        **kwargs:
            Keyword arguments to be passed to the method function.

        Returns
        -------
        s_shifts : HyperSpy Signal1D
            Array containing the shifts for each SED pattern, with the first
            signal index being the x-shift and the second the y-shift.

        """
        if lazy_result is None:
            lazy_result = self._lazy

        signal_shape = self.axes_manager.signal_shape
        origin_coordinates = np.array(signal_shape) / 2

        method_dict = {
            "cross_correlate": find_beam_offset_cross_correlation,
            "blur": find_beam_center_blur,
            "interpolate": find_beam_center_interpolate,
        }

        method_function = select_method_from_method_dict(method, method_dict, **kwargs)

        dask_array = _get_dask_array(self)

        drop_axis = (len(self.axes_manager.shape) - 2, len(self.axes_manager.shape) - 1)
        new_axis = self.axes_manager.navigation_dimension
        chunks_output = dask_array.chunks[:-2] + ((2,),)

        if method == "cross_correlate":
            shifts = _process_dask_array(
                dask_array,
                method_function,
                dtype=np.float32,
                output_signal_size=(2,),
                drop_axis=drop_axis,
                new_axis=new_axis,
                chunks=chunks_output,
                **kwargs,
            )
        elif method == "blur":
            centers = _process_dask_array(
                dask_array,
                method_function,
                dtype=np.int16,
                output_signal_size=(2,),
                drop_axis=drop_axis,
                new_axis=new_axis,
                chunks=chunks_output,
                **kwargs,
            )
            shifts = origin_coordinates - centers
        elif method == "interpolate":
            centers = _process_dask_array(
                dask_array,
                method_function,
                dtype=np.float32,
                output_signal_size=(2,),
                drop_axis=drop_axis,
                new_axis=new_axis,
                chunks=chunks_output,
                **kwargs,
            )
            shifts = origin_coordinates - centers

        s_shifts = LazySignal1D(shifts)

        if not lazy_result:
            s_shifts.compute()

        return s_shifts

    def center_direct_beam(
        self,
        method=None,
        half_square_width=None,
        shifts=None,
        return_shifts=False,
        subpixel=True,
        lazy_result=None,
        align_kwargs=None,
        *args,
        **kwargs,
    ):
        """Estimate the direct beam position in each experimentally acquired
        electron diffraction pattern and translate it to the center of the
        image square.

        Parameters
        ----------
        method : str {'cross_correlate', 'blur', 'interpolate'}
            Method used to estimate the direct beam position. The direct
            beam position can also be passed directly with the shifts parameter.
        half_square_width : int
            Half the side length of square that captures the direct beam in all
            scans. Means that the centering algorithm is stable against
            diffracted spots brighter than the direct beam.
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
        lazy_result : optional
            If True, the result will be a lazy signal. If False, a non-lazy signal.
            By default, if the signal is lazy, the result will also be lazy.
            If the signal is non-lazy, the result will be non-lazy.
        align_kwargs : dict
            Parameters passed to the alignment function. See scipy.ndimage.shift
            for more information about the parameters.
        *args, **kwargs :
            Passed to the function which estimate the direct beam position

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
        This is due to subpixel=True utilizing interpolation. To keep the total intensity
        use a float dtype, which can be done by s.change_dtype('float32', rechunk=False).

        """
        if (shifts is None) and (method is None):
            raise ValueError("Either method or shifts parameter must be specified")
        if (shifts is not None) and (method is not None):
            raise ValueError(
                "Only one of the shifts or method parameters should be specified, "
                "not both"
            )
        if lazy_result is None:
            lazy_result = self._lazy
        if align_kwargs is None:
            align_kwargs = {}

        signal_shape = self.axes_manager.signal_shape
        origin_coordinates = np.array(signal_shape) / 2

        data_dask_array = _get_dask_array(self)
        if not self._lazy:
            temp_signal = self.as_lazy()
            temp_signal.data = da.from_array(self.data, chunks=data_dask_array.chunks)
        else:
            temp_signal = self

        if shifts is None:
            if half_square_width is not None:
                min_index = np.int(origin_coordinates[0] - half_square_width)
                # fails if non-square dp
                max_index = np.int(origin_coordinates[0] + half_square_width)
                temp_signal = temp_signal.isig[min_index:max_index, min_index:max_index]
            shifts = temp_signal.get_direct_beam_position(
                method=method,
                lazy_result=True,
                **kwargs,
            )

        if not "order" in align_kwargs:
            if subpixel:
                align_kwargs["order"] = 1
            else:
                align_kwargs["order"] = 0

        shifts_dask_array = _get_dask_array(shifts)

        output_dask_array = _process_dask_array(
            data_dask_array,
            align_single_frame,
            iter_array=shifts_dask_array,
            **align_kwargs,
        )

        if lazy_result:
            if not self._lazy:
                self._lazy = True
                self._assign_subclass()
            self.data = output_dask_array
        else:
            if self._lazy:
                self.data = np.empty(
                    output_dask_array.shape, dtype=output_dask_array.dtype
                )
                self._lazy = False
                self._assign_subclass()
            shifts.data = np.empty(
                shifts_dask_array.shape, dtype=shifts_dask_array.dtype
            )
            shifts._lazy = False
            shifts._assign_subclass()
            with ProgressBar():
                da.store(
                    [shifts_dask_array, output_dask_array],
                    [shifts.data, self.data],
                )

        self.events.data_changed.trigger(obj=self)

        if return_shifts:
            return shifts

    def threshold_and_mask(self, threshold=None, mask=None, show_progressbar=True):
        """Get a thresholded and masked of the signal.

        Useful for figuring out optimal settings for the center_of_mass
        method.

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
        s_out : Diffraction2D signal

        Examples
        --------
        >>> s = pxm.dummy_data.get_disk_shift_simple_test_signal()
        >>> mask = (25, 25, 10)
        >>> s_out = s.threshold_and_mask(
        ...     mask=mask, threshold=2, show_progressbar=False)
        >>> s_out.plot()

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
            parallel=True,
            show_progressbar=show_progressbar,
            threshold=threshold,
            mask=mask,
        )
        return s_out

    def center_of_mass(
        self,
        threshold=None,
        mask=None,
        lazy_result=False,
        show_progressbar=True,
        chunk_calculations=None,
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
            Round mask centered on x and y, with radius r.
        lazy_result : bool, optional
            If True, will not compute the data directly, but
            return a lazy signal. Default False
        show_progressbar : bool, optional
            Default True
        chunk_calculations : tuple, optional
            Chunking values when running the calculations.

        Returns
        -------
        s_com : DPCSignal
            DPCSignal with beam shifts along the navigation dimension
            and spatial dimensions as the signal dimension(s).

        Examples
        --------
        With mask centered at x=105, y=120 and 30 pixel radius

        >>> s = pxm.dummy_data.get_disk_shift_simple_test_signal()
        >>> mask = (25, 25, 10)
        >>> s_com = s.center_of_mass(mask=mask, show_progressbar=False)
        >>> s_color = s_com.get_color_signal()

        Also threshold

        >>> s_com = s.center_of_mass(threshold=1.5, show_progressbar=False)

        Get a lazy signal, then calculate afterwards

        >>> s_com = s.center_of_mass(lazy_result=True, show_progressbar=False)
        >>> s_com.compute(progressbar=False)

        """
        det_shape = self.axes_manager.signal_shape
        nav_dim = self.axes_manager.navigation_dimension
        if chunk_calculations is None:
            chunk_calculations = [16] * nav_dim + list(det_shape)
        if mask is not None:
            x, y, r = mask
            mask_array = pst._make_circular_mask(x, y, det_shape[0], det_shape[1], r)
            mask_array = np.invert(mask_array)
        else:
            mask_array = None
        if self._lazy:
            dask_array = self.data.rechunk(chunk_calculations)
        else:
            dask_array = da.from_array(self.data, chunks=chunk_calculations)
        data = dt._center_of_mass_array(
            dask_array, threshold_value=threshold, mask_array=mask_array
        )
        if lazy_result:
            if nav_dim == 2:
                s_com = LazyDPCSignal2D(data)
            elif nav_dim == 1:
                s_com = LazyDPCSignal1D(data)
            elif nav_dim == 0:
                s_com = LazyDPCBaseSignal(data).T
        else:
            if show_progressbar:
                pbar = ProgressBar()
                pbar.register()
            data = data.compute()
            if show_progressbar:
                pbar.unregister()
            if nav_dim == 2:
                s_com = DPCSignal2D(data)
            elif nav_dim == 1:
                s_com = DPCSignal1D(data)
            elif nav_dim == 0:
                s_com = DPCBaseSignal(data).T
        s_com.axes_manager.navigation_axes[0].name = "Beam position"
        for nav_axes, sig_axes in zip(
            self.axes_manager.navigation_axes, s_com.axes_manager.signal_axes
        ):
            pst._copy_axes_object_metadata(nav_axes, sig_axes)
        return s_com

    def template_match_disk(self, disk_r=4, lazy_result=True, show_progressbar=True):
        """Template match the signal dimensions with a disk.

        Used to find diffraction disks in convergent beam electron
        diffraction data.

        Parameters
        ----------
        disk_r : scalar, optional
            Radius of the disk. Default 4.
        lazy_result : bool, default True
            If True, will return a LazyDiffraction2D object. If False,
            will compute the result and return a Diffraction2D object.
        show_progressbar : bool, default True

        Returns
        -------
        template_match : Diffraction2D object

        Examples
        --------
        >>> s = pxm.dummy_data.get_cbed_signal()
        >>> s_template = s.template_match_disk(
        ...     disk_r=5, show_progressbar=False)
        >>> s.plot()

        See Also
        --------
        template_match_ring
        template_match_with_binary_image

        """
        disk = morphology.disk(disk_r, self.data.dtype)
        s = self.template_match_with_binary_image(
            disk, lazy_result=lazy_result, show_progressbar=show_progressbar
        )
        return s

    def template_match_ring(
        self, r_inner=5, r_outer=7, lazy_result=True, show_progressbar=True
    ):
        """Template match the signal dimensions with a ring.

        Used to find diffraction rings in convergent beam electron
        diffraction data.

        Parameters
        ----------
        r_inner, r_outer : scalar, optional
            Inner and outer radius of the rings.
        lazy_result : bool, default True
            If True, will return a LazyDiffraction2D object. If False,
            will compute the result and return a Diffraction2D object.
        show_progressbar : bool, default True

        Returns
        -------
        template_match : Diffraction2D object

        Examples
        --------
        >>> s = pxm.dummy_data.get_cbed_signal()
        >>> s_template = s.template_match_ring(show_progressbar=False)
        >>> s.plot()

        See Also
        --------
        template_match_disk
        template_match_with_binary_image

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
        s = self.template_match_with_binary_image(
            ring, lazy_result=lazy_result, show_progressbar=show_progressbar
        )
        return s

    def template_match_with_binary_image(
        self, binary_image, lazy_result=True, show_progressbar=True
    ):
        """Template match the signal dimensions with a binary image.

        Used to find diffraction disks in convergent beam electron
        diffraction data.

        Might also work with non-binary images, but this haven't been
        extensively tested.

        Parameters
        ----------
        binary_image : 2-D NumPy array
        lazy_result : bool, default True
            If True, will return a LazyDiffraction2D object. If False,
            will compute the result and return a Diffraction2D object.
        show_progressbar : bool, default True

        Returns
        -------
        template_match : Diffraction2D object

        Examples
        --------
        >>> s = pxm.dummy_data.get_cbed_signal()
        >>> binary_image = np.random.randint(0, 2, (6, 6))
        >>> s_template = s.template_match_with_binary_image(
        ...     binary_image, show_progressbar=False)
        >>> s.plot()

        See Also
        --------
        template_match_disk
        template_match_ring

        """
        dask_array = _get_dask_array(self)

        output_array = dt._template_match_with_binary_image(dask_array, binary_image)
        if not lazy_result:
            if show_progressbar:
                pbar = ProgressBar()
                pbar.register()
            output_array = output_array.compute()
            if show_progressbar:
                pbar.unregister()
            s = Diffraction2D(output_array)
        else:
            s = LazyDiffraction2D(output_array)
        pst._copy_signal_all_axes_metadata(self, s)
        return s

    def find_peaks_lazy(
        self, method="dog", lazy_result=True, show_progressbar=True, **kwargs
    ):
        """Find peaks in the signal dimensions.

        Can use either skimage's blob_dog or blob_log.

        Parameters
        ----------
        method : string, optional
            'dog'(default) for difference of Gaussians. 'log' for Laplacian of Gaussian.
        lazy_result : bool, optional
            Default True
        show_progressbar : bool, optional
            Default True
        **kwargs :
            Passed to the peakfinder, see skimage docs for details

        Returns
        -------
        peak_array : dask 2D object array
            Same size as the two last dimensions in data, in the form
            [[y0, x0], [y1, x1], ...].
            The peak positions themselves are stored in 2D NumPy arrays
            inside each position in peak_array. This is done instead of
            making a 4D NumPy array, since the number of found peaks can
            vary in each position.

        Example
        -------
        >>> s = pxm.dummy_data.get_cbed_signal()
        >>> peak_array = s.find_peaks_lazy()
        >>> peak_array_computed = peak_array.compute(show_progressbar=False)
        >>> peak02 = peak_array_computed[0, 2]
        >>> s.add_peak_array_as_markers(peak_array_computed)
        >>> s.plot()

        Change parameters

        >>> peak_array = s.find_peaks_lazy(
        ...     method='dog', min_sigma=1.2, max_sigma=27, sigma_ratio=2.2,
        ...     threshold=0.6, overlap=0.6, lazy_result=False,
        ...     show_progressbar=False)

        Using Laplacian of Gaussian

        >>> s = pxm.dummy_data.get_cbed_signal()
        >>> peak_array = s.find_peaks_lazy(
        ...     method='log', min_sigma=5, max_sigma=55, num_sigma=10,
        ...     threshold=0.2, overlap=0.86, lazy_result=False,
        ...     show_progressbar=False)
        >>> s.add_peak_array_as_markers(peak_array)
        >>> s.plot()

        """
        if not self._lazy:
            raise ValueError("Signal is not lazy, please use the non-lazy version")

        dask_array = self.data

        if method == "dog":
            output_array = dt._peak_find_dog(dask_array, **kwargs)
        elif method == "log":
            output_array = dt._peak_find_log(dask_array, **kwargs)
        else:
            raise ValueError("Method is not a valid name, should be dog or log")

        if not lazy_result:
            if show_progressbar:
                pbar = ProgressBar()
                pbar.register()
            output_array = output_array.compute()
            if show_progressbar:
                pbar.unregister()
        return output_array

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
        >>> s = pxm.dummy_data.get_cbed_signal()
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
            If True, will return a LazyDiffraction2D object. If False,
            will compute the result and return a Diffraction2D object.
        show_progressbar : bool, default True

        Returns
        -------
        intensity_array: Numpy or Dask array
            Same navigation shape as this signal, with peak position in
            x and y coordinates and the mean intensity.

        Examples
        --------
        >>> s = pxm.dummy_data.get_cbed_signal()
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
                isig_slice = get_signal_dimension_host_chunk_slice(
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

    def add_peak_array_as_markers(
        self, peak_array, color="red", size=20, bool_array=None, bool_invert=False
    ):
        """Add a peak array to the signal as HyperSpy markers.

        Parameters
        ----------
        peak_array : NumPy 4D array
        color : string, optional
            Default 'red'
        size : scalar, optional
            Default 20
        bool_array : NumPy array
            Must be the same size as peak_array
        bool_invert : bool

        Examples
        --------
        >>> s, parray = pxm.dummy_data.get_simple_ellipse_signal_peak_array()
        >>> s.add_peak_array_as_markers(parray)
        >>> s.plot()

        """
        mt.add_peak_array_to_signal_as_markers(
            self,
            peak_array,
            color=color,
            size=size,
            bool_array=bool_array,
            bool_invert=bool_invert,
        )

    def add_ellipse_array_as_markers(
        self,
        ellipse_array,
        inlier_array=None,
        peak_array=None,
        nr=20,
        color_ellipse="blue",
        linewidth=1,
        linestyle="solid",
        color_inlier="blue",
        color_outlier="red",
        point_size=20,
    ):
        """Add a ellipse parameters array to a signal as HyperSpy markers.

        Useful to visualize the ellipse results.

        Parameters
        ----------
        ellipse_array : NumPy array
        inlier_array : NumPy array, optional
        peak_array : NumPy array, optional
        nr : scalar, optional
            Default 20
        color_ellipse : string, optional
            Default 'blue'
        linewidth : scalar, optional
            Default 1
        linestyle : string, optional
            Default 'solid'
        color_inlier : string, optional
            Default 'blue'
        color_outlier : string, optional
            Default 'red'
        point_size : scalar, optional

        Examples
        --------
        >>> s, parray = pxm.dummy_data.get_simple_ellipse_signal_peak_array()
        >>> import pyxem.utils.ransac_ellipse_tools as ret
        >>> ellipse_array, inlier_array = ret.get_ellipse_model_ransac(
        ...     parray, xf=95, yf=95, rf_lim=20, semi_len_min=40,
        ...     semi_len_max=100, semi_len_ratio_lim=5, max_trails=50)
        >>> s.add_ellipse_array_as_markers(
        ...     ellipse_array, inlier_array=inlier_array, peak_array=parray)
        >>> s.plot()

        """
        if len(self.data.shape) != 4:
            raise ValueError("Signal must be 4 dims to use this function")
        marker_list = ret._get_ellipse_markers(
            ellipse_array,
            inlier_array,
            peak_array,
            nr=20,
            color_ellipse="blue",
            linewidth=1,
            linestyle="solid",
            color_inlier="blue",
            color_outlier="red",
            point_size=20,
            signal_axes=self.axes_manager.signal_axes,
        )

        mt._add_permanent_markers_to_signal(self, marker_list)

    def lazy_virtual_bright_field(
        self, cx=None, cy=None, r=None, lazy_result=False, show_progressbar=True
    ):
        """Get a virtual bright field signal.

        Can be sum the whole diffraction plane, or a circle subset.
        If any of the parameters are None, it will sum the whole diffraction
        plane.

        Parameters
        ----------
        cx, cy : floats, optional
            x- and y-centre positions.
        r : float, optional
            Outer radius.
        lazy_result : bool, optional
            If True, will not compute the data directly, but
            return a lazy signal. Default False
        show_progressbar : bool, optional
            Default True.

        Returns
        -------
        virtual_bf_signal : HyperSpy 2D signal

        See Also
        --------
        get_integrated_intensity
        """
        warnings.warn("This method is depreacted and will be removed in the next version",FutureWarning)
        det_shape = self.axes_manager.signal_shape
        if (cx is None) or (cy is None) or (r is None):
            mask_array = np.zeros(det_shape[::-1], dtype=bool)
        else:
            mask_array = pst._make_circular_mask(cx, cy, det_shape[0], det_shape[1], r)
            mask_array = np.invert(mask_array)
        data = dt._mask_array(self.data, mask_array=mask_array).sum(axis=(-2, -1))
        s_bf = LazySignal2D(data)
        if not lazy_result:
            s_bf.compute(progressbar=show_progressbar)
        for nav_axes, sig_axes in zip(
            self.axes_manager.navigation_axes, s_bf.axes_manager.signal_axes
        ):
            pst._copy_axes_object_metadata(nav_axes, sig_axes)

        return s_bf

    def lazy_virtual_annular_dark_field(
        self, cx, cy, r_inner, r, lazy_result=False, show_progressbar=True
    ):
        """Get a virtual annular dark field signal.

        Parameters
        ----------
        cx, cy : floats
            x- and y-centre positions.
        r_inner : float
            Inner radius.
        r : float
            Outer radius.
        lazy_result : bool, optional
            If True, will not compute the data directly, but
            return a lazy signal. Default False
        show_progressbar : bool, default True

        Returns
        -------
        virtual_adf_signal : HyperSpy 2D signal

        See Also
        --------
        get_integrated_intensity
        """
        warnings.warn("This method is depreacted and will be removed in the next version",FutureWarning)

        if r_inner > r:
            raise ValueError(
                "r_inner must be higher than r. The argument order is "
                + "(cx, cy, r_inner, r)"
            )
        det_shape = self.axes_manager.signal_shape

        mask_array0 = pst._make_circular_mask(cx, cy, det_shape[0], det_shape[1], r)
        mask_array1 = pst._make_circular_mask(
            cx, cy, det_shape[0], det_shape[1], r_inner
        )
        mask_array = mask_array0 == mask_array1

        data = dt._mask_array(self.data, mask_array=mask_array).sum(axis=(-2, -1))
        s_adf = LazySignal2D(data)
        if not lazy_result:
            s_adf.compute(progressbar=show_progressbar)
        for nav_axes, sig_axes in zip(
            self.axes_manager.navigation_axes, s_adf.axes_manager.signal_axes
        ):
            pst._copy_axes_object_metadata(nav_axes, sig_axes)
        return s_adf

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
        mask_array : NumPy array
            The True values will be the region between angle0 and angle1.
            The array will have the same dimensions as the signal.

        Examples
        --------
        >>> s = pxm.dummy_data.get_holz_simple_test_signal()
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
        """Calculates the variance using one of the methods described in [1]. A shot noise correction and
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
            Included intermediate spatial variance in output (only avaliable if method=='r')
        navigation_axes : list or none, optional
            The axes to calculate the variance over.  The default is to use the navigation axes.
        **kwargs: dict
            Any keywords accepted for the get_azimuthal_integral1d() or get_azimuthal_integral2d() function

        Returns
        -------
        variance : array-like
            Calculate variance as it's own signal

        References
        ----------
        [1] Daulton, T. L et al, Ultramicroscopy, 110(10), 1279–1289, https://doi.org/10.1016/j.ultramic.2010.05.010
            Nanobeam diffraction fluctuation electron microscopy technique for structural characterization of disordered
            materials-Application to Al88-xY7Fe5Tix metallic glasses.
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
            one_d_integration = self.get_azimuthal_integral1d(npt=npt, **kwargs)
            variance = (
                (one_d_integration ** 2).mean(axis=navigation_axes)
                / one_d_integration.mean(axis=navigation_axes) ** 2
            ) - 1
            if dqe is not None:
                sum_points = self.get_azimuthal_integral1d(
                    npt=npt, sum=True, **kwargs
                ).mean(axis=navigation_axes)
                variance = variance - ((sum_points ** -1) * dqe)

        elif method == "r":
            one_d_integration = self.get_azimuthal_integral1d(npt=npt, **kwargs)
            integration_squared = (self ** 2).get_azimuthal_integral1d(
                npt=npt, **kwargs
            )
            # Full variance is the same as the unshifted phi=0 term in angular correlation
            full_variance = (integration_squared / one_d_integration ** 2) - 1

            if dqe is not None:
                full_variance = full_variance - ((one_d_integration ** -1) * dqe)

            variance = full_variance.mean(axis=navigation_axes)

            if spatial:
                return variance, full_variance

        elif method == "re":
            one_d_integration = self.get_azimuthal_integral1d(npt=npt, **kwargs).mean(
                axis=navigation_axes
            )
            integration_squared = (
                (self ** 2)
                .get_azimuthal_integral1d(npt=npt, **kwargs)
                .mean(axis=navigation_axes)
            )
            variance = (integration_squared / one_d_integration ** 2) - 1

            if dqe is not None:
                sum_int = self.get_azimuthal_integral1d(npt=npt, **kwargs).mean()
                variance = variance - (sum_int ** -1) * (1 / dqe)

        elif method == "VImage":
            variance_image = (
                (self ** 2).mean(axis=navigation_axes)
                / self.mean(axis=navigation_axes) ** 2
            ) - 1
            if dqe is not None:
                variance_image = variance_image - (
                    self.sum(axis=navigation_axes) ** -1
                ) * (1 / dqe)
            variance = variance_image.get_azimuthal_integral1d(npt=npt, **kwargs)
        return variance

    """ Methods associated with radial integration, not pyFAI based """

    def radial_integration(self):
        raise Exception("radial_integration has been renamed radial_average")

    def radial_average(
        self,
        centre_x=None,
        centre_y=None,
        mask_array=None,
        normalize=True,
        parallel=True,
        show_progressbar=True,
    ):
        """Radially average a pixelated STEM diffraction signal.

        Done by integrating over the azimuthal dimension, giving a
        profile of intensity as a function of scattering angle.

        Parameters
        ----------
        centre_x, centre_y : int or NumPy array, optional
            If given as int, all the diffraction patterns will have the same
            centre position. Each diffraction pattern can also have different
            centre position, by passing a NumPy array with the same dimensions
            as the navigation axes.
            Note: in either case both x and y values must be given. If one is
            missing, both will be set from the signal (0., 0.) positions.
            If no values are given, the (0., 0.) positions in the signal will
            be used.
        mask_array : Boolean NumPy array, optional
            Mask with the same shape as the signal.
        normalize : bool, default True
            If true, the returned radial profile will be normalized by the
            number of bins used for each average.
        parallel : bool, default True
            If True, run the processing on several cores.
            In most cases this should be True, but for debugging False can be
            useful.
        show_progressbar : bool
            Default True

        Returns
        -------
        HyperSpy signal, one less signal dimension than the input signal.

        See Also
        --------
        .get_azimuthal_integral1d

        """
        warnings.warn(
            "This method is depreacted, and will be removed in version 0.14.0, please use .get_azimuthal_integral1d",
            FutureWarning,
        )
        if (centre_x is None) or (centre_y is None):
            centre_x, centre_y = pst._make_centre_array_from_signal(self)
        elif (not isiterable(centre_x)) or (not isiterable(centre_y)):
            centre_x, centre_y = pst._make_centre_array_from_signal(
                self, x=centre_x, y=centre_y
            )
        radial_array_size = (
            pst._find_longest_distance(
                self.axes_manager.signal_axes[0].size,
                self.axes_manager.signal_axes[1].size,
                centre_x.min(),
                centre_y.min(),
                centre_x.max(),
                centre_y.max(),
            )
            + 1
        )
        centre_x = centre_x.flatten()
        centre_y = centre_y.flatten()
        iterating_kwargs = [("centre_x", centre_x), ("centre_y", centre_y)]
        if mask_array is not None:
            #  This line flattens the mask array, except for the two
            #  last dimensions. This to make the mask array work for the
            #  _map_iterate function.
            mask_flat = mask_array.reshape(-1, *mask_array.shape[-2:])
            iterating_kwargs.append(("mask", mask_flat))

        if self._lazy:
            data = pst._radial_average_dask_array(
                self.data,
                return_sig_size=radial_array_size,
                centre_x=centre_x,
                centre_y=centre_y,
                mask_array=mask_array,
                normalize=normalize,
                show_progressbar=show_progressbar,
            )
            s_radial = hs.signals.Signal1D(data)
        else:
            s_radial = self._map_iterate(
                pst._get_radial_profile_of_diff_image,
                normalize=normalize,
                iterating_kwargs=iterating_kwargs,
                inplace=False,
                ragged=False,
                parallel=parallel,
                radial_array_size=radial_array_size,
                show_progressbar=show_progressbar,
            )
            data = s_radial.data
        s_radial = hs.signals.Signal1D(data)
        return s_radial

    def angular_slice_radial_integration(self):
        raise Exception(
            "angular_slice_radial_integration has been renamed "
            "angular_slice_radial_average"
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
        >>> s = pxm.dummy_data.get_holz_simple_test_signal()
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
        except (AttributeError):
            raise ValueError("ai property is not currently set")

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

    def get_azimuthal_integral1d(
        self,
        npt,
        mask=None,
        radial_range=None,
        azimuth_range=None,
        wavelength=None,
        inplace=False,
        method="splitpixel",
        sum=False,
        lazy_result=None,
        show_progressbar=None,
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
        wavelength : None or float
            The wavelength of for the microscope. Has to be in the same units as the pyxem units if you want
            it to properly work.
        inplace : bool
            If the signal is overwritten or copied to a new signal
        method : str
             Can be “numpy”, “cython”, “BBox” or “splitpixel”, “lut”, “csr”,
             “nosplit_csr”, “full_csr”, “lut_ocl” and “csr_ocl” if you want
             to go on GPU. To Specify the device: “csr_ocl_1,2”
        sum : bool
            If true returns the pixel split sum rather than the azimuthal integration which
            gives the mean.
        lazy_result : optional
            If True, the result will be a lazy signal. If False, a non-lazy signal.
            By default, if the signal is lazy, the result will also be lazy.
            If the signal is non-lazy, the result will be non-lazy.
        show_progressbar : None or bool
            If True and lazy_result is True, show a progressbar for the calculation.
            If None, the preference from the settings will be used.

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
        >>> ds.get_azimuthal_integral1d(npt=100, wavelength=2.5e-12)

        Using pyFAI to define a detector case using a curved Ewald Sphere approximation and pyXEM units

        >>> from pyFAI.detectors import Detector
        >>> det = Detector(pixel1=1e-4, pixel2=1e-4)
        >>> ds.get_azimuthal_integral1d(npt=100, detector_dist=.2, detector= det, wavelength=2.508e-12)
        """
        if lazy_result is None:
            lazy_result = self._lazy
        signal_type = self._signal_type
        unit = to_unit(self.unit)
        sig_shape = self.axes_manager.signal_shape
        if radial_range is None:
            radial_range = _get_radial_extent(ai=self.ai, shape=sig_shape, unit=unit)
            radial_range[0] = 0

        data_dask_array = _get_dask_array(self)
        chunks = data_dask_array.chunks[:-2] + ((npt,),)
        drop_axis = (len(self.axes_manager.shape) - 2, len(self.axes_manager.shape) - 1)
        new_axis = self.axes_manager.navigation_dimension
        integration_dask_array = _process_dask_array(
            data_dask_array,
            azimuthal_integrate1d,
            drop_axis=drop_axis,
            new_axis=new_axis,
            chunks=chunks,
            output_signal_size=(npt,),
            azimuthal_integrator=self.ai,
            npt_rad=npt,
            azimuth_range=azimuth_range,
            radial_range=radial_range,
            method=method,
            unit=unit,
            sum=sum,
            dtype=np.float32,  # pyFAI does the calculation in float32
            **kwargs,
        )

        # Dealing with axis changes
        if inplace:
            result = self
            result.axes_manager.remove(self.axes_manager.signal_axes[0])
            result.data = integration_dask_array
            result._lazy = True
        else:
            result = LazySignal1D(integration_dask_array)
            result.set_signal_type(signal_type)
            transfer_navigation_axes(result, self)
        k_axis = result.axes_manager.signal_axes[0]
        k_axis.name = "Radius"
        k_axis.scale = (radial_range[1] - radial_range[0]) / npt
        k_axis.units = unit.unit_symbol
        k_axis.offset = radial_range[0]

        result.set_signal_type(signal_type)
        if not lazy_result:
            result.compute(show_progressbar=show_progressbar)

        if not inplace:
            return result

    def get_azimuthal_integral2d(
        self,
        npt,
        npt_azim=360,
        mask=None,
        radial_range=None,
        azimuth_range=None,
        inplace=False,
        method="splitpixel",
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
            If mask is a baseSignal then it is itereated over as well.
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
        parallel: bool
            If true launches paralell workers for the integration
        max_workers: int
            The number of streams to initialize. Only used if parallel=True

        Returns
        -------
        polar: PolarDiffraction2D
            A polar diffraction signal

        Examples
        --------
        Basic case using "2th_deg" units (no wavelength needed)

        >>> ds.unit = "2th_deg"
        >>> ds.get_azimuthal_integral2d(npt_rad=100)

        Basic case using a curved Ewald Sphere approximation and pyXEM units
        (wavelength needed)

        >>> ds.unit = "k_nm^-1" # setting units
        >>> ds.get_azimuthal_integral2d(npt_rad=100, wavelength=2.5e-12)

        Using pyFAI to define a detector case using a curved Ewald Sphere approximation and pyXEM units

        >>> from pyFAI.detectors import Detector
        >>> det = Detector(pixel1=1e-4, pixel2=1e-4)
        >>> ds.get_azimuthal_integral2d(npt_rad=100, detector_dist=.2, detector= det, wavelength=2.508e-12)
        """
        sig_shape = self.axes_manager.signal_shape
        unit = self.unit
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

        # Dealing with axis changes
        if inplace:
            t_axis = self.axes_manager.signal_axes[0]
            k_axis = self.axes_manager.signal_axes[1]
            self.set_signal_type("polar_diffraction")
        else:
            transfer_navigation_axes(integration, self)
            integration.set_signal_type("polar_diffraction")
            t_axis = integration.axes_manager.signal_axes[0]
            k_axis = integration.axes_manager.signal_axes[1]
        t_axis.name = "Radians"
        if azimuth_range is None:
            t_axis.scale = np.pi * 2 / npt_azim
            t_axis.offset = -np.pi
        else:
            t_axis.scale = (azimuth_range[1] - azimuth_range[0]) / npt
            t_axis.offset = azimuth_range[0]
        k_axis.name = "Radius"
        k_axis.scale = (radial_range[1] - radial_range[0]) / npt
        k_axis.units = unit
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
            If mask is a baseSignal then it is itereated over as well.
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
        parallel: bool
            If true launches paralell workers for the integration
        max_workers: int
            The number of streams to initialize. Only used if parallel=True

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
        signal_type = self._signal_type
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

        # Dealing with axis changes
        if inplace:
            k_axis = self.axes_manager.signal_axes[0]
            self.set_signal_type(signal_type)
        else:
            integration.set_signal_type(signal_type)
            transfer_navigation_axes(integration, self)
            k_axis = integration.axes_manager.signal_axes[0]
        k_axis.name = "Radius"
        k_axis.scale = (radial_range[1] - radial_range[0]) / npt
        # k_axis.units = unit.unit_symbol
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
            If mask is a baseSignal then it is itereated over as well.
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
        parallel: bool
            If true launches paralell workers for the integration
        max_workers: int
            The number of streams to initialize. Only used if parallel=True

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
        signal_type = self._signal_type
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

        # Dealing with axis changes
        if inplace:
            k_axis = self.axes_manager.signal_axes[0]
            self.set_signal_type(signal_type)
        else:
            integration.set_signal_type(signal_type)
            transfer_navigation_axes(integration, self)
            k_axis = integration.axes_manager.signal_axes[0]
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
            If mask is a baseSignal then it is itereated over as well.
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
        parallel: bool
            If true launches paralell workers for the integration
        max_workers: int
            The number of streams to initialize. Only used if parallel=True

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
        signal_type = self._signal_type
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

        # Dealing with axis changes
        if inplace:
            k_axis = self.axes_manager.signal_axes[0]
            self.set_signal_type(signal_type)
        else:
            integration.set_signal_type(signal_type)
            transfer_navigation_axes(integration, self)
            k_axis = integration.axes_manager.signal_axes[0]
        k_axis.name = "Radius"
        k_axis.scale = (radial_range[1] - radial_range[0]) / npt_rad
        # k_axis.units = unit.unit_symbol
        k_axis.offset = radial_range[0]

        return integration


class LazyDiffraction2D(LazySignal, Diffraction2D):
    pass
