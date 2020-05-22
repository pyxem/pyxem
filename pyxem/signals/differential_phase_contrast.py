# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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

import copy
import numpy as np

from pyxem.utils.dpc_utils import make_bivariate_histogram
import pyxem.utils.pixelated_stem_tools as pst

from hyperspy.signals import BaseSignal, Signal1D, Signal2D
from hyperspy._signals.lazy import LazySignal

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.ndimage import rotate, gaussian_filter
from tqdm import tqdm


class DPCBaseSignal(BaseSignal):
    """
    Signal for processing differential phase contrast (DPC) acquired using
    scanning transmission electron microscopy (STEM).

    The signal assumes the data is 3 dimensions, where the two
    signal dimensions are the probe positions, and the navigation
    dimension is the x and y disk shifts.

    The first navigation index (s.inav[0]) is assumed to the be x-shift
    and the second navigation is the y-shift (s.inav[1]).

    """

    _signal_type = "dpc"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DPCSignal1D(Signal1D):
    """
    Signal for processing differential phase contrast (DPC) acquired using
    scanning transmission electron microscopy (STEM).

    The signal assumes the data is 2 dimensions, where the
    signal dimension is the probe position, and the navigation
    dimension is the x and y disk shifts.

    The first navigation index (s.inav[0]) is assumed to the be x-shift
    and the second navigation is the y-shift (s.inav[1]).

    """

    _signal_type = "dpc"

    def get_bivariate_histogram(
        self, histogram_range=None, masked=None, bins=200, spatial_std=3
    ):
        """
        Useful for finding the distribution of magnetic vectors(?).

        Parameters
        ----------
        histogram_range : tuple, optional
            Set the minimum and maximum of the histogram range.
            Default is setting it automatically.
        masked : 1-D NumPy bool array, optional
            Mask parts of the data. The array must be the same
            size as the signal. The True values are masked.
            Default is not masking anything.
        bins : integer, default 200
            Number of bins in the histogram
        spatial_std : number, optional
            If histogram_range is not given, this value will be
            used to set the automatic histogram range.
            Default value is 3.

        Returns
        -------
        s_hist : Signal2D

        """
        x_position = self.inav[0].data
        y_position = self.inav[1].data
        s_hist = make_bivariate_histogram(
            x_position,
            y_position,
            histogram_range=histogram_range,
            masked=masked,
            bins=bins,
            spatial_std=spatial_std,
        )

        return s_hist


class DPCSignal2D(Signal2D):
    """
    Signal for processing differential phase contrast (DPC) acquired using
    scanning transmission electron microscopy (STEM).

    The signal assumes the data is 3 dimensions, where the two
    signal dimensions are the probe positions, and the navigation
    dimension is the x and y disk shifts.

    The first navigation index (s.inav[0]) is assumed to the be x-shift
    and the second navigation is the y-shift (s.inav[1]).

    """

    _signal_type = "dpc"

    def correct_ramp(self, corner_size=0.05, only_offset=False, out=None):
        """
        Subtracts a plane from the signal, useful for removing
        the effects of d-scan in a STEM beam shift dataset.

        The plane is calculated by fitting a plane to the corner values
        of the signal. This will only work well when the property one
        wants to measure is zero in these corners.

        Parameters
        ----------
        corner_size : number, optional
            The size of the corners, as a percentage of the image's axis.
            If corner_size is 0.05 (5%), and the image is 500 x 1000,
            the size of the corners will be (500*0.05) x (1000*0.05) = 25 x 50.
            Default 0.05
        only_offset : bool, optional
            If True, will subtract a "flat" plane, i.e. it will subtract the
            mean value of the corners. Default False
        out : optional, DPCSignal2D signal

        Returns
        -------
        corrected_signal : Signal2D

        Examples
        --------
        >>> s = ps.dummy_data.get_square_dpc_signal(add_ramp=True)
        >>> s_corr = s.correct_ramp()
        >>> s_corr.plot()

        Only correct offset

        >>> s_corr = s.correct_ramp(only_offset=True)
        >>> s_corr.plot()

        """
        if out is None:
            output = self.deepcopy()
        else:
            output = out

        for i, s in enumerate(self):
            if only_offset:
                corners = pst._get_corner_values(s, corner_size=corner_size)[2]
                ramp = corners.mean()
            else:
                ramp = pst._fit_ramp_to_image(s, corner_size=0.05)
            output.data[i, :, :] -= ramp
        if out is None:
            return output

    def get_magnitude_signal(self, autolim=True, autolim_sigma=4):
        """Get DPC magnitude image visualized as greyscale.

        Converts the x and y beam shifts into a magnitude map, showing the
        magnitude of the beam shifts.

        Useful for visualizing magnetic domain structures.

        Parameters
        ----------
        autolim : bool, default True
        autolim_sigma : float, default 4

        Returns
        -------
        magnitude_signal : HyperSpy 2D signal

        Examples
        --------
        >>> s = ps.dummy_data.get_simple_dpc_signal()
        >>> s_magnitude = s.get_magnitude_signal()
        >>> s_magnitude.plot()

        See also
        --------
        get_color_signal : Signal showing both phase and magnitude
        get_phase_signal : Signal showing the phase

        """
        inav02 = np.abs(self.inav[0].data) ** 2
        inav12 = np.abs(self.inav[1].data) ** 2
        magnitude = np.sqrt(inav02 + inav12)
        magnitude_limits = None
        if autolim:
            magnitude_limits = pst._get_limits_from_array(
                magnitude, sigma=autolim_sigma
            )
            np.clip(magnitude, magnitude_limits[0], magnitude_limits[1], out=magnitude)

        signal = Signal2D(magnitude)
        pst._copy_signal2d_axes_manager_metadata(self, signal)
        return signal

    def get_phase_signal(self, rotation=None):
        """Get DPC phase image visualized using continuous color scale.

        Converts the x and y beam shifts into an RGB array, showing the
        direction of the beam shifts.

        Useful for visualizing magnetic domain structures.

        Parameters
        ----------
        rotation : float, optional
            In degrees. Useful for correcting the mismatch between
            scan direction and diffraction pattern rotation.
        autolim : bool, default True
        autolim_sigma : float, default 4

        Returns
        -------
        phase_signal : HyperSpy 2D RGB signal

        Examples
        --------
        >>> s = ps.dummy_data.get_simple_dpc_signal()
        >>> s_color = s.get_phase_signal(rotation=20)
        >>> s_color.plot()

        See also
        --------
        get_color_signal : Signal showing both phase and magnitude
        get_magnitude_signal : Signal showing the magnitude

        """
        # Rotate the phase by -30 degrees in the color "wheel", to get better
        # visualization in the vertical and horizontal direction.
        if rotation is None:
            rotation = -30
        else:
            rotation = rotation - 30
        phase = np.arctan2(self.inav[0].data, self.inav[1].data) % (2 * np.pi)
        rgb_array = pst._get_rgb_phase_array(phase=phase, rotation=rotation)
        signal_rgb = Signal1D(rgb_array * (2 ** 16 - 1))
        signal_rgb.change_dtype("uint16")
        signal_rgb.change_dtype("rgb16")
        pst._copy_signal2d_axes_manager_metadata(self, signal_rgb)
        return signal_rgb

    def get_color_signal(self, rotation=None, autolim=True, autolim_sigma=4):
        """Get DPC image visualized using continuous color scale.

        Converts the x and y beam shifts into an RGB array, showing the
        magnitude and direction of the beam shifts.

        Useful for visualizing magnetic domain structures.

        Parameters
        ----------
        rotation : float, optional
            In degrees. Useful for correcting the mismatch between
            scan direction and diffraction pattern rotation.
        autolim : bool, default True
        autolim_sigma : float, default 4

        Returns
        -------
        color_signal : HyperSpy 2D RGB signal

        Examples
        --------
        >>> s = ps.dummy_data.get_simple_dpc_signal()
        >>> s_color = s.get_color_signal()
        >>> s_color.plot()

        Rotate the beam shift by 30 degrees

        >>> s_color = s.get_color_signal(rotation=30)

        See also
        --------
        get_color_signal : Signal showing both phase and magnitude
        get_phase_signal : Signal showing the phase

        """
        # Rotate the phase by -30 degrees in the color "wheel", to get better
        # visualization in the vertical and horizontal direction.
        if rotation is None:
            rotation = -30
        else:
            rotation = rotation - 30
        inav0 = self.inav[0].data
        inav1 = self.inav[1].data
        phase = np.arctan2(inav0, inav1) % (2 * np.pi)
        magnitude = np.sqrt(np.abs(inav0) ** 2 + np.abs(inav1) ** 2)

        magnitude_limits = None
        if autolim:
            magnitude_limits = pst._get_limits_from_array(
                magnitude, sigma=autolim_sigma
            )
        rgb_array = pst._get_rgb_phase_magnitude_array(
            phase=phase,
            magnitude=magnitude,
            rotation=rotation,
            magnitude_limits=magnitude_limits,
        )
        signal_rgb = Signal1D(rgb_array * (2 ** 16 - 1))
        signal_rgb.change_dtype("uint16")
        signal_rgb.change_dtype("rgb16")
        pst._copy_signal2d_axes_manager_metadata(self, signal_rgb)
        return signal_rgb

    def get_color_image_with_indicator(
        self,
        phase_rotation=0,
        indicator_rotation=0,
        only_phase=False,
        autolim=True,
        autolim_sigma=4,
        scalebar_size=None,
        ax=None,
        ax_indicator=None,
    ):
        """Make a matplotlib figure showing DPC contrast.

        Parameters
        ----------
        phase_rotation : float, default 0
            Changes the phase of the plotted data.
            Useful for correcting scan rotation.
        indicator_rotation : float, default 0
            Changes the color wheel rotation.
        only_phase : bool, default False
            If False, will plot both the magnitude and phase.
            If True, will only plot the phase.
        autolim : bool, default True
        autolim_sigma : float, default 4
        scalebar_size : int, optional
        ax : Matplotlib subplot, optional
        ax_indicator : Matplotlib subplot, optional
            If None, generate a new subplot for the indicator.
            If False, do not include an indicator

        Examples
        --------
        >>> s = ps.dummy_data.get_simple_dpc_signal()
        >>> fig = s.get_color_image_with_indicator()
        >>> fig.savefig("simple_dpc_test_signal.png")

        Only plotting the phase

        >>> fig = s.get_color_image_with_indicator(only_phase=True)
        >>> fig.savefig("simple_dpc_test_signal.png")

        Matplotlib subplot as input

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax_indicator = fig.add_subplot(331)
        >>> fig_return = s.get_color_image_with_indicator(
        ...     scalebar_size=10, ax=ax, ax_indicator=ax_indicator)

        """
        indicator_rotation = indicator_rotation + 60
        if ax is None:
            set_fig = True
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        else:
            fig = ax.figure
            set_fig = False
        if only_phase:
            s = self.get_phase_signal(rotation=phase_rotation)
        else:
            s = self.get_color_signal(
                rotation=phase_rotation, autolim=autolim, autolim_sigma=autolim_sigma
            )
        s.change_dtype("uint16")
        s.change_dtype("float64")
        extent = self.axes_manager.signal_extent
        extent = [extent[0], extent[1], extent[3], extent[2]]
        ax.imshow(s.data / 65536.0, extent=extent)
        if ax_indicator is not False:
            if ax_indicator is None:
                ax_indicator = fig.add_subplot(331)
            pst._make_color_wheel(
                ax_indicator, rotation=indicator_rotation + phase_rotation
            )
        ax.set_axis_off()
        if scalebar_size is not None:
            scalebar_label = "{0} {1}".format(scalebar_size, s.axes_manager[0].units)
            sb = AnchoredSizeBar(ax.transData, scalebar_size, scalebar_label, loc=4)
            ax.add_artist(sb)
        if set_fig:
            fig.subplots_adjust(0, 0, 1, 1)
        return fig

    def get_bivariate_histogram(
        self, histogram_range=None, masked=None, bins=200, spatial_std=3
    ):
        """
        Useful for finding the distribution of magnetic vectors(?).

        Parameters
        ----------
        histogram_range : tuple, optional
            Set the minimum and maximum of the histogram range.
            Default is setting it automatically.
        masked : 2-D NumPy bool array, optional
            Mask parts of the data. The array must be the same
            size as the signal. The True values are masked.
            Default is not masking anything.
        bins : integer, default 200
            Number of bins in the histogram
        spatial_std : number, optional
            If histogram_range is not given, this value will be
            used to set the automatic histogram range.
            Default value is 3.

        Returns
        -------
        s_hist : HyperSpy Signal2D

        Examples
        --------
        >>> s = ps.dummy_data.get_stripe_pattern_dpc_signal()
        >>> s_hist = s.get_bivariate_histogram()
        >>> s_hist.plot()

        """
        x_position = self.inav[0].data
        y_position = self.inav[1].data
        s_hist = make_bivariate_histogram(
            x_position,
            y_position,
            histogram_range=histogram_range,
            masked=masked,
            bins=bins,
            spatial_std=spatial_std,
        )
        s_hist.metadata.General.title = "Bivariate histogram of {0}".format(
            self.metadata.General.title
        )
        return s_hist

    def flip_axis_90_degrees(self, flips=1):
        """Flip both the spatial and beam deflection axis

        Will rotate both the image and the beam deflections
        by 90 degrees.

        Parameters
        ----------
        flips : int, default 1
            Number of flips. The default (1) gives 90 degrees rotation.
            2 gives 180, 3 gives 270, ...

        Examples
        --------
        >>> s = ps.dummy_data.get_stripe_pattern_dpc_signal()
        >>> s
        <DPCSignal2D, title: , dimensions: (2|50, 100)>
        >>> s_rot = s.flip_axis_90_degrees()
        >>> s_rot
        <DPCSignal2D, title: , dimensions: (2|100, 50)>

        Do several flips

        >>> s_rot = s.flip_axis_90_degrees(2)
        >>> s_rot
        <DPCSignal2D, title: , dimensions: (2|50, 100)>
        >>> s_rot = s.flip_axis_90_degrees(3)
        >>> s_rot
        <DPCSignal2D, title: , dimensions: (2|100, 50)>

        """
        s_out = self.deepcopy()
        for i in range(flips):
            data0 = copy.deepcopy(s_out.data[0])
            data1 = copy.deepcopy(s_out.data[1])
            s_out = s_out.swap_axes(1, 2)
            s_out.data[0] = np.rot90(data0, -1)
            s_out.data[1] = np.rot90(data1, -1)
            s_out = s_out.rotate_beam_shifts(90)
        return s_out

    def rotate_data(self, angle, reshape=False):
        """Rotate the scan dimensions by angle.

        Parameters
        ----------
        angle : float
            Clockwise rotation in degrees

        Returns
        -------
        rotated_signal : DPCSignal2D

        Example
        -------

        Rotate data by 10 degrees clockwise

        >>> s = ps.dummy_data.get_simple_dpc_signal()
        >>> s_rot = s.rotate_data(10)
        >>> s_rot.plot()

        """
        s_new = self.map(
            rotate, show_progressbar=False, inplace=False, reshape=reshape, angle=-angle
        )
        return s_new

    def rotate_beam_shifts(self, angle):
        """Rotate the beam shift vector.

        Parameters
        ----------
        angle : float
            Clockwise rotation in degrees

        Returns
        -------
        shift_rotated_signal : DPCSignal2D

        Example
        -------

        Rotate beam shifts by 10 degrees clockwise

        >>> s = ps.dummy_data.get_simple_dpc_signal()
        >>> s_new = s.rotate_beam_shifts(10)
        >>> s_new.plot()

        """
        s_new = self.deepcopy()
        angle_rad = np.deg2rad(angle)
        x, y = self.inav[0].data, self.inav[1].data
        s_new.data[0] = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        s_new.data[1] = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        return s_new

    def gaussian_blur(self, sigma=2, output=None):
        """Blur the x- and y-beam shifts.

        Useful for reducing the effects of structural diffraction effects.

        Parameters
        ----------
        sigma : scalar, default 2
        output : HyperSpy signal

        Returns
        -------
        blurred_signal : HyperSpy 2D Signal

        Examples
        --------
        >>> s = ps.dummy_data.get_square_dpc_signal(add_ramp=False)
        >>> s_blur = s.gaussian_blur()

        Different sigma

        >>> s_blur = s.gaussian_blur(sigma=1.2)

        Using the signal itself as output

        >>> s.gaussian_blur(output=s)
        >>> s.plot()

        """
        if output is None:
            s_out = self.deepcopy()
        else:
            s_out = output
        gaussian_filter(self.data[0], sigma=sigma, output=s_out.data[0])
        gaussian_filter(self.data[1], sigma=sigma, output=s_out.data[1])
        if output is None:
            return s_out


class LazyDPCBaseSignal(LazySignal, DPCBaseSignal):
    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LazyDPCSignal1D(LazySignal, DPCSignal1D):
    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LazyDPCSignal2D(LazySignal, DPCSignal2D):
    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
