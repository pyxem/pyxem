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

"""Utils for plotting 2D Diffraction Patterns."""

import math
import matplotlib.pyplot as plt
import numpy as np
from pyxem.utils.polar_transform_utils import (
    get_template_cartesian_coordinates,
    get_template_polar_coordinates,
    image_to_polar,
)
from pyxem.utils.diffraction import find_beam_center_blur
import pyxem.utils._beam_shift_tools as bst
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

__all__ = [
    "plot_template_over_pattern",
]


def plot_template_over_pattern(
    pattern,
    simulation,
    ax=None,
    in_plane_angle=0.0,
    max_r=None,
    find_direct_beam=True,
    direct_beam_position=None,
    mirrored_template=False,
    coordinate_system="cartesian",
    marker_color="red",
    marker_type="x",
    size_factor=1.0,
    **kwargs
):
    """A quick utility function to plot a simulated pattern over an experimental image

    Parameters
    ----------
    pattern : 2D np.ndarray
        The diffraction pattern
    simulation : :class:`diffsims.sims.diffraction_simulation.DiffractionSimulation`
        The simulated diffraction pattern. It must be calibrated.
    axis : matplotlib.AxesSubplot, optional
        An axis object on which to plot. If None is provided, one will be created.
    in_plane_angle : float, optional
        An in-plane rotation angle to apply to the template in degrees
    max_r : float, optional
        Maximum radius to consider in the polar transform in pixel coordinates.
        Will only influence the result if ``coordinate_system="polar"``.
    find_direct_beam: bool, optional
        Roughly find the optimal direct beam position if it is not centered.
    direct_beam_position: 2-tuple
        The (x, y) position of the direct beam in pixel coordinates. Takes
        precedence over `find_direct_beam`
    mirrored_template: bool, optional
        Whether to mirror the given template
    coordinate_system : str, optional
        Type of coordinate system to plot the image and template in. Either
        ``cartesian`` or ``polar``
    marker_color : str, optional
        Color of the spot markers
    marker_type : str, optional
        Type of marker used for the spots
    size_factor : float, optional
        Scaling factor for the spots. See notes on size.
    **kwargs :
        See :func:`matplotlib.pyplot.imshow`

    Returns
    -------
    ax : matplotlib.AxesSubplot
        The axes object
    im : matplotlib.image.AxesImage
        The representation of the image on the axes
    sp : matplotlib.collections.PathCollection
        The scatter plot representing the diffraction pattern

    Notes
    -----
    The spot marker sizes are scaled by the square root of their intensity
    """
    if ax is None:
        _, ax = plt.subplots()
    if coordinate_system == "polar":
        pattern = image_to_polar(
            pattern,
            max_r=max_r,
            find_direct_beam=find_direct_beam,
            direct_beam_position=direct_beam_position,
        )
        x, y, intensities = get_template_polar_coordinates(
            simulation,
            in_plane_angle=in_plane_angle,
            max_r=max_r,
            mirrored=mirrored_template,
        )
    elif coordinate_system == "cartesian":
        if direct_beam_position is not None:
            c_x, c_y = direct_beam_position
        elif find_direct_beam:
            c_x, c_y = find_beam_center_blur(pattern, 1)
        else:
            c_y, c_x = pattern.shape[0] / 2, pattern.shape[1] / 2
        x, y, intensities = get_template_cartesian_coordinates(
            simulation,
            center=(c_x, c_y),
            in_plane_angle=in_plane_angle,
            window_size=(pattern.shape[1], pattern.shape[0]),
            mirrored=mirrored_template,
        )
        y = pattern.shape[0] - y
    else:
        raise NotImplementedError(
            "Only polar and cartesian are accepted coordinate systems"
        )
    im = ax.imshow(pattern, **kwargs)
    sp = ax.scatter(
        x,
        y,
        s=size_factor * np.sqrt(intensities),
        marker=marker_type,
        color=marker_color,
    )
    return (ax, im, sp)


def plot_beam_shift_color(
    signal,
    phase_rotation=0,
    indicator_rotation=0,
    only_phase=False,
    autolim=True,
    autolim_sigma=4,
    magnitude_limits=None,
    scalebar_size=None,
    ax=None,
    ax_indicator=None,
):
    """Make a matplotlib figure showing beam shift.

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
    magnitude_limits : tuple of floats, default None
        Manually sets the value limits for the color signal.
        For this, autolim needs to be False.
    scalebar_size : int, optional
    ax : Matplotlib subplot, optional
    ax_indicator : Matplotlib subplot, optional
        If None, generate a new subplot for the indicator.
        If False, do not include an indicator

    Examples
    --------
    >>> s = pxm.data.dummy_data.get_simple_beam_shift_signal()
    >>> fig = pxm.utils.plotting.plot_beam_shift_color(s)
    >>> fig.savefig("simple_beam_shift_test_signal.png")

    Only plotting the phase

    >>> fig = pxm.utils.plotting.plot_beam_shift_color(s, only_phase=True)
    >>> fig.savefig("simple_beam_shift_test_signal.png")

    Matplotlib subplot as input

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax_indicator = fig.add_subplot(331)
    >>> fig = pxm.utils.plotting.plot_beam_shift_color(
    ...     s, scalebar_size=10, ax=ax, ax_indicator=ax_indicator)

    """
    indicator_rotation = indicator_rotation + 60
    if ax is None:
        set_fig = True
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    else:
        fig = ax.figure
        set_fig = False
    if only_phase:
        s = signal.get_phase_signal(rotation=phase_rotation)
    else:
        s = signal.get_magnitude_phase_signal(
            rotation=phase_rotation,
            autolim=autolim,
            autolim_sigma=autolim_sigma,
            magnitude_limits=magnitude_limits,
        )
    s.change_dtype("uint16")
    s.change_dtype("float64")
    extent = s.axes_manager.navigation_extent
    extent = [extent[0], extent[1], extent[3], extent[2]]
    ax.imshow(s.data / 65536.0, extent=extent)
    if ax_indicator is not False:
        if ax_indicator is None:
            ax_indicator = fig.add_subplot(331)
        _make_color_wheel(ax_indicator, rotation=indicator_rotation + phase_rotation)
    ax.set_axis_off()
    if scalebar_size is not None:
        scalebar_label = "{0} {1}".format(scalebar_size, s.axes_manager[0].units)
        sb = AnchoredSizeBar(ax.transData, scalebar_size, scalebar_label, loc=4)
        ax.add_artist(sb)
    if set_fig:
        fig.subplots_adjust(0, 0, 1, 1)
    return fig


def _make_color_wheel(ax, rotation=None):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x**2 + y**2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if rotation is not None:
        t += math.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = bst._get_rgb_phase_magnitude_array(t, r_masked.data)
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()
