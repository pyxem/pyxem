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

import numpy as np
import dask.array as da
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
from skimage import morphology
from skimage.draw import polygon

from hyperspy.components1d import Gaussian

from pyxem.dummy_data import make_diffraction_test_data as mdtd
from pyxem.signals import DPCSignal2D, Diffraction2D, LazyDiffraction2D


def get_disk_shift_simple_test_signal(lazy=False):
    """Get HyperSpy 2D signal with 2D navigation dimensions for DPC testing.

    Probe size x/y (20, 20), and image size x/y (50, 50).
    Disk moves from 22-28 x/y.

    Parameters
    ----------
    lazy : bool, default False

    Returns
    -------
    disk_shift_signal : Diffraction2D

    Examples
    --------
    >>> s = pxm.dummy_data.get_disk_shift_simple_test_signal()
    >>> s.plot()

    Load as lazy

    >>> s = pxm.dummy_data.get_disk_shift_simple_test_signal(lazy=True)

    """
    disk_x, disk_y = np.mgrid[22:28:20j, 22:28:20j]
    s = mdtd.generate_4d_data(
        probe_size_x=20,
        probe_size_y=20,
        image_size_x=50,
        image_size_y=50,
        disk_x=disk_x,
        disk_y=disk_y,
        disk_r=2,
        ring_x=None,
        add_noise=True,
        lazy=lazy,
    )
    return s


def get_holz_simple_test_signal(lazy=False):
    """Get HyperSpy 2D signal with 2D navigation dimensions for HOLZ testing.

    Probe size x/y (20, 20), and image size x/y (50, 50).
    Contains a disk and a ring. The disk stays at x, y = 25, 25, with radius 2.
    The ring has a radius of 20, and moves from x, y = 24-26, 24-26.

    Parameters
    ----------
    lazy : bool, default False

    Returns
    -------
    holz_signal : Diffraction2D

    Examples
    --------
    >>> s = pxm.dummy_data.get_holz_simple_test_signal()
    >>> s.plot()

    Load as lazy

    >>> s = pxm.dummy_data.get_holz_simple_test_signal(lazy=True)

    """
    ring_x, ring_y = np.mgrid[24:26:20j, 24:26:20j]
    s = mdtd.generate_4d_data(
        probe_size_x=20,
        probe_size_y=20,
        image_size_x=50,
        image_size_y=50,
        disk_x=25,
        disk_y=25,
        disk_r=2,
        disk_I=20,
        ring_x=ring_x,
        ring_y=ring_y,
        ring_r=15,
        ring_I=10,
        add_noise=True,
        lazy=lazy,
    )
    return s


def get_holz_heterostructure_test_signal(lazy=False):
    """Get HyperSpy 2D signal with 2D navigation dimensions for HOLZ testing.

    The centre, radius and intensity of the ring varies as a function of probe
    position. The disk centre position varies as a function of probe position.

    Parameters
    ----------
    lazy : bool, default False

    Returns
    -------
    holz_signal : HyperSpy 2D signal

    Example
    -------
    >>> s = pxm.dummy_data.get_holz_heterostructure_test_signal()
    >>> s.plot()

    Load as lazy

    >>> s = pxm.dummy_data.get_holz_heterostructure_test_signal(lazy=True)

    """
    probe_size_x, probe_size_y = 40, 40
    px, py = np.mgrid[0:probe_size_x:1, 0:probe_size_y:1]
    x, y = np.mgrid[36:38:40j, 41:43:40j]
    disk_r = 10
    disk_I = np.ones_like(x) * 100 + np.random.random() * 20
    g_r = Gaussian(A=20, centre=25, sigma=5)
    ring_r = np.ones_like(x) * 30 + g_r.function(py)
    g_I = Gaussian(A=30, centre=25, sigma=3)
    ring_I = np.ones_like(x) * 20 + g_I.function(py)
    s = mdtd.generate_4d_data(
        probe_size_x=probe_size_x,
        probe_size_y=probe_size_y,
        image_size_x=80,
        image_size_y=80,
        disk_x=x,
        disk_y=y,
        disk_r=disk_r,
        disk_I=disk_I,
        ring_x=x,
        ring_y=y,
        ring_r=ring_r,
        ring_I=ring_I,
        add_noise=True,
        lazy=lazy,
    )
    return s


def get_single_ring_diffraction_signal():
    """Get HyperSpy 2D signal with a single ring with centre position.

    The ring has a centre at x=105 and y=67, and radius=40.
    """
    data = mdtd.MakeTestData(size_x=200, size_y=150, default=False, blur=True)
    x, y = 105, 67
    data.add_ring(x, y, r=40)
    s = data.signal
    s.axes_manager[0].offset, s.axes_manager[1].offset = -x, -y
    return s


def get_dead_pixel_signal(lazy=False):
    """Get Diffraction2D signal with a disk in the middle.

    Has 4 pixels with value equal to 0, to simulate dead pixels.

    Example
    -------
    >>> s = pxm.dummy_data.get_dead_pixel_signal()

    Lazy signal

    >>> s_lazy = pxm.dummy_data.get_dead_pixel_signal(lazy=True)

    """
    data = mdtd.MakeTestData(size_x=128, size_y=128, default=False, blur=True)
    data.add_disk(64, 64, r=30, intensity=10000)
    s = data.signal
    s.change_dtype("int64")
    s.data += gaussian_filter(s.data, sigma=50)
    s.data[61, 73] = 0
    s.data[46, 53] = 0
    s.data[88, 88] = 0
    s.data[112, 20] = 0
    if lazy:
        s = LazyDiffraction2D(s)
        s.data = da.from_array(s.data, chunks=(64, 64))
    else:
        s = Diffraction2D(s)
    return s


def get_hot_pixel_signal(lazy=False):
    """Get Diffraction2D signal with a disk in the middle.

    Has 4 pixels with value equal to 50000, to simulate hot pixels.

    Example
    -------
    >>> s = pxm.dummy_data.get_hot_pixel_signal()

    Lazy signal

    >>> s_lazy = pxm.dummy_data.get_hot_pixel_signal(lazy=True)

    """
    data = mdtd.MakeTestData(size_x=128, size_y=128, default=False, blur=True)
    data.add_disk(64, 64, r=30, intensity=10000)
    s = data.signal
    s.change_dtype("int64")
    s.data += gaussian_filter(s.data, sigma=50)
    s.data[76, 4] = 50000
    s.data[12, 102] = 50000
    s.data[32, 10] = 50000
    s.data[120, 61] = 50000
    if lazy:
        s = LazyDiffraction2D(s)
        s.data = da.from_array(s.data, chunks=(64, 64))
    else:
        s = Diffraction2D(s)
    return s


def get_simple_dpc_signal():
    """Get a simple DPCSignal2D with a zero point in the centre.

    Example
    -------
    >>> s = pxm.dummy_data.get_simple_dpc_signal()

    """
    data = np.mgrid[-5:5:100j, -5:5:100j]
    s = DPCSignal2D(data)
    return s


def get_stripe_pattern_dpc_signal():
    """Get a 2D DPC signal with a stripe pattern.

    The stripe pattern only has an x-component, with alternating left/right
    directions. There is a small a net moment in the positive x-direction
    (leftwards).

    Returns
    -------
    stripe_dpc_signal : DPCSignal2D

    Example
    -------
    >>> s = pxm.dummy_data.get_stripe_pattern_dpc_signal()

    """
    data = np.zeros((2, 100, 50))
    for i in range(10, 90, 20):
        data[0, i : i + 10, 10:40] = 1.1
        data[0, i + 10 : i + 20, 10:40] = -1
    s = DPCSignal2D(data)
    return s


def get_square_dpc_signal(add_ramp=False):
    """Get a 2D DPC signal resembling a Landau domain.

    Parameters
    ----------
    add_ramp : bool, default False
        If True, will add a ramp in the beam shift across the image
        to emulate the effects of d-scan.

    Returns
    -------
    square_dpc_signal : DPCSignal2D

    Examples
    --------
    >>> s = pxm.dummy_data.get_square_dpc_signal()
    >>> s.plot()

    Adding a ramp

    >>> s = pxm.dummy_data.get_square_dpc_signal(add_ramp=True)
    >>> s.plot()

    """
    imX, imY = 300, 300
    data_y, data_x = np.random.normal(loc=0.1, size=(2, imY, imX))
    x, y = np.mgrid[-150 : 150 : imY * 1j, -150 : 150 : imX * 1j]
    t = np.arctan2(x, y) % (2 * np.pi)
    mask_xy = (x < 100) * (x > -100) * (y < 100) * (y > -100)
    mask0 = (t > np.pi / 4) * (t < 3 * np.pi / 4) * mask_xy
    mask1 = (t > 3 * np.pi / 4) * (t < 5 * np.pi / 4) * mask_xy
    mask2 = (t > 5 * np.pi / 4) * (t < 7 * np.pi / 4) * mask_xy
    mask3 = ((t > 7 * np.pi / 4) + (t < np.pi / 4)) * mask_xy
    data_y[mask0] -= np.random.normal(loc=3, scale=0.1, size=(imY, imX))[mask0]
    data_x[mask1] -= np.random.normal(loc=3, scale=0.1, size=(imY, imX))[mask1]
    data_y[mask2] += np.random.normal(loc=3, scale=0.1, size=(imY, imX))[mask2]
    data_x[mask3] += np.random.normal(loc=3, scale=0.1, size=(imY, imX))[mask3]
    if add_ramp:
        ramp_y, ramp_x = np.mgrid[18 : 28 : imY * 1j, -5.3 : 1.2 : imX * 1j]
        data_x += ramp_x
        data_y += ramp_y
    s = DPCSignal2D((data_y, data_x))
    s.axes_manager.signal_axes[0].name = "Probe position x"
    s.axes_manager.signal_axes[1].name = "Probe position y"
    s.axes_manager.signal_axes[0].units = "nm"
    s.axes_manager.signal_axes[1].units = "nm"
    return s


def get_fem_signal(lazy=False):
    """Get a 2D signal approximating a fluctuation electron microscopy (FEM) dataset.

    Parameters
    ----------
    lazy : bool, default False
        If True, resulting signal will be lazy.

    Returns
    -------
    fem_signal : Diffraction2D

    Examples
    --------
    >>> s = pxm.dummy_data.get_fem_signal()
    >>> s.plot()

    """
    radii1 = 20 * np.random.randint(0, 2, size=(10, 10))
    intensities1 = np.random.randint(0, 30, size=(10, 10))
    radii2 = 35 * np.random.randint(0, 2, size=(10, 10))
    intensities2 = np.random.randint(0, 30, size=(10, 10))

    test1 = mdtd.generate_4d_data(
        probe_size_x=10,
        probe_size_y=10,
        image_size_x=100,
        image_size_y=100,
        disk_x=50,
        disk_y=50,
        disk_r=5,
        disk_I=100,
        ring_x=50,
        ring_y=50,
        ring_r=radii1,
        ring_I=intensities1,
        ring_lw=0,
        blur=True,
        blur_sigma=1,
        downscale=True,
        add_noise=True,
        show_progressbar=False,
        lazy=lazy,
    )

    test2 = mdtd.generate_4d_data(
        probe_size_x=10,
        probe_size_y=10,
        image_size_x=100,
        image_size_y=100,
        disk_x=50,
        disk_y=50,
        disk_r=5,
        disk_I=100,
        ring_x=50,
        ring_y=50,
        ring_r=radii2,
        ring_I=intensities2,
        ring_lw=0,
        blur=True,
        blur_sigma=1,
        downscale=True,
        add_noise=True,
        show_progressbar=False,
        lazy=lazy,
    )

    fem_signal = test1 + test2

    fem_signal.axes_manager.navigation_axes[0].name = "Probe position x"
    fem_signal.axes_manager.navigation_axes[0].units = "nm"
    fem_signal.axes_manager.navigation_axes[0].scale = 1.0
    fem_signal.axes_manager.navigation_axes[0].offset = 0

    fem_signal.axes_manager.navigation_axes[1].name = "Probe position y"
    fem_signal.axes_manager.navigation_axes[1].units = "nm"
    fem_signal.axes_manager.navigation_axes[1].scale = 1.0
    fem_signal.axes_manager.navigation_axes[1].offset = 0

    fem_signal.axes_manager.signal_axes[0].name = "Signal x"
    fem_signal.axes_manager.signal_axes[0].units = "mrads"
    fem_signal.axes_manager.signal_axes[0].scale = 0.25
    fem_signal.axes_manager.signal_axes[0].offset = 0

    fem_signal.axes_manager.signal_axes[1].name = "Signal y"
    fem_signal.axes_manager.signal_axes[1].units = "mrads"
    fem_signal.axes_manager.signal_axes[1].scale = 0.25
    fem_signal.axes_manager.signal_axes[1].offset = 0

    return fem_signal


def get_simple_fem_signal(lazy=False):
    """2D signal approximating a very small fluctuation electron microscopy (FEM) dataset.

    Parameters
    ----------
    lazy : bool, default False
        If True, resulting signal will be lazy.

    Returns
    -------
    fem_signal : Diffraction2D

    Examples
    --------
    >>> s = pxm.dummy_data.get_simple_fem_signal()
    >>> s.plot()

    """

    radii1 = 10 * np.random.randint(0, 2, size=(2, 2))
    intensities1 = np.random.randint(0, 5, size=(2, 2))

    radii2 = 20 * np.random.randint(0, 2, size=(2, 2))
    intensities2 = np.random.randint(0, 15, size=(2, 2))

    test1 = mdtd.generate_4d_data(
        probe_size_x=2,
        probe_size_y=2,
        image_size_x=50,
        image_size_y=50,
        disk_x=25,
        disk_y=25,
        disk_r=5,
        disk_I=100,
        ring_x=25,
        ring_y=25,
        ring_r=radii1,
        ring_I=intensities1,
        ring_lw=0,
        blur=True,
        blur_sigma=1,
        downscale=True,
        add_noise=True,
        show_progressbar=False,
        lazy=lazy,
    )

    test2 = mdtd.generate_4d_data(
        probe_size_x=2,
        probe_size_y=2,
        image_size_x=50,
        image_size_y=50,
        disk_x=25,
        disk_y=25,
        disk_r=5,
        disk_I=100,
        ring_x=25,
        ring_y=25,
        ring_r=radii2,
        ring_I=intensities2,
        ring_lw=0,
        blur=True,
        blur_sigma=1,
        downscale=True,
        add_noise=True,
        show_progressbar=False,
        lazy=lazy,
    )

    fem_signal = test1 + test2

    fem_signal.axes_manager.navigation_axes[0].name = "Probe position x"
    fem_signal.axes_manager.navigation_axes[0].units = "nm"
    fem_signal.axes_manager.navigation_axes[0].scale = 1.0
    fem_signal.axes_manager.navigation_axes[0].offset = 0

    fem_signal.axes_manager.navigation_axes[1].name = "Probe position y"
    fem_signal.axes_manager.navigation_axes[1].units = "nm"
    fem_signal.axes_manager.navigation_axes[1].scale = 1.0
    fem_signal.axes_manager.navigation_axes[1].offset = 0

    fem_signal.axes_manager.signal_axes[0].name = "Signal x"
    fem_signal.axes_manager.signal_axes[0].units = "mrads"
    fem_signal.axes_manager.signal_axes[0].scale = 0.25
    fem_signal.axes_manager.signal_axes[0].offset = 0

    fem_signal.axes_manager.signal_axes[1].name = "Signal y"
    fem_signal.axes_manager.signal_axes[1].units = "mrads"
    fem_signal.axes_manager.signal_axes[1].scale = 0.25
    fem_signal.axes_manager.signal_axes[1].offset = 0

    return fem_signal


def get_generic_fem_signal(probe_x=2, probe_y=2, image_x=50, image_y=50, lazy=False):
    """2D signal approximating a fluctuation electron microscopy (FEM) dataset with user defined dimensions.

    Parameters
    ----------
    probe_x : int, default 2
        Horizontal dimension of the navigation axes
    probe_y : int, default 2
        Vertical dimension of the navigation axes
    image_x : int, default 2
        Horizontal dimension of the signal axes
    image_y : int, default 2
        Vertical dimension of the signal axes
    lazy : bool, default False
        If True, resulting signal will be lazy.

    Returns
    -------
    fem_signal : Diffraction2D

    Examples
    --------
    >>> s = pxm.dummy_data.get_generic_fem_signal(probe_x=5, probe_y=10,
    ...     image_x=25, image_y=30, lazy=False)
    >>> s.plot()

    """
    image_center = [np.int(image_x / 2), np.int(image_y / 2)]

    radii1 = 10 * np.random.randint(0, 2, size=(probe_y, probe_x))
    intensities1 = np.random.randint(0, 5, size=(probe_y, probe_x))

    radii2 = 20 * np.random.randint(0, 2, size=(probe_y, probe_x))
    intensities2 = np.random.randint(0, 15, size=(probe_y, probe_x))

    test1 = mdtd.generate_4d_data(
        probe_size_x=probe_x,
        probe_size_y=probe_y,
        image_size_x=image_x,
        image_size_y=image_y,
        disk_x=image_center[0],
        disk_y=image_center[1],
        disk_r=5,
        disk_I=100,
        ring_x=image_center[0],
        ring_y=image_center[1],
        ring_r=radii1,
        ring_I=intensities1,
        ring_lw=0,
        blur=True,
        blur_sigma=1,
        downscale=True,
        add_noise=True,
        show_progressbar=False,
        lazy=lazy,
    )

    test2 = mdtd.generate_4d_data(
        probe_size_x=probe_x,
        probe_size_y=probe_y,
        image_size_x=image_x,
        image_size_y=image_y,
        disk_x=image_center[0],
        disk_y=image_center[1],
        disk_r=5,
        disk_I=100,
        ring_x=image_center[0],
        ring_y=image_center[1],
        ring_r=radii2,
        ring_I=intensities2,
        ring_lw=0,
        blur=True,
        blur_sigma=1,
        downscale=True,
        add_noise=True,
        show_progressbar=False,
        lazy=lazy,
    )

    fem_signal = test1 + test2

    fem_signal.axes_manager.navigation_axes[0].name = "Probe position x"
    fem_signal.axes_manager.navigation_axes[0].units = "nm"
    fem_signal.axes_manager.navigation_axes[0].scale = 1.0
    fem_signal.axes_manager.navigation_axes[0].offset = 0

    fem_signal.axes_manager.navigation_axes[1].name = "Probe position y"
    fem_signal.axes_manager.navigation_axes[1].units = "nm"
    fem_signal.axes_manager.navigation_axes[1].scale = 1.0
    fem_signal.axes_manager.navigation_axes[1].offset = 0

    fem_signal.axes_manager.signal_axes[0].name = "Signal x"
    fem_signal.axes_manager.signal_axes[0].units = "mrads"
    fem_signal.axes_manager.signal_axes[0].scale = 0.25
    fem_signal.axes_manager.signal_axes[0].offset = 0

    fem_signal.axes_manager.signal_axes[1].name = "Signal y"
    fem_signal.axes_manager.signal_axes[1].units = "mrads"
    fem_signal.axes_manager.signal_axes[1].scale = 0.25
    fem_signal.axes_manager.signal_axes[1].offset = 0
    return fem_signal


def get_cbed_signal():
    """Get artificial pixelated STEM signal similar to CBED data.

    Returns
    -------
    cbed_signal : Diffraction2D

    Example
    -------
    >>> s = pxm.dummy_data.get_cbed_signal()
    >>> s.plot()

    """
    data = np.zeros(shape=(10, 10, 100, 100), dtype=np.uint16)
    diff_point_list = [
        [50, 50, 1000],
        [25, 25, 500],
        [50, 25, 500],
        [25, 50, 500],
        [75, 75, 500],
        [75, 50, 500],
        [50, 75, 500],
        [75, 25, 500],
        [25, 75, 500],
    ]
    disk = morphology.disk(5, np.uint16)
    for x, y, intensity in diff_point_list:
        for ix, iy in np.ndindex(data.shape[:2]):
            temp_x = x
            temp_y = y
            temp_intensity = intensity
            if iy < 5:
                if y < 40:
                    temp_y = y + 2
                    temp_intensity = intensity + 20
                elif y > 60:
                    temp_y = y - 2

            data[iy, ix, temp_y, temp_x] = temp_intensity
    for ix, iy in np.ndindex(data.shape[:2]):
        noise = np.random.randint(10, size=(100, 100))
        image = convolve2d(data[iy, ix], disk, mode="same") + noise
        data[iy, ix] = gaussian_filter(image, 1.5)
    s_cbed = Diffraction2D(data)
    return s_cbed


def get_simple_ellipse_signal_peak_array():
    """Get a signal and peak array of an ellipse.

    Returns
    -------
    signal, peak_array : HyperSpy Signal2D, NumPy array

    Examples
    --------
    >>> s, peak_array = pxm.dummy_data.get_simple_ellipse_signal_peak_array()
    >>> s.add_peak_array_as_markers(peak_array, color='blue', size=30)

    """
    xc = np.random.randint(95, 105, size=(4, 5))
    yc = np.random.randint(95, 105, size=(4, 5))
    semi0 = np.random.randint(55, 60, size=(4, 5))
    semi1 = np.random.randint(75, 80, size=(4, 5))
    rot = np.random.random(size=(4, 5)) * np.pi
    peak_array = mdtd._make_4d_peak_array_test_data(xc, yc, semi0, semi1, rot)
    s = Diffraction2D(np.zeros((4, 5, 200, 200)))
    return s, peak_array


def get_nanobeam_electron_diffraction_signal():
    """Get a signal emulating a NBED dataset.

    Returns
    -------
    signal : Diffraction2D

    Example
    -------
    >>> s = pxm.dummy_data.get_nanobeam_electron_diffraction_signal()
    >>> s.plot()

    """
    di0 = mdtd.DiffractionTestImage(intensity_noise=False)
    di0.add_disk(x=128, y=128, intensity=10.0)
    di0.add_cubic_disks(vx=20, vy=20, intensity=2.0, n=5)
    di0.add_background_lorentz(intensity=50, width=30)

    di1 = di0.copy()
    di1.rotation = 10
    di2 = di0.copy()
    di2.rotation = -10

    position_array0 = np.zeros((50, 50), dtype=bool)
    r = np.array([15, 15, 0, 0])
    c = np.array([0, 15, 31, 0])
    rr, cc = polygon(r, c)
    position_array0[rr, cc] = True

    r = np.array([10, 19, 29, 40])
    c = np.array([49, 35, 35, 49])
    rr, cc = polygon(r, c)
    position_array0[rr, cc] = True

    position_array1 = np.zeros((50, 50), dtype=bool)
    r = np.array([32, 41, 41, 49, 49])
    c = np.array([0, 18, 49, 49, 0])
    rr, cc = polygon(r, c)
    position_array1[rr, cc] = True

    position_array2 = np.invert(np.bitwise_or(position_array0, position_array1))

    dtd = mdtd.DiffractionTestDataset(50, 50, 256, 256)
    dtd.add_diffraction_image(di0, position_array0)
    dtd.add_diffraction_image(di1, position_array1)
    dtd.add_diffraction_image(di2, position_array2)

    s = dtd.get_signal()
    return s
