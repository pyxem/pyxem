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
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate
from scipy.signal import convolve2d
from skimage import morphology
import dask.array as da

from hyperspy.misc.utils import isiterable

from pyxem.signals import Diffraction2D, LazyDiffraction2D
import pyxem.utils.ransac_ellipse_tools as ret


def _get_elliptical_disk(xx, yy, x, y, semi_len0, semi_len1, rotation):
    """Private function to generate an elliptical disk.

    Parameters
    ----------
    xx, yy : 2D NumPy array
    x, y : float
    semi_len0, semi_len1 : float
    rotation : float
        In radians

    Returns
    -------
    ellipse_array : 2D NumPy array

    Examples
    --------
    >>> from hyperspy.signals import Signal2D
    >>> from pyxem.dummy_data import make_diffraction_test_data as mdtd
    >>> s = Signal2D(np.zeros((110, 130)))
    >>> s.axes_manager[0].offset, s.axes_manager[1].offset = -50, -80
    >>> xx, yy = np.meshgrid(
    ...     s.axes_manager.signal_axes[0].axis,
    ...     s.axes_manager.signal_axes[1].axis)
    >>> ellipse_data = mdtd._get_elliptical_disk(xx, yy, 10, -10, 12, 18, 1.5)
    >>> s.data += ellipse_data
    >>> s.plot()

    """
    rot = rotation
    xx0 = xx - x
    yy0 = yy - y
    semi_len02 = semi_len0 * semi_len0
    semi_len12 = semi_len1 * semi_len1
    z0 = ((xx0 * np.cos(rot) + yy0 * np.sin(rot)) ** 2) / semi_len02
    z1 = ((xx0 * np.sin(rot) - yy0 * np.cos(rot)) ** 2) / semi_len12
    zz = z0 + z1
    elli_mask = zz <= 1.0
    return elli_mask


def _get_elliptical_ring(xx, yy, x, y, semi_len0, semi_len1, rotation, lw_r=1):
    """Private function to generate an elliptical ring.

    Parameters
    ----------
    xx, yy : 2D NumPy array
    x, y : float
    semi_len0, semi_len1 : float
    rotation : float
        In radians
    lw_r : optional, default 1

    Returns
    -------
    ellipse_array : 2D NumPy array

    Examples
    --------
    >>> from hyperspy.signals import Signal2D
    >>> from pyxem.dummy_data import make_diffraction_test_data as mdtd
    >>> s = Signal2D(np.zeros((110, 130)))
    >>> s.axes_manager[0].offset, s.axes_manager[1].offset = -50, -80
    >>> xx, yy = np.meshgrid(
    ...     s.axes_manager.signal_axes[0].axis,
    ...     s.axes_manager.signal_axes[1].axis)
    >>> ellipse_data = mdtd._get_elliptical_ring(
    ...     xx, yy, 10, -10, 12, 18, 1.5, 2)
    >>> s.data += ellipse_data
    >>> s.plot()

    """
    mask_outer = _get_elliptical_disk(
        xx, yy, x, y, semi_len0 + lw_r, semi_len1 + lw_r, rotation
    )
    mask_inner = _get_elliptical_disk(
        xx, yy, x, y, semi_len0 - lw_r, semi_len1 - lw_r, rotation
    )
    ellipse = np.logical_xor(mask_outer, mask_inner)
    ellipse = ellipse.astype("uint32")
    return ellipse


class EllipseRing:
    """Generate an elliptical ring.

    Parameters
    ----------
    xx, yy : int
        Size of the image
    x, y : float
        Centre positions for the ellipse
    semi_len0, semi_len1 : float
    rotation : float
        In radians
    lw_r : optional, default 1

    Methods
    -------
    get_signal
        Generate numpy array with the ellipse ring

    Examples
    --------
    >>> from pyxem.dummy_data import make_diffraction_test_data as mdtd
    >>> ellipse = mdtd.EllipseRing(
    ...     xx=20, yy=30, x0=10, y0=15, semi_len0=4, semi_len1=6,
    ...     rotation=1.57, intensity=10, lw_r=2)

    """

    def __init__(
        self, xx, yy, x0, y0, semi_len0, semi_len1, rotation, intensity, lw_r=1
    ):
        self.x0, self.y0 = x0, y0
        self.semi_len0 = semi_len0
        self.semi_len1 = semi_len1
        self.rotation = rotation
        self.intensity = intensity
        self.xx, self.yy = xx, yy
        self.lw_r = lw_r

    def __repr__(self):
        return (
            "<%s, ((x0, y0): (%s, %s), (sl0, sl1): (%s, %s),"
            " r: %s, I: %s, lw: %s)>"
            % (
                self.__class__.__name__,
                self.x0,
                self.y0,
                self.semi_len0,
                self.semi_len1,
                self.rotation,
                self.intensity,
                self.lw_r,
            )
        )

    def get_signal(self):
        ellipse = _get_elliptical_ring(
            xx=self.xx,
            yy=self.yy,
            x=self.x0,
            y=self.y0,
            semi_len0=self.semi_len0,
            semi_len1=self.semi_len1,
            rotation=self.rotation,
            lw_r=self.lw_r,
        )
        self.ellipse = ellipse * self.intensity
        return self.ellipse


class EllipseDisk:
    """Generate an elliptical disk.

    Parameters
    ----------
    xx, yy : int
        Size of the image
    x, y : float
        Centre positions for the ellipse
    semi_len0, semi_len1 : float
    rotation : float
        In radians
    lw_r : optional, default 1

    Methods
    -------
    get_signal
        Generate numpy array with the elliptical disk

    Examples
    --------
    >>> from pyxem.dummy_data import make_diffraction_test_data as mdtd
    >>> ellipse_disk = mdtd.EllipseDisk(
    ...     xx=20, yy=30, x0=10, y0=15, semi_len0=4, semi_len1=6,
    ...     rotation=1.57, intensity=10)

    """

    def __init__(self, xx, yy, x0, y0, semi_len0, semi_len1, rotation, intensity):
        self.x0, self.y0 = x0, y0
        self.semi_len0 = semi_len0
        self.semi_len1 = semi_len1
        self.rotation = rotation
        self.intensity = intensity
        self.xx, self.yy = xx, yy

    def __repr__(self):
        return "<%s, ((x0, y0): (%s, %s), (sl0, sl1): (%s, %s)," " r: %s, I: %s)>" % (
            self.__class__.__name__,
            self.x0,
            self.y0,
            self.semi_len0,
            self.semi_len1,
            self.rotation,
            self.intensity,
        )

    def get_signal(self):
        ellipse = _get_elliptical_disk(
            xx=self.xx,
            yy=self.yy,
            x=self.x0,
            y=self.y0,
            semi_len0=self.semi_len0,
            semi_len1=self.semi_len1,
            rotation=self.rotation,
        )
        self.ellipse = ellipse * self.intensity
        return self.ellipse


class Circle:
    def __init__(self, xx, yy, x0, y0, r, intensity, scale, lw=None):
        self.x0 = x0
        self.y0 = y0
        self.r = r
        self.intensity = intensity
        self.lw = lw
        self.circle = (yy - self.y0) ** 2 + (xx - self.x0) ** 2
        self.mask_outside_r(scale)
        self.get_centre_pixel(xx, yy, scale)

    def __repr__(self):
        return "<%s, (r: %s, (x0, y0): (%s, %s), I: %s)>" % (
            self.__class__.__name__,
            self.r,
            self.x0,
            self.y0,
            self.intensity,
        )

    def mask_outside_r(self, scale):
        if self.lw is None:
            indices = self.circle >= (self.r + scale) ** 2
        else:
            indices = self.circle >= (self.r + self.lw + scale) ** 2
        self.circle[indices] = 0

    def centre_on_image(self, xx, yy):
        if self.x0 < xx[0][0] or self.x0 > xx[0][-1]:
            return False
        elif self.y0 < yy[0][0] or self.y0 > yy[-1][-1]:
            return False
        else:
            return True

    def get_centre_pixel(self, xx, yy, scale):
        """Sets the indices for the pixels on which the centre point is.

        Because the centre point can sometimes be exactly on the
        boundary of two pixels, the pixels are held in a list. One
        list for x (self.centre_x_pixels) and one for y
        (self.centre_x_pixels). If the centre is outside the image, the
        lists will be empty.
        """
        if self.centre_on_image(xx, yy):
            x1 = np.where(xx > (self.x0 - 0.5 * scale))[1][0]
            x2 = np.where(xx < (self.x0 + 0.5 * scale))[1][-1]
            self.centre_x_pixels = [x1, x2]
            y1 = np.where(yy > (self.y0 - 0.5 * scale))[0][0]
            y2 = np.where(yy < (self.y0 + 0.5 * scale))[0][-1]
            self.centre_y_pixels = [y1, y2]
        else:
            self.centre_x_pixels = []
            self.centre_y_pixels = []

    def set_uniform_intensity(self):
        circle_ring_indices = self.circle > 0
        self.circle[circle_ring_indices] = self.intensity


class Disk:
    """Disk object, with outer edge of the ring at r."""

    def __init__(self, xx, yy, scale, x0, y0, r, intensity):
        self.z = Circle(xx, yy, x0, y0, r, intensity, scale)
        self.z.set_uniform_intensity()
        self.set_centre_intensity()

    def __repr__(self):
        return "<%s, (r: %s, (x0, y0): (%s, %s), I: %s)>" % (
            self.__class__.__name__,
            self.z.r,
            self.z.x0,
            self.z.y0,
            self.z.intensity,
        )

    def set_centre_intensity(self):
        """Sets the intensity of the centre pixels to I.

        Coordinates are self.z.circle[y, x], due to how numpy works.
        """
        for x in self.z.centre_x_pixels:
            for y in self.z.centre_y_pixels:
                self.z.circle[y, x] = self.z.intensity  # This is correct

    def get_signal(self):
        return self.z.circle


class Ring:
    """Ring object, with outer edge of the ring at r+lr, and inner r-lr.

    The radius of the ring is defined as in the middle of the line making
    up the ring.

    """

    def __init__(self, xx, yy, scale, x0, y0, r, intensity, lr):
        if lr > r:
            raise ValueError(f"Ring line width too big ({lr} > {r})")
        self.lr = lr
        self.lw = 1 + 2 * lr  # scalar line width of the ring
        self.z = Circle(xx, yy, x0, y0, r, intensity, scale, lw=lr)
        self.mask_inside_r(scale)
        self.z.set_uniform_intensity()

    def __repr__(self):
        return "<%s, (r: %s, (x0, y0): (%s, %s), I: %s)>" % (
            self.__class__.__name__,
            self.z.r,
            self.z.x0,
            self.z.y0,
            self.z.intensity,
        )

    def mask_inside_r(self, scale):
        indices = self.z.circle < (self.z.r - self.lr) ** 2
        self.z.circle[indices] = 0

    def get_signal(self):
        return self.z.circle


class MakeTestData:
    """MakeTestData is an object containing a generated test signal.

    The default signal consists of a Disk and concentric Ring, with the
    Ring being less intensive than the centre Disk. Unlimited number of
    Rings and Disk can be added separately.

    Parameters
    ----------
    size_x, size_y : float, int
        The range of the x and y axis goes from 0 to size_x, size_y
    scale : float, int
        The step size of the x and y axis
    default : bool, default False
        If True, the default object should be generated. If false, Ring and
        Disk must be added separately by self.add_ring(), self.add_disk()
    blur : bool, default True
        If True, do a Gaussian blur of the disk.
    blur_sigma : int, default 1
        Sigma of the Gaussian blurring, if blur is True.
    downscale : bool, default True
        Note: currently using downscale and adding a disk, will lead to the
        center of the disk being shifted. Ergo: the disk_x and disk_y will not
        be correct when downscale is True.

    Attributes
    ----------
    signal : hyperspy.signals.Signal2D
        Test signal
    z_list : list
        List containing Ring and Disk objects added to the signal
    downscale_factor : int
        The data is upscaled before Circle is added, and similarly
        downscaled to return to given dimensions. This improves the
        quality of Circle

    Examples
    --------
    Default settings

    >>> from pyxem.dummy_data.make_diffraction_test_data import MakeTestData
    >>> test_data = MakeTestData()
    >>> test_data.signal.plot()

    More control

    >>> test_data = MakeTestData(default=False)
    >>> test_data.add_disk(x0=50, y0=50, r=10, intensity=30)
    >>> test_data.add_ring(x0=45, y0=52, r=25, intensity=10)
    >>> test_data.signal.plot()

    """

    def __init__(
        self,
        size_x=100,
        size_y=100,
        scale=1,
        default=False,
        blur=True,
        blur_sigma=1,
        downscale=True,
    ):
        self.scale = scale
        self.blur_on = blur
        self.blur_sigma = blur_sigma
        self.downscale_on = downscale
        self.downscale_factor = 5
        if not self.downscale_on:
            self.downscale_factor = 1
        self.size_x, self.size_y = size_x, size_y
        self.generate_mesh()
        self.z_list = []
        if default:
            self.add_disk()
            self.add_ring(lw_pix=1)
        else:
            self.update_signal()

    def __repr__(self):
        return "<%s, ((x, y): (%s, %s), s: %s, z: %s)>" % (
            self.__class__.__name__,
            self.size_x,
            self.size_y,
            self.scale,
            len(self.z_list),
        )

    def update_signal(self):
        self.make_signal()
        self.downscale()
        self.blur()
        self.to_signal()

    def generate_mesh(self):
        self.X = np.arange(0, self.size_x, self.scale / self.downscale_factor)
        self.Y = np.arange(0, self.size_y, self.scale / self.downscale_factor)
        self.xx, self.yy = np.meshgrid(self.X, self.Y, sparse=True)

    def add_disk(self, x0=50, y0=50, r=5, intensity=10):
        scale = self.scale / self.downscale_factor
        self.z_list.append(Disk(self.xx, self.yy, scale, x0, y0, r, intensity))
        self.update_signal()

    def add_disk_ellipse(
        self, x0=50, y0=50, semi_len0=5, semi_len1=8, rotation=0.78, intensity=10
    ):
        ellipse = EllipseDisk(
            xx=self.xx,
            yy=self.yy,
            x0=x0,
            y0=y0,
            semi_len0=semi_len0,
            semi_len1=semi_len1,
            rotation=rotation,
            intensity=intensity,
        )
        self.z_list.append(ellipse)
        self.update_signal()

    def add_ring(self, x0=50, y0=50, r=20, intensity=10, lw_pix=0):
        """
        Add a ring to the test data.

        Parameters
        ----------
        x0, y0 : number, default 50
            Centre position of the ring
        r : number, default 20
            Radius of the ring, defined as the distance from the centre to the
            middle of the line of the ring, which will be most intense after
            blurring.
        intensity : number, default 10
            Pixel value of the ring. Note, this value will be lowered
            if blur or downscale is True
        lw_pix : number, default 0
            Distance in pixels from radius to the outer and inner edge of the
            ring. Inner radius: r-lw, outer radius: r+lw. In total this gives
            a ring line width in pixels of 2*lw+1.

        """
        scale = self.scale / self.downscale_factor
        lr = lw_pix * self.scale  # scalar
        self.z_list.append(Ring(self.xx, self.yy, scale, x0, y0, r, intensity, lr))
        self.update_signal()

    def add_ring_ellipse(
        self,
        x0=50,
        y0=50,
        semi_len0=11,
        semi_len1=13,
        rotation=0.78,
        intensity=10,
        lw_r=2,
    ):
        ellipse = EllipseRing(
            xx=self.xx,
            yy=self.yy,
            x0=x0,
            y0=y0,
            semi_len0=semi_len0,
            semi_len1=semi_len1,
            rotation=rotation,
            intensity=intensity,
            lw_r=lw_r,
        )
        self.z_list.append(ellipse)
        self.update_signal()

    def make_signal(self):
        if len(self.z_list) == 0:
            self.z = self.xx * 0 + self.yy * 0
        elif len(self.z_list) == 1:
            self.z = self.z_list[0].get_signal()
        elif len(self.z_list) > 1:
            z_temp = self.z_list[0].get_signal()
            for i in self.z_list[1:]:
                z_temp = np.add(z_temp, i.get_signal())
            self.z = z_temp

    def downscale(self):
        if self.downscale_on:
            shape = (
                int(self.z.shape[0] / self.downscale_factor),
                int(self.z.shape[1] / self.downscale_factor),
            )
            sh = (
                shape[0],
                self.z.shape[0] // shape[0],
                shape[1],
                self.z.shape[1] // shape[1],
            )
            self.z_downscaled = self.z.reshape(sh).mean(-1).mean(1)
        else:
            self.z_downscaled = self.z

    def blur(self):
        if self.blur_on:
            self.z_blurred = gaussian_filter(self.z_downscaled, sigma=self.blur_sigma)
        else:
            self.z_blurred = self.z_downscaled

    def to_signal(self):
        self.signal = Diffraction2D(self.z_blurred)
        self.signal.axes_manager[0].scale = self.scale
        self.signal.axes_manager[1].scale = self.scale

    def set_signal_zero(self):
        self.z_list = []
        self.update_signal()


def generate_4d_data(
    probe_size_x=10,
    probe_size_y=10,
    image_size_x=50,
    image_size_y=50,
    disk_x=25,
    disk_y=25,
    disk_r=5,
    disk_I=20,
    ring_x=25,
    ring_y=25,
    ring_r=20,
    ring_I=6,
    ring_lw=0,
    ring_e_x=None,
    ring_e_y=25,
    ring_e_semi_len0=15,
    ring_e_semi_len1=15,
    ring_e_r=0,
    ring_e_I=6,
    ring_e_lw=1,
    blur=True,
    blur_sigma=1,
    downscale=True,
    add_noise=False,
    noise_amplitude=1,
    lazy=False,
    lazy_chunks=None,
    show_progressbar=True,
):
    """Generate a test dataset containing a disk and diffraction ring.

    Useful for checking that radial average algorithms are working
    properly.

    The centre, intensity and radius position of the ring and disk can vary
    as a function of probe position, through the disk_x, disk_y, disk_r,
    disk_I, ring_x, ring_y, ring_r and ring_I arguments.
    In addition, the line width of the ring can be varied with ring_lw.

    There is also an elliptical ring, which can be added separately
    to the circular ring. This elliptical ring uses the ring_e_*
    arguments. It is disabled by default.

    The ring can be deactivated by setting ring_x=None.
    The disk can be deactivated by setting disk_x=None.
    The elliptical ring can be deactivated by setting ring_e_x=None.

    Parameters
    ----------
    probe_size_x, probe_size_y : int, default 10
        Size of the navigation dimension.
    image_size_x, image_size_y : int, default 50
        Size of the signal dimension.
    disk_x, disk_y : int or NumPy 2D-array, default 20
        Centre position of the disk. Either integer or NumPy 2-D array.
        See examples on how to make them the correct size.
        To deactivate the disk, set disk_x=None.
    disk_r : int or NumPy 2D-array, default 5
        Radius of the disk. Either integer or NumPy 2-D array.
        See examples on how to make it the correct size.
    disk_I : int or NumPy 2D-array, default 20
        Intensity of the disk, for each of the pixels.
        So if I=30, the each pixel in the disk will have a value of 30.
        Note, this value will change if blur=True or downscale=True.
    ring_x, ring_y : int or NumPy 2D-array, default 20
        Centre position of the ring. Either integer or NumPy 2-D array.
        See examples on how to make them the correct size.
        To deactivate the ring, set ring_x=None.
    ring_r : int or NumPy 2D-array, default 20
        Radius of the ring. Either integer or NumPy 2-D array.
        See examples on how to make it the correct size.
    ring_I : int or NumPy 2D-array, default 6
        Intensity of the ring, for each of the pixels.
        So if I=5, each pixel in the ring will have a value of 5.
        Note, this value will change if blur=True or downscale=True.
    ring_lw : int or NumPy 2D-array, default 0
        Line width of the ring. If ring_lw=1, the line will be 3 pixels wide.
        If ring_lw=2, the line will be 5 pixels wide.
    ring_e_x, ring_e_y : int or NumPy 2D-array, default 20
        Centre position of the elliptical ring. Either integer or
        NumPy 2-D array. See examples on how to make them the correct size.
        To deactivate the ring, set ring_x=None (which is the default).
    ring_e_semi_len0, ring_e_semi_len1 : int or NumPy 2D-array, default 15
        Semi lengths of the elliptical ring. Either integer or NumPy 2-D
        arrays. See examples on how to make it the correct size.
    ring_e_I : int or NumPy 2D-array, default 6
        Intensity of the elliptical ring, for each of the pixels.
        So if I=5, each pixel in the ring will have a value of 5.
        Note, this value will change if blur=True or downscale=True.
    ring_e_r : int or NumPy 2D-array, default 0
        Rotation of the elliptical ring, in radians.
    ring_e_lw : int or NumPy 2D-array, default 0
        Line width of the ring. If ring_lw=1, the line will be 3 pixels wide.
        If ring_lw=2, the line will be 5 pixels wide.
    blur : bool, default True
        If True, do a Gaussian blur of the disk.
    blur_sigma : int, default 1
        Sigma of the Gaussian blurring, if blur is True.
    downscale : bool, default True
        If True, use upscaling (then downscaling) to anti-alise the disk.
    add_noise : bool, default False
        Add Gaussian random noise.
    noise_amplitude : float, default 1
        The amplitude of the noise, if add_noise is True.
    lazy : bool, default False
        If True, the signal will be lazy
    lazy_chunks : tuple, optional
        Used if lazy is True, default (10, 10, 10, 10).

    Returns
    -------
    signal : HyperSpy Signal2D
        Signal with 2 navigation dimensions and 2 signal dimensions.

    Examples
    --------
    >>> from pyxem.dummy_data import make_diffraction_test_data as mdtd
    >>> s = mdtd.generate_4d_data(show_progressbar=False)
    >>> s.plot()

    Using more arguments

    >>> s = mdtd.generate_4d_data(probe_size_x=20, probe_size_y=30,
    ...         image_size_x=50, image_size_y=90,
    ...         disk_x=30, disk_y=70, disk_r=9, disk_I=30,
    ...         ring_x=35, ring_y=65, ring_r=20, ring_I=10,
    ...         blur=False, downscale=False, show_progressbar=False)

    Adding some Gaussian random noise

    >>> s = mdtd.generate_4d_data(add_noise=True, noise_amplitude=3,
    ...         show_progressbar=False)

    Different centre positions for each probe position.
    Note the size=(20, 10), and probe_x=10, probe_y=20: size=(y, x).

    >>> import numpy as np
    >>> disk_x = np.random.randint(5, 35, size=(20, 10))
    >>> disk_y = np.random.randint(5, 45, size=(20, 10))
    >>> disk_I = np.random.randint(50, 100, size=(20, 10))
    >>> ring_x = np.random.randint(5, 35, size=(20, 10))
    >>> ring_y = np.random.randint(5, 45, size=(20, 10))
    >>> ring_r = np.random.randint(10, 15, size=(20, 10))
    >>> ring_I = np.random.randint(1, 30, size=(20, 10))
    >>> ring_lw = np.random.randint(1, 5, size=(20, 10))
    >>> s = mdtd.generate_4d_data(probe_size_x=10, probe_size_y=20,
    ...         image_size_x=40, image_size_y=50, disk_x=disk_x, disk_y=disk_y,
    ...         disk_I=disk_I, ring_x=ring_x, ring_y=ring_y, ring_r=ring_r,
    ...         ring_I=ring_I, ring_lw=ring_lw, show_progressbar=False)

    Do not plot the disk

    >>> s = mdtd.generate_4d_data(disk_x=None, show_progressbar=False)

    Do not plot the ring

    >>> s = mdtd.generate_4d_data(ring_x=None, show_progressbar=False)

    Plot only an elliptical ring

    >>> from numpy.random import randint, random
    >>> s = mdtd.generate_4d_data(
    ...        probe_size_x=10, probe_size_y=10,
    ...        disk_x=None, ring_x=None,
    ...        ring_e_x=randint(20, 30, (10, 10)),
    ...        ring_e_y=randint(20, 30, (10, 10)),
    ...        ring_e_semi_len0=randint(10, 20, (10, 10)),
    ...        ring_e_semi_len1=randint(10, 20, (10, 10)),
    ...        ring_e_r=random((10, 10))*np.pi,
    ...        ring_e_lw=randint(1, 3, (10, 10)))

    """
    if disk_x is None:
        plot_disk = False
    else:
        plot_disk = True
        if not isiterable(disk_x):
            disk_x = np.ones((probe_size_y, probe_size_x)) * disk_x
    if not isiterable(disk_y):
        disk_y = np.ones((probe_size_y, probe_size_x)) * disk_y
    if not isiterable(disk_r):
        disk_r = np.ones((probe_size_y, probe_size_x)) * disk_r
    if not isiterable(disk_I):
        disk_I = np.ones((probe_size_y, probe_size_x)) * disk_I

    if ring_x is None:
        plot_ring = False
    else:
        plot_ring = True
        if not isiterable(ring_x):
            ring_x = np.ones((probe_size_y, probe_size_x)) * ring_x
    if not isiterable(ring_y):
        ring_y = np.ones((probe_size_y, probe_size_x)) * ring_y
    if not isiterable(ring_r):
        ring_r = np.ones((probe_size_y, probe_size_x)) * ring_r
    if not isiterable(ring_I):
        ring_I = np.ones((probe_size_y, probe_size_x)) * ring_I
    if not isiterable(ring_lw):
        ring_lw = np.ones((probe_size_y, probe_size_x)) * ring_lw

    if ring_e_x is None:
        plot_ring_e = False
    else:
        plot_ring_e = True
        if not isiterable(ring_e_x):
            ring_e_x = np.ones((probe_size_y, probe_size_x)) * ring_e_x
    if not isiterable(ring_e_y):
        ring_e_y = np.ones((probe_size_y, probe_size_x)) * ring_e_y
    if not isiterable(ring_e_semi_len0):
        ring_e_semi_len0 = np.ones((probe_size_y, probe_size_x)) * ring_e_semi_len0
    if not isiterable(ring_e_semi_len1):
        ring_e_semi_len1 = np.ones((probe_size_y, probe_size_x)) * ring_e_semi_len1
    if not isiterable(ring_e_I):
        ring_e_I = np.ones((probe_size_y, probe_size_x)) * ring_e_I
    if not isiterable(ring_e_lw):
        ring_e_lw = np.ones((probe_size_y, probe_size_x)) * ring_e_lw
    if not isiterable(ring_e_r):
        ring_e_r = np.ones((probe_size_y, probe_size_x)) * ring_e_r

    signal_shape = (probe_size_y, probe_size_x, image_size_y, image_size_x)
    s = Diffraction2D(np.zeros(shape=signal_shape))
    for i in tqdm(s, desc="Make test data", disable=not show_progressbar):
        index = s.axes_manager.indices[::-1]
        test_data = MakeTestData(
            size_x=image_size_x,
            size_y=image_size_y,
            default=False,
            blur=blur,
            blur_sigma=blur_sigma,
            downscale=downscale,
        )
        if plot_disk:
            dx, dy, dr = disk_x[index], disk_y[index], disk_r[index]
            dI = disk_I[index]
            test_data.add_disk(dx, dy, dr, intensity=dI)
        if plot_ring:
            rx, ry, rr = ring_x[index], ring_y[index], ring_r[index]
            rI, rLW = ring_I[index], ring_lw[index]
            test_data.add_ring(rx, ry, rr, intensity=rI, lw_pix=rLW)
        if plot_ring_e:
            rex, rey = ring_e_x[index], ring_e_y[index]
            resl0, resl1 = ring_e_semi_len0[index], ring_e_semi_len1[index]
            reI, reLW, rer = ring_e_I[index], ring_e_lw[index], ring_e_r[index]
            test_data.add_ring_ellipse(
                x0=rex,
                y0=rey,
                semi_len0=resl0,
                semi_len1=resl1,
                rotation=rer,
                intensity=reI,
                lw_r=reLW,
            )
        s.data[index][:] = test_data.signal.data[:]
        if add_noise:
            s.data[index][:] += (
                np.random.random(size=(image_size_y, image_size_x)) * noise_amplitude
            )
    s.axes_manager.indices = [0] * s.axes_manager.navigation_dimension
    if lazy:
        if lazy_chunks is None:
            lazy_chunks = 10, 10, 10, 10
        data_lazy = da.from_array(s.data, lazy_chunks)
        s = LazyDiffraction2D(data_lazy)
    return s


def _make_4d_peak_array_test_data(xf, yf, semi0, semi1, rot, nt=20):
    """Get a 4D NumPy array with peak_array test data.

    Parameters
    ----------
    xf, yf : scalar, 2D NumPy array
        Centre position of the ellipse. The size of the xf array gives the
        size of the peak_array.
    semi0, semi1 : scalar, 2D NumPy array
        Semi length of the ellipse, of rot is 0 semi0 is the x-direction
        semi length, and semi1 the y-direction.
    rot : scalar, 2D NumPy array
        Rotation in radians.
    nt : scalar
        Number of points in the ellipse.

    Returns
    -------
    peak_array : NumPy 4D array

    Examples
    --------
    >>> import pyxem as pxm
    >>> from pyxem.dummy_data import make_diffraction_test_data as mdtd
    >>> xf = np.random.randint(65, 70, size=(4, 5))
    >>> yf = np.random.randint(115, 120, size=(4, 5))
    >>> semi0 = np.random.randint(35, 40, size=(4, 5))
    >>> semi1 = np.random.randint(45, 50, size=(4, 5))
    >>> rot = np.random.random(size=(4, 5)) * 0.2
    >>> peak_array = mdtd._make_4d_peak_array_test_data(
    ...        xf, yf, semi0, semi1, rot)
    >>> s = pxm.signals.Diffraction2D(np.zeros(shape=(4, 5, 200, 210)))
    >>> import pyxem.utils.marker_tools as mt
    >>> mt.add_peak_array_to_signal_as_markers(s, peak_array)

    """
    peak_array = np.empty_like(xf, dtype=object)
    for iy, ix in np.ndindex(peak_array.shape):
        params = (xf[iy, ix], yf[iy, ix], semi0[iy, ix], semi1[iy, ix], rot[iy, ix], nt)
        ellipse_points = ret.make_ellipse_data_points(*params)
        peak_array[iy, ix] = np.fliplr(ellipse_points)
    return peak_array


class DiffractionTestImage:
    def __init__(
        self,
        disk_r=7,
        blur=2,
        image_x=256,
        image_y=256,
        rotation=0,
        diff_intensity_reduction=1.0,
        intensity_noise=0.5,
    ):
        """Make an artificial diffraction image, similar to NBED data.

        This class is for creating images which are similar to
        nanobeam electron diffraction patterns, with functionality for
        adding diffraction disks, Lorentzian background and intensity noise.

        Can be combined with the DiffractionTestDataset class, for creating
        4-dimensional datasets. Similar to the ones acquired with
        fast pixelated direct electron detectors in
        scanning transmission electron microscopy.

        Parameters
        ----------
        disk_r : int
            Radius for all of the disks added. Default 7
        blur : scalar
            Amount of Gaussian blur applied to the image.
            Use False to disable blurring. Default 2.
        image_x, image_y : int
            Dimensions for the image. Default 256 for both.
        rotation : int
            Rotation of the image in relation to the centre point.
            For an image with the size (256, 256), the rotation will
            be around (128, 128).
        diff_intensity_reduction : scalar
            In diffraction patterns, the intensity of the disk are
            reduced with increasing scattering angle. This parameter
            emulates this, with decreasing the intensity of the disks
            as a function of distance from the centre point. Default 1.0.
            Set False to disable.
        intensity_noise : scalar
            The width of the Gaussian intensity noise added to the image.
            Set False to disable. Default 0.5.


        Examples
        --------
        >>> from pyxem.dummy_data import make_diffraction_test_data as mdtd
        >>> di = mdtd.DiffractionTestImage()
        >>> di.add_disk(x=128, y=128, intensity=10.)
        >>> di.add_cubic_disks(vx=20, vy=20, intensity=2., n=5)
        >>> di.add_background_lorentz()
        >>> s = di.get_signal()
        >>> s.plot()

        Get a slightly rotated version of the diffraction image

        >>> di.rotation = 10
        >>> s = di.get_signal()
        >>> s.plot()

        """
        self.disk_r = disk_r
        self.blur = blur
        self.image_x = image_x
        self.image_y = image_y
        self.rotation = rotation
        self.diff_intensity_reduction = diff_intensity_reduction
        self.intensity_noise = intensity_noise
        self._background_lorentz_width = False
        self._background_lorentz_intensity = None
        self._x_list = []
        self._y_list = []
        self._intensity_list = []

    def __repr__(self):
        return "<%s, disks:%s, r:%s, rot:%s, im:(%s,%s)>" % (
            self.__class__.__name__,
            len(self._x_list),
            self.disk_r,
            self.rotation,
            self.image_x,
            self.image_y,
        )

    def __copy__(self):
        d = DiffractionTestImage(
            disk_r=self.disk_r,
            blur=self.blur,
            image_x=self.image_x,
            image_y=self.image_y,
            rotation=self.rotation,
            diff_intensity_reduction=self.diff_intensity_reduction,
            intensity_noise=self.intensity_noise,
        )
        d._background_lorentz_width = self._background_lorentz_width
        d._background_lorentz_intensity = self._background_lorentz_intensity
        d._x_list = self._x_list.copy()
        d._y_list = self._y_list.copy()
        d._intensity_list = self._intensity_list.copy()
        return d

    def copy(self):
        return self.__copy__()

    def add_disk(self, x, y, intensity=1):
        if not isinstance(x, int):
            raise ValueError("x needs to be integer, not {0}".format(x))
        if not isinstance(y, int):
            raise ValueError("y needs to be integer, not {0}".format(y))
        self._x_list.append(x)
        self._y_list.append(y)
        self._intensity_list.append(intensity)

    def add_background_lorentz(self, width=10, intensity=5):
        self._background_lorentz_width = width
        self._background_lorentz_intensity = intensity

    def _get_diff_intensity_reduction(self, dr, i):
        r_max = np.hypot(self.image_x / 2, self.image_y / 2)
        if dr == 0.0:
            dr = 0.00000000001
        x = np.pi * dr * 1.5 / (r_max * self.diff_intensity_reduction)
        i_new = np.sin(x) / x * i
        if i_new < 0:
            i_new = 0
        return i_new

    def add_cubic_disks(self, vx, vy, intensity=1, n=1):
        """Add disks in a cubic pattern around the centre point of the image.

        Parameters
        ----------
        vx, vy : int
        intensity : scalar
        n : int
            Number of orders of diffraction disks.
            If n=1, 8 disks, if n=2, 24 disks.
        """
        cx, cy = self.image_x / 2, self.image_y / 2
        for px in range(-n, n + 1):
            for py in range(-n, n + 1):
                if not (px == 0 and py == 0):
                    x = int(round(vx * px + cx))
                    y = int(round(vy * py + cy))
                    self.add_disk(x, y, intensity=intensity)

    def _get_background_lorentz(self):
        width = self._background_lorentz_width
        intensity = self._background_lorentz_intensity
        x = np.linspace(-width, width, self.image_y)
        y = np.linspace(-width, width, self.image_x)
        YY, XX = np.meshgrid(y, x)
        YY = YY.astype(np.float32)
        XX = XX.astype(np.float32)
        b = 1 / (np.pi * (1 + np.hypot(YY, XX) ** 2)) * intensity
        return b

    def get_diffraction_test_image(self, dtype=np.float32):
        image_x, image_y = self.image_x, self.image_y
        cx, cy = image_x / 2, image_y / 2
        image = np.zeros((image_y, image_x), dtype=np.float32)
        iterator = zip(self._x_list, self._y_list, self._intensity_list)
        for x, y, i in iterator:
            if self.diff_intensity_reduction is not False:
                dr = np.hypot(x - cx, y - cy)
                i = self._get_diff_intensity_reduction(dr, i)
            image[y, x] = i
        disk = morphology.disk(self.disk_r, dtype=dtype)
        image = convolve2d(image, disk, mode="same")
        if self.rotation != 0:
            image = rotate(image, self.rotation, reshape=False)
        if self.blur != 0:
            image = gaussian_filter(image, self.blur)
        if self._background_lorentz_width is not False:
            image += self._get_background_lorentz()
        if self.intensity_noise is not False:
            noise = np.random.random((image_y, image_x)) * self.intensity_noise
            image += noise
        return image

    def get_signal(self):
        s = Diffraction2D(self.get_diffraction_test_image())
        return s

    def plot(self):
        s = self.get_signal()
        s.plot()


class DiffractionTestDataset:
    def __init__(
        self,
        probe_x=10,
        probe_y=10,
        detector_x=256,
        detector_y=256,
        noise=0.5,
        dtype=np.float32,
    ):
        """Make a 4-dimensional dataset similar to NBED.

        This class is for creating datasets which are similar to
        nanobeam electron diffraction patterns. It is used in combination
        with one or several DiffractionTestImage objects.

        Parameters
        ----------
        probe_x, probe_y : int
        detector_x, detector_y : int
        noise : scalar

        Examples
        --------
        >>> from pyxem.dummy_data import make_diffraction_test_data as mdtd
        >>> di = mdtd.DiffractionTestImage(intensity_noise=False)
        >>> di.add_disk(x=128, y=128, intensity=10.)
        >>> di.add_cubic_disks(vx=20, vy=20, intensity=2., n=5)
        >>> di.add_background_lorentz()
        >>> di_rot = di.copy()
        >>> di_rot.rotation = 10
        >>> dtd = mdtd.DiffractionTestDataset(10, 10, 256, 256)
        >>> position_array = np.ones((10, 10), dtype=bool)
        >>> position_array[:5] = False
        >>> dtd.add_diffraction_image(di, position_array)
        >>> dtd.add_diffraction_image(di_rot, np.invert(position_array))
        >>> s = dtd.get_signal()

        """
        self.data = np.zeros((probe_x, probe_y, detector_x, detector_y), dtype=dtype)
        self.probe_x = probe_x
        self.probe_y = probe_y
        self.detector_x = detector_x
        self.detector_y = detector_y
        self.noise = noise

    def __repr__(self):
        return "<%s, (%s)>" % (self.__class__.__name__, self.data.shape)

    def add_diffraction_image(self, diffraction_test_image, position_array=None):
        """Add a diffraction image to all or a subset of the dataset.

        See the class docstring for example on how to use this.

        Parameters
        ----------
        diffraction_test_image : DiffractionTestData
        position_array : numpy.ndarray
            Boolean array, specifying which positions in the dataset
            the diffraction_test_image should be added to. Must have two
            dimensions, and the same shape as (probe_x, probe_y).

        """
        probe_x, probe_y = self.probe_x, self.probe_y
        detector_x, detector_y = self.detector_x, self.detector_y
        image = diffraction_test_image.get_diffraction_test_image()
        if position_array is None:
            position_array = np.ones((probe_x, probe_y), dtype=bool)
        for ix, iy in np.ndindex(probe_x, probe_y):
            if position_array[ix, iy]:
                self.data[ix, iy, :, :] = image
                if self.noise is not False:
                    image_noise = np.random.random((detector_x, detector_y))
                    self.data[ix, iy, :, :] += image_noise * self.noise

    def get_signal(self):
        s = Diffraction2D(self.data)
        return s

    def plot(self):
        s = self.get_signal()
        s.plot()
