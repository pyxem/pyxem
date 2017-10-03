import numpy as np
from hyperspy.signals import Signal2D
from hyperspy.misc.utils import isiterable
from scipy.ndimage.filters import gaussian_filter
from fpd_data_processing.pixelated_stem_class import PixelatedSTEM


class Circle(object):
    def __init__(self, xx, yy, x0, y0, r, I, scale, lw=None):
        self.x0 = x0
        self.y0 = y0
        self.r = r
        self.I = I
        self.lw = lw
        self.circle = (yy - self.y0) ** 2 + (xx - self.x0) ** 2
        self.mask_outside_r(scale)
        self.get_centre_pixel(xx, yy, scale)

    def __repr__(self):
        return '<%s, (r: %s, (x0, y0): (%s, %s), I: %s)>' % (
            self.__class__.__name__,
            self.r, self.x0, self.y0, self.I,
            )

    def mask_outside_r(self, scale):
        if self.lw is None:
            indices = self.circle >= (self.r + scale)**2
        else:
            indices = self.circle >= (self.r + self.lw + scale)**2
        self.circle[indices] = 0

    def centre_on_image(self, xx, yy):
        if self.x0 < xx[0][0] or self.x0 > xx[0][-1]:
            return(False)
        elif self.y0 < yy[0][0] or self.y0 > yy[-1][-1]:
            return(False)
        else:
            return(True)

    def get_centre_pixel(self, xx, yy, scale):
        """
        This function sets the indices for the pixels on which the centre
        point is. Because the centre point can sometimes be exactly on the
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
        self.circle[circle_ring_indices] = self.I

    def update_axis(self, xx, yy):
        self.circle = (xx - self.x0) ** 2 + (yy - self.y0) ** 2
        self.mask_outside_r()


class Disk(object):
    """
    Disk object, with outer edge of the ring at r
    """
    def __init__(self, xx, yy, scale, x0, y0, r, I):
        self.z = Circle(xx, yy, x0, y0, r, I, scale)
        self.z.set_uniform_intensity()
        self.set_centre_intensity()

    def __repr__(self):
        return '<%s, (r: %s, (x0, y0): (%s, %s), I: %s)>' % (
            self.__class__.__name__,
            self.z.r,
            self.z.x0,
            self.z.y0,
            self.z.I,
            )

    def set_centre_intensity(self):
        """
        Sets the intensity of the centre pixels to I. Coordinates are
        self.z.circle[y, x], due to how numpy works.
        """
        for x in self.z.centre_x_pixels:
            for y in self.z.centre_y_pixels:
                self.z.circle[y, x] = self.z.I  # This is correct

    def get_signal(self):
        return(self.z.circle)

    def update_axis(self, xx, yy):
        self.z.update_axis(xx, yy)
        self.z.set_uniform_intensity()
        self.set_centre_intensity()


class Ring(object):
    """
    Ring object, with outer edge of the ring at r+lr, and inner r-lr.
    The radius of the ring is defined as in the middle of the line making
    up the ring.
    """
    def __init__(self, xx, yy, scale, x0, y0, r, I, lr):
        if lr > r:
            raise ValueError('Ring line width too big'.format(lr, r))
        self.lr = lr
        self.lw = 1 + 2*lr  # scalar line width of the ring
        self.z = Circle(xx, yy, x0, y0, r, I, scale, lw=lr)
        self.mask_inside_r(scale)
        self.z.set_uniform_intensity()

    def __repr__(self):
        return '<%s, (r: %s, (x0, y0): (%s, %s), I: %s)>' % (
            self.__class__.__name__, self.z.r, self.z.x0, self.z.y0, self.z.I,
            )

    def mask_inside_r(self, scale):
        indices = self.z.circle < (self.z.r - self.lr)**2
        self.z.circle[indices] = 0

    def get_signal(self):
        return(self.z.circle)

    def update_axis(self, xx, yy):
        self.z.update_axis(xx, yy)
        self.mask_inside_r()
        self.z.set_uniform_intensity()


class TestData:
    """
    TestData is an object containing a generated test signal. The default
    signal is consisting of a Disk and concentric Ring, with the Ring being
    less intensive than the centre Disk. Unlimited number of Rings and Disks
    can be added separately.

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

    >>> from fpd_data_processing.make_diffraction_test_data import TestData
    >>> test_data = TestData()
    >>> test_data.signal.plot()

    More control

    >>> test_data = TestData(default=False)
    >>> test_data.add_disk(x0=50, y0=50, r=10, I=30)
    >>> test_data.add_ring(x0=45, y0=52, r=25, I=10)
    >>> test_data.signal.plot()
    """
    def __init__(
            self, size_x=100, size_y=100, scale=1,
            default=False, blur=True, blur_sigma=1, downscale=True):
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
        return '<%s, ((x, y): (%s, %s), s: %s, z: %s)>' % (
            self.__class__.__name__,
            self.size_x, self.size_y,
            self.scale, len(self.z_list),
            )

    def update_signal(self):
        self.make_signal()
        self.downscale()
        self.blur()
        self.to_signal()

    def generate_mesh(self):
        self.X = np.arange(0, self.size_x, self.scale/self.downscale_factor)
        self.Y = np.arange(0, self.size_y, self.scale/self.downscale_factor)
        self.xx, self.yy = np.meshgrid(self.X, self.Y, sparse=True)

    def add_disk(self, x0=50, y0=50, r=5, I=10):
        scale = self.scale/self.downscale_factor
        self.z_list.append(Disk(self.xx, self.yy, scale, x0, y0, r, I))
        self.update_signal()

    def add_ring(self, x0=50, y0=50, r=20, I=10, lw_pix=0):
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
        I : number, default 10
            Pixel value of the ring. Note, this value will be lowered
            if blur or downscale is True
        lw_pix : number, default 0
            Distance in pixels from radius to the outer and inner edge of the
            ring. Inner radius: r-lw, outer radius: r+lw. In total this gives
            a ring line width in pixels of 2*lw+1.

        """
        scale = self.scale/self.downscale_factor
        lr = lw_pix*self.scale  # scalar
        self.z_list.append(
                Ring(self.xx, self.yy, scale, x0, y0, r, I, lr))
        self.update_signal()

    def make_signal(self):
        if len(self.z_list) == 0:
            self.z = self.xx*0 + self.yy*0
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
                    int(self.z.shape[0]/self.downscale_factor),
                    int(self.z.shape[1]/self.downscale_factor))
            sh = (
                    shape[0],
                    self.z.shape[0]//shape[0], shape[1],
                    self.z.shape[1]//shape[1])
            self.z_downscaled = self.z.reshape(sh).mean(-1).mean(1)
        else:
            self.z_downscaled = self.z

    def blur(self):
        if self.blur_on:
            self.z_blurred = gaussian_filter(
                    self.z_downscaled, sigma=self.blur_sigma)
        else:
            self.z_blurred = self.z_downscaled

    def to_signal(self):
        self.signal = PixelatedSTEM(self.z_blurred)
        self.signal.axes_manager[0].scale = self.scale
        self.signal.axes_manager[1].scale = self.scale

    def set_downscale_factor(self, factor):
            self.downscale_factor = factor
            self.generate_mesh()
            for i in self.z_list:
                i.update_axis(self.xx, self.yy)
            self.update_signal()

    def set_signal_zero(self):
        self.z_list = []
        self.update_signal()


def generate_4d_disk_data(
        probe_size_x=10, probe_size_y=10, image_size_x=50, image_size_y=50,
        disk_x=20, disk_y=20, disk_r=5, I=20, blur=True, blur_sigma=1,
        downscale=True, add_noise=False, noise_amplitude=1,
        ):
    """
    Generate a test dataset containing a single disk.
    Useful for testing if disk-center-position-finding
    algorithms are working properly.

    The centre and radius position of the disk can vary as a function
    of probe position, through the disk_x, disk_y and disk_r arguments.

    Parameters
    ----------
    probe_size_x, probe_size_y : int, default 10
        Size of the navigation dimension.
    image_size_x, image_size_y : int, default 50
        Size of the signal dimension.
    disk_x, disk_y : int or NumPy 2D-array, default 20
        Centre position of the disk. Either integer or Numpy 2-D array.
        See examples on how to make them the correct size.
    disk_r : int or NumPy 2D-array, default 5
        Radius of the disk. Either integer or NumPy 2-D array.
        See examples on how to make it the correct size.
    I : int, default 20
        Intensity of the disk, for each of the pixels.
        So if I=30, the each pixel in the disk will have a value of 30.
        Note, this value will change if blur=True or downscale=True.
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

    Returns
    -------
    signal : HyperSpy Signal2D
        Signal with 2 navigation dimensions and 2 signal dimensions.

    Examples
    --------
    >>> import fpd_data_processing.make_diffraction_test_data as mdtd
    >>> s = mdtd.generate_4d_disk_data()
    >>> s.plot()

    Using more arguments
    >>> s = mdtd.generate_4d_disk_data(probe_size_x=20, probe_size_y=30,
    ...         image_size_x=50, image_size_y=90, disk_x=30, disk_y=70,
    ...         disk_r=9, I=30, blur=False, downscale=False)

    Adding some Gaussian random noise
    >>> s = mdtd.generate_4d_disk_data(add_noise=True, noise_amplitude=3)

    Different centre positions for each probe position.
    Note the size=(20, 10), and probe_x=10, probe_y=20: size=(y, x).
    >>> import numpy as np
    >>> disk_x = np.random.randint(5, 35, size=(20, 10))
    >>> disk_y = np.random.randint(5, 45, size=(20, 10))
    >>> s = mdtd.generate_4d_disk_data(probe_size_x=10, probe_size_y=20,
    ...         image_size_x=40, image_size_y=50, disk_x=disk_x, disk_y=disk_y)
    """

    if not isiterable(disk_x):
        disk_x = np.ones((probe_size_y, probe_size_x))*disk_x
    if not isiterable(disk_y):
        disk_y = np.ones((probe_size_y, probe_size_x))*disk_y
    if not isiterable(disk_r):
        disk_r = np.ones((probe_size_y, probe_size_x))*disk_r
    disk_x, disk_y = np.array(disk_x), np.array(disk_y)
    disk_r = np.array(disk_r)

    signal_shape = (probe_size_y, probe_size_x, image_size_y, image_size_x)
    s = PixelatedSTEM(np.zeros(shape=signal_shape))
    for i in s:
        index = s.axes_manager.indices[::-1]
        test_data = TestData(
                size_x=image_size_x, size_y=image_size_y,
                default=False, blur=blur, blur_sigma=blur_sigma,
                downscale=downscale)
        dx, dy, dr = disk_x[index], disk_y[index], disk_r[index]
        test_data.add_disk(dx, dy, dr, I=I)
        s.data[index][:] = test_data.signal.data[:]
        if add_noise:
            s.data[index][:] += np.random.random(
                    size=(image_size_y, image_size_x)) * noise_amplitude
    return s


def generate_4d_holz_data(
        probe_size_x=10, probe_size_y=10, image_size_x=50, image_size_y=50,
        disk_x=25, disk_y=25, disk_r=5, disk_I=20,
        ring_x=25, ring_y=25, ring_r=20, ring_I=6, ring_lw=0,
        blur=True, blur_sigma=1, downscale=True, add_noise=False,
        noise_amplitude=1,
        ):
    """
    Generate a test dataset containing a disk and diffraction ring.
    Useful for checking that radial integration
    algorithms are working properly.

    The centre and radius position of the ring and  disk can vary as a function
    of probe position, through the disk_x, disk_y and disk_r arguments.

    Parameters
    ----------
    probe_size_x, probe_size_y : int, default 10
        Size of the navigation dimension.
    image_size_x, image_size_y : int, default 50
        Size of the signal dimension.
    disk_x, disk_y : int or NumPy 2D-array, default 20
        Centre position of the disk. Either integer or Numpy 2-D array.
        See examples on how to make them the correct size.
    disk_r : int or NumPy 2D-array, default 5
        Radius of the disk. Either integer or NumPy 2-D array.
        See examples on how to make it the correct size.
    disk_I : int, default 20
        Intensity of the disk, for each of the pixels.
        So if I=30, the each pixel in the disk will have a value of 30.
        Note, this value will change if blur=True or downscale=True.
    ring_x, ring_y : int or NumPy 2D-array, default 20
        Centre position of the ring. Either integer or NumPy 2-D array.
        See examples on how to make them the correct size.
    ring_r : int or NumPy 2D-array, default 5
        Radius of the ring. Either integer or NumPy 2-D array.
        See examples on how to make it the correct size.
    ring_I : int, default 6
        Intensity of the ring, for each of the pixels.
        So if I=5, each pixel in the ring will have a value of 5.
        Note, this value will change if blur=True or downscale=True.
    ring_lw : int, default 0
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

    Returns
    -------
    signal : HyperSpy Signal2D
        Signal with 2 navigation dimensions and 2 signal dimensions.

    Examples
    --------
    >>> import fpd_data_processing.make_diffraction_test_data as mdtd
    >>> s = mdtd.generate_4d_holz_data()
    >>> s.plot()

    Using more arguments
    >>> s = mdtd.generate_4d_holz_data(probe_size_x=20, probe_size_y=30,
    ...         image_size_x=50, image_size_y=90,
    ...         disk_x=30, disk_y=70, disk_r=9, disk_I=30,
    ...         ring_x=35, ring_y=65, ring_r=20, ring_I=10,
    ...         blur=False, downscale=False)

    Adding some Gaussian random noise
    >>> s = mdtd.generate_4d_holz_data(add_noise=True, noise_amplitude=3)

    Different centre positions for each probe position.
    Note the size=(20, 10), and probe_x=10, probe_y=20: size=(y, x).
    >>> import numpy as np
    >>> disk_x = np.random.randint(5, 35, size=(20, 10))
    >>> disk_y = np.random.randint(5, 45, size=(20, 10))
    >>> ring_x = np.random.randint(5, 35, size=(20, 10))
    >>> ring_y = np.random.randint(5, 45, size=(20, 10))
    >>> ring_r = np.random.randint(10, 15, size=(20, 10))
    >>> s = mdtd.generate_4d_holz_data(probe_size_x=10, probe_size_y=20,
    ...         image_size_x=40, image_size_y=50, disk_x=disk_x, disk_y=disk_y,
    ...         ring_x=ring_x, ring_y=ring_y, ring_r=ring_r, ring_lw=1)
    """

    if not isiterable(disk_x):
        disk_x = np.ones((probe_size_y, probe_size_x))*disk_x
    if not isiterable(disk_y):
        disk_y = np.ones((probe_size_y, probe_size_x))*disk_y
    if not isiterable(disk_r):
        disk_r = np.ones((probe_size_y, probe_size_x))*disk_r
    if not isiterable(ring_x):
        ring_x = np.ones((probe_size_y, probe_size_x))*ring_x
    if not isiterable(ring_y):
        ring_y = np.ones((probe_size_y, probe_size_x))*ring_y
    if not isiterable(ring_r):
        ring_r = np.ones((probe_size_y, probe_size_x))*ring_r

    signal_shape = (probe_size_y, probe_size_x, image_size_y, image_size_x)
    s = PixelatedSTEM(np.zeros(shape=signal_shape))
    for i in s:
        index = s.axes_manager.indices[::-1]
        test_data = TestData(
                size_x=image_size_x, size_y=image_size_y,
                default=False, blur=blur, blur_sigma=blur_sigma,
                downscale=downscale)
        dx, dy, dr = disk_x[index], disk_y[index], disk_r[index]
        rx, ry, rr = ring_x[index], ring_y[index], ring_r[index]
        test_data.add_disk(dx, dy, dr, I=disk_I)
        test_data.add_ring(rx, ry, rr, I=ring_I, lw_pix=ring_lw)
        s.data[index][:] = test_data.signal.data[:]
        if add_noise:
            s.data[index][:] += np.random.random(
                    size=(image_size_y, image_size_x)) * noise_amplitude
    return s


def get_disk_shift_simple_test_signal():
    disk_x, disk_y = np.mgrid[22:28:20j, 22:28:20j]
    s = generate_4d_disk_data(
            probe_size_x=20, probe_size_y=20,
            image_size_x=50, image_size_y=50,
            disk_x=disk_x, disk_y=disk_y, disk_r=2,
            add_noise=True)
    return(s)


def get_holz_simple_test_signal():
    ring_x, ring_y = np.mgrid[24:26:20j, 24:26:20j]
    s = generate_4d_holz_data(
            probe_size_x=20, probe_size_y=20,
            image_size_x=50, image_size_y=50,
            disk_x=25, disk_y=25, disk_r=2, disk_I=20,
            ring_x=ring_x, ring_y=ring_y, ring_r=15, ring_I=10,
            add_noise=True)
    return(s)
