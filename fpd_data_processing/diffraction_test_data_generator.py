import numpy as np
from scipy.ndimage.filters import gaussian_filter
from fpd_data_processing.pixelated_stem_class import PixelatedSTEM


class Circle(object):
    def __init__(self, xx, yy, x0, y0, r, I, scale):
        self.x0 = x0
        self.y0 = y0
        self.r = r
        self.I = I
        self.circle = (yy - self.y0) ** 2 + (xx - self.x0) ** 2 
        self.mask_outside_r(scale)
        self.get_centre_pixel(xx, yy, scale)

    def __repr__(self):
        return '<%s, (r: %s, (x0, y0): (%s, %s), I: %s)>' % (
            self.__class__.__name__,
            self.r, self.x0, self.y0, self.I,
            )

    def mask_outside_r(self, scale):
        indices = self.circle >= (self.r + scale)**2
        self.circle[indices] = 0

    def centre_on_image(self,xx,yy):
        if self.x0 < xx[0][0] or self.x0 > xx[0][-1]:
            return(False)
        elif self.y0 < yy[0][0] or self.y0 > yy[-1][-1]:
            return(False)
        else:
            return(True)

    def get_centre_pixel(self, xx, yy, scale):
        """
        This function sets the indices for the pixels on which the centre
        point is. Because the centrepoint can sometimes be exactly on the
        boundary of two pixels, the pixles are held in a list. One
        list for x (self.centre_x_pixels) and one for y
        (self.centre_x_pixels). If the centre is outside the image, the
        lists will be empty.
        """
        if self.centre_on_image(xx,yy):
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
        Sets the intensity of the centre pixles to I. Coordinates are
        self.z.circle[y, x], due to how numpy works.
        """
        for x in self.z.centre_x_pixels:
            for y in self.z.centre_y_pixels:
                self.z.circle[y, x] = self.z.I #This is correct

    def get_signal(self):
        return(self.z.circle)

    def update_axis(self, xx, yy):
        self.z.update_axis(xx, yy)
        self.z.set_uniform_intensity()
        self.set_centre_intensity()


class Ring(object):
    """
    Ring object, with outer edge of the ring at r, and inner r-lw
    """
    def __init__(self, xx, yy, scale, x0, y0, r, I, lw=1):
        self.lw = lw  # in coordinates
        self.z = Circle(xx, yy, x0, y0, r, I, scale)
        self.mask_inside_r(scale)
        self.z.set_uniform_intensity()

    def __repr__(self):
        return '<%s, (r: %s, (x0, y0): (%s, %s), I: %s)>' % (
            self.__class__.__name__, self.z.r, self.z.x0, self.z.y0, self.z.I,
            )

    def mask_inside_r(self, scale):
        indices = self.z.circle < (self.z.r - self.lw+scale)**2
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
    less intensive than the center Disk. Unlimited number of Ring and Disk can
    be added separately.

    Parameters
    ----------

    size_x, size_y : float, int
        The range of the x and y axis goes from 0 to size_x, size_y

    scale : float, int
        The step size of the x and y axis

    default : bool, default True
        If true, the default object should be generated. If false, Ring and
        Disk must be added separately by self.add_ring(), self.add_disk()

    downscale : bool, default True

    Attributes
    ----------

    signal : hyperspy.signals.Signal2D
        Test signal

    z_list : list
        List containing Ring and Disk objects added to the signal

    downscale_factor : int
        The data is upscaled before Circle is added, and similaraly
        downscaled to return to given dimensions. This improves the
        quality of Circle

    """
    def __init__(
            self, size_x=100, size_y=100, scale=1,
            default=True, blur=True, sigma_blur=1, downscale=True):
        self.scale = scale
        self.blur_on = blur
        self.sigma_blur = sigma_blur
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

    def add_ring(self, x0=50, y0=50, r=20, I=10, lw_pix=1):
        lw = lw_pix*self.scale
        scale = self.scale/self.downscale_factor
        self.z_list.append(
                Ring(self.xx, self.yy, scale, x0, y0, r, I, lw=lw))
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
                    self.z_downscaled, sigma=self.sigma_blur)
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
