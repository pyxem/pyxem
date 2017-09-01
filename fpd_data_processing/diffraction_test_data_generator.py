import matplotlib.pyplot as plt
import numpy as np
from hyperspy.signals import Signal2D
from scipy.ndimage.filters import gaussian_filter

class Circle:
    def __init__(self,xx,yy,x0,y0,r,I):
        self.x0 = x0
        self.y0 = y0
        self.r = r
        self.I = I
        self.circle = (xx - self.x0) ** 2 + (yy - self.y0) ** 2
        self.mask_outside_r()

    def mask_outside_r(self):
        lw = 0.5
        indices = self.circle > np.sqrt(self.r+lw)
        self.circle[indices] = 0

    def set_uniform_intensity(self):
        circle_ring_indices = self.circle > 0
        self.circle[circle_ring_indices] = self.I

    def update_axis(self,xx,yy):
        self.circle = (xx - self.x0) ** 2 + (yy - self.y0) ** 2
        self.mask_outside_r()

class Disk(object):
    def __init__(self,xx,yy,x0,y0,r,I):
        self.z = Circle(xx,yy,x0,y0,r,I)
        self.center_x, self.center_y = np.argmin(self.z.circle,axis=0)[0], np.argmin(self.z.circle,axis=1)[0]
        self.z.set_uniform_intensity()
        self.z.circle[self.center_x,self.center_y] = I

    def __repr__(self):
        return '<%s, (r: %s, (x0, y0): (%s, %s), I: %s)>' % (
            self.__class__.__name__,
            self.z.r,
            self.z.x0,
            self.z.y0,
            self.z.I,
            )


    def get_signal(self):
        return(self.z.circle)

    def update_axis(self,xx,yy):
        self.z.update_axis(xx,yy)
        self.center_x, self.center_y = np.argmin(self.z.circle,axis=0)[0], np.argmin(self.z.circle,axis=1)[0]
        self.z.set_uniform_intensity()
        self.z.circle[self.center_x,self.center_y] = self.z.I


class Ring(object):
    def __init__(self,xx,yy,x0,y0,r,I):
        self.z = Circle(xx,yy,x0,y0,r,I)
        self.mask_inside_r()
        self.z.set_uniform_intensity()
        
    def __repr__(self):
        return '<%s, (r: %s, (x0, y0): (%s, %s), I: %s)>' % (
            self.__class__.__name__,
            self.z.r,
            self.z.x0,
            self.z.y0,
            self.z.I,
            )

    def mask_inside_r(self):
        lw = 1
        indices = self.z.circle < np.sqrt(self.z.r-lw)
        self.z.circle[indices] = 0
        
    def get_signal(self):
        return(self.z.circle)
        
    def update_axis(self,xx,yy):
        self.z.update_axis(xx,yy)
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
    
    default : bool
        If true, the default object should be generated. If false, Ring and
        Disk must be added separately by self.add_ring(), self.add_disk()
    
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
    def __init__(self,size_x=10,size_y=10,scale=0.05,default=True):
        self.scale = scale
        self.size_x, self.size_y = size_x, size_y
        self.downscale_factor = 5
        self.generate_mesh()
        self.z_list = []
        if default:
            self.add_disk()
            self.add_ring()

    def update_signal(self):
        self.make_signal()
        self.downscale()
        self.blur()

    def generate_mesh(self):
        self.X = np.arange(0, self.size_x, self.scale/self.downscale_factor)
        self.Y = np.arange(0, self.size_y, self.scale/self.downscale_factor)
        self.xx, self.yy = np.meshgrid(self.X, self.Y, sparse=True)
        
    def add_disk(self,x0=5,y0=5,r=1,I=10):
        self.z_list.append(Disk(self.xx,self.yy,x0,y0,r,I))
        self.update_signal()

    def add_ring(self,x0=5,y0=5,r=20,I=10):
        self.z_list.append(Ring(self.xx,self.yy,x0,y0,r,I))        
        self.update_signal()

    def make_signal(self):
        if len(self.z_list) == 0:
            print('Empty test data')
            self.z = None
        elif len(self.z_list) == 1:
            self.z = self.z_list[0].get_signal()
        elif len(self.z_list) > 1:
            z_temp = self.z_list[0].get_signal()
            for i in self.z_list[1:]:
                z_temp = np.add(z_temp,i.get_signal())
            self.z = z_temp
            
    def downscale(self):
        shape = (int(self.z.shape[0]/self.downscale_factor),int(self.z.shape[1]/self.downscale_factor))
        sh = shape[0],self.z.shape[0]//shape[0],shape[1],self.z.shape[1]//shape[1]
        self.z_downscaled = self.z.reshape(sh).mean(-1).mean(1)

    def blur(self):
        self.signal = Signal2D(gaussian_filter(self.z_downscaled, sigma=2))
        self.signal.axes_manager[0].scale = self.scale
        self.signal.axes_manager[1].scale = self.scale
        
    def set_downscale_factor(self,factor):
            self.downscale_factor = factor
            self.generate_mesh()
            for i in self.z_list:
                i.update_axis(self.xx,self.yy)
            self.update_signal()
