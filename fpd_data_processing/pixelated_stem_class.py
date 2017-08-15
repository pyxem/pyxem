import numpy as np
from scipy.optimize import leastsq
from hyperspy.signals import Signal1D, Signal2D
import fpd_data_processing.pixelated_stem_tools as pst

class PixelatedSTEM(Signal2D):

    def center_of_mass(self, threshold=None, mask=None):
        """Get the centre of the STEM diffraction pattern using
        center of mass. Threshold can be set to only use the most
        intense parts of the pattern. A mask can be used to exclude
        parts of the diffraction pattern.

        Parameters
        ----------
        threshold : number, optional
            The thresholding will be done at mean times 
            this threshold value.
        mask : tuple (x, y, r)
            Round mask centered on x and y, with radius r.

        Returns
        -------
        tuple with center x and y arrays. (com x, com y)"""

        if mask is not None:
            x, y, r = mask
            im_x, im_y = self.axes_manager.signal_shape
            pst._make_circular_mask(x, y, im_x, im_y, r)

        s_com = self.map(
                pst._center_of_mass_single_frame,
                threshold=threshold, mask=mask,
                ragged=False, inplace=False)
        s_com = DPCImage(s_com.T.data)
        return(s_com)

    def radial_integration(self, centre_x_array=None, centre_y_array=None):
        """Radially integrates a 4-D pixelated STEM diffraction signal.

        Parameters
        ----------
        centre_x_array, centre_y_array : NumPy 2D array, optional
            Has to have the same shape as the navigation axis of
            the signal.

        Returns
        -------
        3-D HyperSpy signal, 2 spatial dimensions,
        1 integrated reciprocal dimension."""

        s_radial = pst._do_radial_integration(
                self,
                centre_x_array=centre_x_array,
                centre_y_array=centre_y_array)
        return(s_radial)


class DPCImage(Signal2D):
    """
    Signal for processing differential phase contrast (DPC) acquired using
    scanning transmission electron microscopy (STEM).

    The signal assumes the data is 3 dimensions, where the two
    signal dimensions are the probe positions, and the navigation
    dimension is the x and y disk shifts.

    The first navigation index (s.inav[0]) is assumed to the be x-shift
    and the second navigation is the y-shift (s.inav[1]).
    """

    def correct_ramp(self, corner_size=0.05, out=None):
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
        out : optional, DCPImage signal

        Returns
        -------
        corrected_signal : Signal2D
        """
        def _f_min(X,p):
            plane_xyz = p[0:3]
            distance = (plane_xyz*X.T).sum(axis=1) + p[3]
            return distance / np.linalg.norm(plane_xyz)

        def _residuals(params, signal, X):
            return _f_min(X, params)

        if out is None:
            out = self.deepcopy()

        for i, s in enumerate(self):
            corner_values = pst._get_corner_value(s, corner_size=corner_size)
            p0 = [0.1, 0.1, 0.1, 0.1]

            p = leastsq(_residuals, p0, args=(None, corner_values))[0]
            
            xx, yy = np.meshgrid(s.axes_manager[0].axis, s.axes_manager[1].axis)
            zz = (-p[0]*xx-p[1]*yy-p[3])/p[2]
            out.data[i,:,:] -= zz
        if out is None:
            return(out)

    def get_color_signal(self):
        rgb_array = pst._get_rgb_array(self.inav[0], self.inav[1])
        signal_rgb = Signal1D(rgb_array*255)
        signal_rgb.change_dtype("uint8")
        signal_rgb.change_dtype("rgb8")
        return(signal_rgb)

    def get_bivariate_histogram(
            self,
            histogram_range=None, 
            masked=None, 
            bins=200,
            spatial_std=3):
        """
        Useful for finding the distribution of magnetic vectors(?).

        Parameters:
        -----------
        histogram_range : tuple, optional
            Set the minimum and maximum of the histogram range.
            Default is setting it automatically.
        masked : 2-D numpy bool array, optional
            Mask parts of the data. The array must be the same
            size as the signal. The True values are masked.
            Default is not masking anything.
        bins : integer, optional
            Number of bins in the histogram
        spatial_std : number, optional
            If histogram_range is not given, this value will be
            used to set the automatic histogram range.
            Default value is 3.

        Returns
        -------
        s_hist : Signal2D
        """
        s0_flat = self.inav[0].data.flatten()
        s1_flat = self.inav[1].data.flatten()
            
        if masked is not None:
            temp_s0_flat = []
            temp_s1_flat = []
            for data0, data1, masked_value in zip(s0_flat, s1_flat, masked.flatten()):
                if not (masked_value == True):
                    temp_s0_flat.append(data0)
                    temp_s1_flat.append(data1)
            s0_flat = np.array(temp_s0_flat)
            s1_flat = np.array(temp_s1_flat)

        if histogram_range is None:
            if (s0_flat.std() > s1_flat.std()):
                s0_range = (
                    s0_flat.mean()-s0_flat.std()*spatial_std,
                    s0_flat.mean()+s0_flat.std()*spatial_std)
                s1_range = (
                    s1_flat.mean()-s0_flat.std()*spatial_std,
                    s1_flat.mean()+s0_flat.std()*spatial_std)
            else:
                s0_range = (
                    s0_flat.mean()-s1_flat.std()*spatial_std,
                    s0_flat.mean()+s1_flat.std()*spatial_std)
                s1_range = (
                    s1_flat.mean()-s1_flat.std()*spatial_std,
                    s1_flat.mean()+s1_flat.std()*spatial_std)
        else:
            s0_range = histogram_range
            s1_range = histogram_range

        hist2d, xedges, yedges = np.histogram2d(
                s0_flat,
                s1_flat,
                bins=bins,
                range=[
                    [s0_range[0],s0_range[1]],
                    [s1_range[0],s1_range[1]]])

        s_hist = Signal2D(hist2d)
        s_hist.axes_manager[0].offset = xedges[0]
        s_hist.axes_manager[0].scale = xedges[1] - xedges[0]
        s_hist.axes_manager[1].offset = yedges[0]
        s_hist.axes_manager[1].scale = yedges[1] - yedges[0]
        return(s_hist)
