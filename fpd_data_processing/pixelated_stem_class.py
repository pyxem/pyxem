import numpy as np
import hyperspy.api as hs
from hyperspy.signals import BaseSignal, Signal1D, Signal2D
import fpd_data_processing.pixelated_stem_tools as pst
from tqdm import tqdm


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
            mask = pst._make_circular_mask(x, y, im_x, im_y, r)

        s_com = self.map(
                pst._center_of_mass_single_frame,
                threshold=threshold, mask=mask,
                ragged=False, inplace=False)
        if len(s_com.axes_manager.shape) != 1:
            s_com = DPCSignal(s_com.T.data)
        return(s_com)

    def radial_integration(
            self, centre_x_array=None, centre_y_array=None, mask_array=None):
        """Radially integrates a 4-D pixelated STEM diffraction signal.

        Parameters
        ----------
        centre_x_array, centre_y_array : NumPy 2D array, optional
            Has to have the same shape as the navigation axis of
            the signal.
        mask_array : Boolean numpy array
            Mask with the same shape as the signal.

        Returns
        -------
        3-D HyperSpy signal, 2 spatial dimensions,
        1 integrated reciprocal dimension."""
        s_radial = pst._do_radial_integration(
                self,
                centre_x_array=centre_x_array,
                centre_y_array=centre_y_array,
                mask_array=mask_array)
        return(s_radial)

    def angular_mask(
            self, angle0, angle1,
            centre_x_array=None, centre_y_array=None):
        """Get a bool array with True values between angle0 and angle1.
        Will use the (0, 0) point as given by the signal as the centre,
        giving an "angular" slice. Useful for analysing anisotropy in
        diffraction patterns.

        Parameters
        ----------
        angle0, angle1 : numbers
            Must be between 0 and 2*pi.
        centre_x_array, centre_y_array : NumPy 2D array, optional
            Has to have the same shape as the navigation axis of
            the signal.

        Returns
        -------
        mask_array : Numpy array
            The True values will be the region between angle0 and angle1.
            The array will have the same dimensions as the signal.

        Examples
        --------
        >>> mask_array = s.annular_mask(0.5*np.pi, np.pi)
        """

        bool_array = pst._get_angle_sector_mask(
                self, angle0, angle1,
                centre_x_array=centre_x_array,
                centre_y_array=centre_y_array)
        return(bool_array)

    def angular_slice_radial_integration(
            self, angleN=20,
            centre_x_array=None, centre_y_array=None):
        signal_list = []
        angle_list = []
        for i in range(angleN):
            angle_list.append((2*np.pi*i/angleN, 2*np.pi*(i+1)/angleN))
        if (centre_x_array is None) or (centre_y_array is None):
            centre_x_array, centre_y_array = pst._make_centre_array_from_signal(
                    self)
        for angle in tqdm(angle_list):
            mask_array = self.angular_mask(
                    angle[0], angle[1],
                    centre_x_array=centre_x_array,
                    centre_y_array=centre_y_array)
            s_r = self.radial_integration(
                    centre_x_array=centre_x_array,
                    centre_y_array=centre_y_array,
                    mask_array=mask_array)
            signal_list.append(s_r)
        angle_scale = angle_list[1][1] - angle_list[0][1]
        signal = hs.stack(signal_list, new_axis_name='Angle slice')
        signal.axes_manager['Angle slice'].scale = angle_scale
        return(signal)


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
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


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
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

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
        masked : 1-D numpy bool array, optional
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
        s_hist : Signal1D
        """
        x_position = self.inav[0].data
        y_position = self.inav[1].data
        s_hist = _get_bivariate_histogram(
                    x_position, y_position,
                    histogram_range=histogram_range,
                    masked=masked,
                    bins=bins,
                    spatial_std=spatial_std)
        return(s_hist)
