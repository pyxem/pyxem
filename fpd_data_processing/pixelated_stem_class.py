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
        tuple with center x and y arrays. (com x, com y)
        
        Examples
        --------
        With mask centered at x=105, y=120 and 30 pixel radius
        >>> mask = (105, 120, 30)
        >>> s_com = s.center_of_mass(mask=mask)
        
        Also threshold
        >>> s_com = s.center_of_mass(threshold=1.5)
        """

        if mask is not None:
            x, y, r = mask
            im_x, im_y = self.axes_manager.signal_shape
            mask = pst._make_circular_mask(x, y, im_x, im_y, r)

        # Signal is transposed, due to the DPC signals having the
        # x and y deflections as navigation dimension, and probe
        # positions as signal dimensions
        s_com = self.map(
                pst._center_of_mass_single_frame,
                threshold=threshold, mask=mask,
                ragged=False, inplace=False)
        if self.axes_manager.navigation_dimension == 0:
            s_com = DPCBaseSignal(s_com.data).T
        elif self.axes_manager.navigation_dimension == 1:
            s_com = DPCSignal1D(s_com.T.data)
        elif self.axes_manager.navigation_dimension == 2:
            s_com = DPCSignal2D(s_com.T.data)
        s_com.axes_manager.navigation_axes[0].name = "Beam position"
        return(s_com)

    def radial_integration(
            self, centre_x_array=None, centre_y_array=None, mask_array=None):
        """Radially integrates a pixelated STEM diffraction signal.

        Parameters
        ----------
        centre_x_array, centre_y_array : NumPy 2D array, optional
            Has to have the same shape as the navigation axis of
            the signal.
        mask_array : Boolean numpy array
            Mask with the same shape as the signal.

        Returns
        -------
        HyperSpy signal, one less signal dimension than the input signal."""
        if (centre_x_array is None) or (centre_y_array is None):
            centre_x_array, centre_y_array = pst._make_centre_array_from_signal(self)

        centre_x_array = centre_x_array.flatten()
        centre_y_array = centre_y_array.flatten()
        iterating_kwargs = (('centre_x', centre_x_array), ('centre_y', centre_x_array))
        s_radial = self._map_iterate(
                pst._get_radial_profile_of_diff_image,
                iterating_kwargs=iterating_kwargs,
                parallel=False, inplace=False)
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
        s_hist = pst._make_bivariate_histogram(
                    x_position, y_position,
                    histogram_range=histogram_range,
                    masked=masked,
                    bins=bins,
                    spatial_std=spatial_std)
        return(s_hist)


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
        out : optional, DPCImage signal

        Returns
        -------
        corrected_signal : Signal2D
        """
        if out is None:
            output = self.deepcopy()
        else:
            output = out

        for i, s in enumerate(self):
            ramp = pst._fit_ramp_to_image(s, corner_size=0.05)
            output.data[i, :, :] -= ramp
        if out is None:
            return(output)

    def get_color_signal(self, autolim=True, autolim_sigma=4):
        angle = np.arctan2(self.inav[0].data, self.inav[1].data)
        magnitude = np.sqrt(
                np.abs(self.inav[0].data)**2+np.abs(self.inav[1].data)**2)
        
        magnitude_limits = None
        if autolim:
            magnitude_limits = pst._get_limits_from_array(
                    magnitude, sigma=autolim_sigma)
        rgb_array = pst._get_rgb_array(
                angle=angle, magnitude=magnitude,
                magnitude_limits=magnitude_limits)
        signal_rgb = Signal1D(rgb_array*(2**16-1))
        signal_rgb.change_dtype("uint16")
        signal_rgb.change_dtype("rgb16")
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
            for data0, data1, masked_value in zip(
                    s0_flat, s1_flat, masked.flatten()):
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
                    [s0_range[0], s0_range[1]],
                    [s1_range[0], s1_range[1]]])

        s_hist = Signal2D(hist2d)
        s_hist.axes_manager[0].offset = xedges[0]
        s_hist.axes_manager[0].scale = xedges[1] - xedges[0]
        s_hist.axes_manager[1].offset = yedges[0]
        s_hist.axes_manager[1].scale = yedges[1] - yedges[0]
        return(s_hist)
