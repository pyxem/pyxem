import numpy as np
from scipy.ndimage import measurements
from hyperspy.signals import Signal1D
import copy
from matplotlib.colors import hsv_to_rgb, to_rgba

def _center_of_mass_single_frame(im, threshold=None, mask=None):
    if (mask is not None) or (threshold is not None):
        image = copy.deepcopy(im)
    else:
        image = im
    if threshold is not None:
        mean_value = measurements.mean(image, mask)*threshold
        image[image <= mean_value] = 0
        image[image > mean_value] = 1
    data = measurements.center_of_mass(image, labels=mask)
    return(np.array(data)[::-1])


def _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius):
    """
    Make a circular mask in a bool array for masking a region in an image.

    Parameters
    ----------
    centreX, centreY : float
        Centre point of the mask.
    imageSizeX, imageSizeY : int
        Size of the image to be masked.
    radius : float
        Radius of the mask.

    Returns
    -------
    Boolean Numpy 2D Array
        Array with the shape (imageSizeX, imageSizeY) with the mask.

    See also
    --------
    _make_mask_from_positions

    Examples
    --------
    >>> import numpy as np
    >>> from atomap.atom_finding_refining import _make_circular_mask
    >>> image = np.ones((9, 9))
    >>> mask = _make_circular_mask(4, 4, 9, 9, 2)
    >>> image_masked = image*mask
    >>> import matplotlib.pyplot as plt
    >>> cax = plt.imshow(image_masked)
    """
    x, y = np.ogrid[-centerY:imageSizeY-centerY, -centerX:imageSizeX-centerX]
    mask = x*x + y*y <= radius*radius
    return(mask)


def _get_corner_value(signal, corner_size=0.05):
    d_axis0_range = (
            signal.axes_manager[0].high_value - 
            signal.axes_manager[0].low_value)*corner_size
    d_axis1_range = (
            signal.axes_manager[1].high_value - 
            signal.axes_manager[1].low_value)*corner_size
    s_corner00 = signal.isig[0:d_axis0_range,0:d_axis1_range]
    s_corner01 = signal.isig[0:d_axis0_range,-d_axis1_range:-1]
    s_corner10 = signal.isig[-d_axis0_range:-1,0:d_axis1_range]
    s_corner11 = signal.isig[-d_axis0_range:-1,-d_axis1_range:-1]

    corner00 = (
            s_corner00.axes_manager[0].axis.mean(),
            s_corner00.axes_manager[1].axis.mean(),
            s_corner00.data.mean())
    corner01 = (
            s_corner01.axes_manager[0].axis.mean(),
            s_corner01.axes_manager[1].axis.mean(),
            s_corner01.data.mean())
    corner10 = (
            s_corner10.axes_manager[0].axis.mean(),
            s_corner10.axes_manager[1].axis.mean(),
            s_corner10.data.mean())
    corner11 = (
            s_corner11.axes_manager[0].axis.mean(),
            s_corner11.axes_manager[1].axis.mean(),
            s_corner11.data.mean())
    
    return(np.array((corner00,corner01,corner10,corner11)).T)


def normalize_array(np_array, max_number=1.0):
    np_array = copy.deepcopy(np_array)
    np_array -= np_array.min()
    np_array /= np_array.max()
    return(np_array*max_number)


def _get_rgb_array(
        angle, magnitude, rotation=0, angle_lim=None,
        magnitude_lim=None, max_angle=2*np.pi):
    if not (rotation == 0):
        angle = (angle + math.radians(rotation)) % (max_angle)
    if angle_lim is not None:
        np.clip(angle, angle_lim[0], angle_lim[1], out=angle)
    else:
        angle = normalize_array(angle)
    if magnitude_lim is not None:
        np.clip(magnitude, magnitude_lim[0], magnitude_lim[1], out=magnitude)
    magnitude = normalize_array(magnitude)
    S = np.ones_like(angle)
    HSV = np.dstack((angle, S, magnitude))
    RGB = hsv_to_rgb(HSV)
    return(RGB)


def _find_longest_distance(
        imX, imY,
        centreX_min, centreY_min,
        centreX_max, centreY_max,
        ):
    max_value = max(
            int(((imX-centreX_min)**2+(imY-centreY_min)**2)**0.5),
            int(((centreX_max)**2+(imY-centreY_min)**2)**0.5),
            int(((imX-centreX_min)**2+(centreY_max)**2)**0.5),
            int((centreX_max**2+centreY_max**2)**0.5))
    return(max_value)


def _do_radial_integration(
        signal, centre_x_array, centre_y_array, mask_array=None):
    if centre_x_array is None:
        a_m = signal.axes_manager
        shape = a_m.navigation_shape
        centre_x_array = np.ones(shape)*a_m.signal_axes[0].value2index(0)
    if centre_y_array is None:
        a_m = signal.axes_manager
        shape = a_m.navigation_shape
        centre_y_array = np.ones(shape)*a_m.signal_axes[1].value2index(0)
    radial_array_size = _find_longest_distance(
            signal.axes_manager.signal_axes[1].size,
            signal.axes_manager.signal_axes[0].size,
            centre_x_array.min(), centre_y_array.min(),
            centre_x_array.max(), centre_y_array.max())+1
    radial_array_shape = list(signal.axes_manager.navigation_shape)
    radial_array_shape.append(radial_array_size)
    radial_profile_array = np.zeros(radial_array_shape, dtype=np.float64)
    diff_image = np.zeros(signal.data.shape[2:], dtype=np.uint16)
    for x in range(signal.axes_manager[1].size):
        for y in range(signal.axes_manager[0].size):
            diff_image[:] = signal.data[x,y,:,:]
            if mask_array is None:
                mask = None
            else:
                mask = mask_array[x, y, :, :]
            centre_x = centre_x_array[x,y]
            centre_y = centre_y_array[x,y]
            radial_profile = _get_radial_profile_of_diff_image(
                    diff_image, centre_x, centre_y, mask=mask)
            radial_profile_array[x,y,0:len(radial_profile)] = radial_profile
    
    signal_radial = Signal1D(radial_profile_array)
    return(signal_radial)


def _get_lowest_index_radial_array(radial_array):
    """Returns the lowest index of in a radial array.

    Parameters
    ----------
    radial_array : 3-D numpy array
        The last dimension must be the reciprocal dimension.

    Returns
    -------
    Number, lowest index."""
    lowest_index = radial_array.shape[-1]
    for x in range(radial_array.shape[0]):
        for y in range(radial_array.shape[1]):
            radial_data = radial_array[x,y,:]
            lowest_index_in_image = np.where(radial_data==0)[0][0]
            if lowest_index_in_image < lowest_index:
                lowest_index = lowest_index_in_image
    return(lowest_index)


def _get_radial_profile_of_diff_image(
        diff_image, centre_x, centre_y, mask=None):
    """Radially integrates a single diffraction image.

    Parameters
    ----------
    diff_image : 2-D numpy array
        Array consisting of a single diffraction image.
    centre_x : number
        Centre x position of the diffraction image.
    centre_y : number
        Centre y position of the diffraction image.
    mask : numpy bool array, optional
        Mask parts of the diffraction image, regions where
        the mask is True will be included in the radial profile.

    Returns
    -------
    1-D numpy array of the radial profile."""
#   Radially profiles the data, integrating the intensity in rings out from the centre. 
#   Unreliable as we approach the edges of the image as it just profiles the corners.
#   Less pixels there so become effectively zero after a certain point
    y, x = np.indices((diff_image.shape))
    r = np.sqrt((x - centre_x)**2 + (y - centre_y)**2)
    r = r.astype(int)       
    if mask is None:
        r_flat = r.ravel()
        diff_image_flat = diff_image.ravel()
    else:
        r_flat = r[mask].ravel()
        diff_image_flat = diff_image[mask].ravel()
    tbin =  np.bincount(r_flat, diff_image_flat)
    nr = np.bincount(r_flat)
    nr.clip(1, out=nr) # To avoid NaN in data due to dividing by 0
    radialProfile = tbin / nr
    return(radialProfile)


def _get_angle_sector_mask(
        signal, angle0, angle1,
        centre_x_array=None, centre_y_array=None):
    """Get a bool array with True values between angle0 and angle1.
    Will use the (0, 0) point as given by the signal as the centre,
    giving an "angular" slice. Useful for analysing anisotropy in
    diffraction patterns. 

    Parameters
    ----------
    signal : HyperSpy 2-D signal
        Can have several navigation dimensions.
    angle0, angle1 : numbers
        Must be between 0 and 2*pi.

    Returns
    -------
    Mask : Numpy array
        The True values will be the region between angle0 and angle1.
        The array will have the same dimensions as the input signal.

    Examples
    --------
    >>> mask = _get_angle_sector_mask(signal, 0.5*np.pi, np.pi)
    """
    if signal.axes_manager.navigation_dimension != 2:
        raise ValueError(
            "The signal must be 4-D, with 2 navigation dimenions "
            "and 2 signal dimensions")
    bool_array = np.zeros_like(signal.data, dtype=np.bool)
    nX, nY = signal.axes_manager.navigation_shape
    for i, s in enumerate(signal):
        iX, iY = int(i/nX), i%nX 
        signal_axes = s.axes_manager.signal_axes
        if centre_x_array is not None:
            signal_axes[0].offset = -centre_x_array[iX, iY]
        if centre_y_array is not None:
            signal_axes[1].offset = -centre_y_array[iX, iY]
        x_size = signal_axes[0].size*1j
        y_size = signal_axes[1].size*1j
        x, y = np.mgrid[
                signal_axes[0].low_value:signal_axes[0].high_value:x_size,
                signal_axes[1].low_value:signal_axes[1].high_value:y_size]
        r = (x**2+y**2)**0.5
        t = np.arctan2(x,y)+np.pi
        bool_array[iX, iY] = (t>angle0)*(t<angle1)
    return(bool_array)
