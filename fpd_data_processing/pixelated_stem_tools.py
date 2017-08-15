import numpy as np
from scipy.ndimage.measurements import center_of_mass
from hyperspy.signals import Signal1D


def _center_of_mass_single_frame(im, threshold=None, mask=None):
    if threshold is not None:
        mean_value = im.mean()*threshold
        im[im <= mean_value] = 0
        im[im > mean_value] = 1
    if mask is not None:
        im = im*mask
    data = center_of_mass(im)
    return(np.array(data))


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
    y, x = np.ogrid[-centerX:imageSizeX-centerX, -centerY:imageSizeY-centerY]
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


def _get_color_channel(a_array, mu0, si0, mu1, si1, mu2, si2):
    color_array = np.zeros((a_array.shape[0], a_array.shape[1]))
    color_array[:] = 1.-(
            np.exp(-1*((a_array-mu0)**2)/si0)+
            np.exp(-1*((a_array-mu1)**2)/si1)+
            np.exp(-1*((a_array-mu2)**2)/si2))
    return(color_array)


def _get_rgb_array(signal0, signal1):
    arctan_array = np.arctan2(signal0.data, signal1.data) + np.pi

    color0 = _get_color_channel(arctan_array, 3.7, 0.8, 5.8, 5.0, 0.0, 0.3)
    color1 = _get_color_channel(arctan_array, 2.9, 0.6, 1.7, 0.3, 2.4, 0.5)
    color2 = _get_color_channel(arctan_array, 0.0, 1.3, 6.4, 1.0, 1.0, 0.75)

    rgb_array = np.zeros((signal0.data.shape[0], signal0.data.shape[1], 3))
    rgb_array[:,:,2] = color0
    rgb_array[:,:,1] = color1 
    rgb_array[:,:,0] = color2 

    return(rgb_array)


def _do_radial_integration(signal, centre_x_array, centre_y_array):
    if centre_x_array is None:
        a_m = signal.axes_manager
        shape = a_m.navigation_shape
        centre_x_array = np.ones(shape)*a_m.signal_axes[0].value2index(0)
    if centre_y_array is None:
        a_m = signal.axes_manager
        shape = a_m.navigation_shape
        centre_y_array = np.ones(shape)*a_m.signal_axes[1].value2index(0)
    radial_profile_array = np.zeros(signal.data.shape[0:-1], dtype=np.float64)
    diff_image = np.zeros(signal.data.shape[2:], dtype=np.uint16)
    for x in range(signal.axes_manager[0].size):
        for y in range(signal.axes_manager[1].size):
            diff_image[:] = signal.data[x,y,:,:]
            centre_x = centre_x_array[x,y]
            centre_y = centre_y_array[x,y]
            radial_profile = _get_radial_profile_of_diff_image(
                    diff_image, centre_x, centre_y)
            radial_profile_array[x,y,0:len(radial_profile)] = radial_profile

    lowest_radial_zero_index = _get_lowest_index_radial_array(
            radial_profile_array)
    signal_radial = Signal1D(
            radial_profile_array[:,:,0:lowest_radial_zero_index])
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
    radialProfile = tbin / nr
    return(radialProfile)


