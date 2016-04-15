import hyperspy.api as hs
import numpy as np
import scipy as sp
from scipy.ndimage.measurements import center_of_mass
import h5py
import copy
import warnings
import laue_zone_processing as lzp

def _set_metadata_from_hdf5(hdf5_file, signal):
    """Get microscope and scan metadata from fpd HDF5-file reference.
    Will set acceleration voltage and camera length, store it
    in the signal. Will set probe position axis scales.
    Operates in-place.

    Parameters
    ----------
    hdf5_file : hdf5 file reference(?)
        An opened fpd hdf5 file reference
    signal : HyperSpy signal instance
    """
    signal.metadata.add_node("Microscope")
    dm_data = hdf5_file['fpd_expt']['DM0']['tags']['ImageList']['TagGroup0']
    metadata_ref = dm_data['ImageTags']['Microscope Info']

    signal.metadata.Microscope['Voltage'] = metadata_ref['Voltage'].value
    signal.metadata.Microscope['Camera length'] = metadata_ref['STEM Camera Length'].value

    axis_scale_x = hdf5_file['fpd_expt']['fpd_data']['dim2'][0:2]
    axis_scale_y = hdf5_file['fpd_expt']['fpd_data']['dim1'][0:2]
    axis_units_x = hdf5_file['fpd_expt']['fpd_data']['dim2'].attrs['units']
    axis_units_y = hdf5_file['fpd_expt']['fpd_data']['dim1'].attrs['units']

    signal.axes_manager[0].scale = axis_scale_x[1]-axis_scale_x[0]
    signal.axes_manager[1].scale = axis_scale_y[1]-axis_scale_y[0]
    signal.axes_manager[0].units = axis_units_x
    signal.axes_manager[1].units = axis_units_y
    
def _set_hardcoded_metadata(signal):
    """Sets some metadata values, like axis names.
    Operates in-place.
    
    Parameters
    ----------

    """
    signal.axes_manager[0].name = "Probe x"
    signal.axes_manager[1].name = "Probe y"
    signal.axes_manager[2].name = "Detector x"
    signal.axes_manager[3].name = "Detector y"

def _load_fpd_dataset(filename, x_range=None, y_range=None):
    """Load data from a fast-pixelated data HDF5-file.
    Can get a partial file by using x_range and y_range.
    
    Parameters
    ----------
    filename : string
        Name of the fpd HDF5 file.
    x_range : tuple, optional
        Instead of returning the full dataset, only parts
        of the x probe positions will be returned. Useful for very large
        datasets. Default is None, which will return all the x probe
        positions. This _might_ greatly increase the loading time.
    y_range : tuple, optional
        Instead of returning the full dataset, only parts
        of the y probe positions will be returned. Useful for very large
        datasets. Default is None, which will return all the y probe
        positions. This _might_ greatly increase the loading time.

    Returns
    -------
    4-D HyperSpy image signal

    """
    fpdfile = h5py.File(filename,'r') #find data file in a read only format
    data_reference = fpdfile['fpd_expt']['fpd_data']['data']
    # Slightly convoluted way of loading parts of a dataset, since it takes
    # a very long time to do slicing directly on a HDF5-file
    if (x_range == None) and (y_range == None):
        data = data_reference[:][:,:,0,:,:]
    elif (not (x_range == None)) and (not(y_range==None)):
        data = data_reference[y_range[0]:y_range[1],x_range[0]:x_range[1],:,:,:][:,:,0,:,:]
    elif x_range == None:
        data = data_reference[y_range[0]:y_range[1],:,:,:,:][:,:,0,:,:]
    elif y_range == None:
        data = data_reference[:,x_range[0]:x_range[1],:,:,:][:,:,0,:,:]
    else:
        print("Something went wrong in the data loading")
    im = hs.signals.Image(data)
    _set_metadata_from_hdf5(fpdfile, im)
    _set_hardcoded_metadata(im)
    fpdfile.close()
    _remove_dead_pixels(im)
    return(im)

def _remove_dead_pixels(s_fpd):
    """Removes dead pixels from a fpd dataset in-place. 
    Replaces them with an average of the four nereast neighbors.
    
    Parameters
    ----------
    s_fpd : HyperSpy signal
    """
    s_sum = s_fpd.sum(0).sum(0)
    dead_pixels_x, dead_pixels_y = np.where(s_sum.data==0)

    for x, y in zip(dead_pixels_x, dead_pixels_y):
        n_pixel0 = s_fpd.data[:,:,x+1,y]
        n_pixel1 = s_fpd.data[:,:,x-1,y]
        n_pixel2 = s_fpd.data[:,:,x,y+1]
        n_pixel3 = s_fpd.data[:,:,x,y-1]
        s_fpd.data[:,:,x,y] = (n_pixel0+n_pixel1+n_pixel2+n_pixel3)/4


def _get_disk_centre_from_signal(signal, threshold=1.):
    """Get the centre of the disk using thresholded center of mass.
    Threshold is set to the mean of individual diffration images.

    Parameters
    ----------
    signal : HyperSpy 4-D fpd signal
    threshold : number, optional
        The thresholding will be done at mean times 
        this threshold value.

    Returns
    -------
    tuple with center x and y arrays. (com x, com y)"""
        

    mean_diff_array = signal.data.mean(axis=(2,3), dtype=np.float32)*threshold

    com_x_array = np.zeros(signal.data.shape[0:2], dtype=np.float64)
    com_y_array = np.zeros(signal.data.shape[0:2], dtype=np.float64)
    
    image_data = np.zeros(signal.data.shape[2:], dtype=np.uint16)

    # Slow, but memory efficient way
    for x in range(signal.axes_manager[0].size):
        for y in range(signal.axes_manager[1].size):
            image_data[:] = signal.data[x,y,:,:]
            image_data[image_data<mean_diff_array[x,y]] = 0
            image_data[image_data>mean_diff_array[x,y]] = 1
            com_y, com_x = center_of_mass(image_data)
            com_x_array[x,y] = com_x
            com_y_array[x,y] = com_y
    return(com_x_array, com_y_array)

def _get_radial_profile_of_diff_image(diff_image, centre_x, centre_y):
    """Radially integrates a single diffraction image.

    Parameters
    ----------
    diff_image : 2-D numpy array
        Array consisting of a single diffraction image.
    centre_x : number
        Centre x position of the diffraction image.
    centre_y : number
        Centre y position of the diffraction image.

    Returns
    -------
    1-D numpy array of the radial profile."""
#   Radially profiles the data, integrating the intensity in rings out from the centre. 
#   Unreliable as we approach the edges of the image as it just profiles the corners.
#   Less pixels there so become effectively zero after a certain point
    y, x = np.indices((diff_image.shape))
    r = np.sqrt((x - centre_x)**2 + (y - centre_y)**2)
    r = r.astype(int)       
    tbin =  np.bincount(r.ravel(), diff_image.ravel())
    nr = np.bincount(r.ravel())   
    radialProfile = tbin / nr

    return(radialProfile)

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

def _copy_fpd_metadata_to_radial_profile(signal, signal_radial):
    signal_radial.axes_manager[0].scale = signal.axes_manager[0].scale
    signal_radial.axes_manager[1].scale = signal.axes_manager[1].scale
    signal_radial.axes_manager[0].units = signal.axes_manager[0].units
    signal_radial.axes_manager[1].units = signal.axes_manager[1].units
    signal_radial.axes_manager[0].name = signal.axes_manager[0].name
    signal_radial.axes_manager[1].name = signal.axes_manager[1].name

    s_meta_micro = signal.metadata.Microscope
    signal_radial.metadata.add_node("Microscope")
    signal_radial.metadata.Microscope['Voltage'] = s_meta_micro['Voltage']
    signal_radial.metadata.Microscope['Camera length'] = s_meta_micro['Camera length']

def _set_radial_profile_axis_metadata(signal_radial):
    axis_m = signal_radial.axes_manager[-1]
    axis_m.name = "Scattering angle"

def _get_radial_profile_signal(signal):
    """Radially integrates a 4-D pixelated STEM diffraction signal.

    Parameters
    ----------
    signal : 4-D HyperSpy signal
        First two axes: 2 spatial dimensions.
        Last two axes: 2 reciprocal dimensions.

    Returns
    -------
    3-D HyperSpy signal, 2 spatial dimensions,
    1 integrated reciprocal dimension."""

    com_x_array, com_y_array = _get_disk_centre_from_signal(
            signal, threshold=1.)
    
    radial_profile_array = np.zeros(signal.data.shape[0:-1], dtype=np.float64)
    diff_image = np.zeros(signal.data.shape[2:], dtype=np.uint16)
    for x in range(signal.axes_manager[0].size):
        for y in range(signal.axes_manager[1].size):
            diff_image[:] = signal.data[x,y,:,:]
            centre_x = com_x_array[x,y]
            centre_y = com_y_array[x,y]
            radial_profile = _get_radial_profile_of_diff_image(
                    diff_image, centre_x, centre_y)
            radial_profile_array[x,y,0:len(radial_profile)] = radial_profile

    lowest_radial_zero_index = _get_lowest_index_radial_array(
            radial_profile_array)
    signal_radial = hs.signals.Spectrum(
            radial_profile_array[:,:,0:lowest_radial_zero_index])

    _copy_fpd_metadata_to_radial_profile(signal, signal_radial)
    _set_radial_profile_axis_metadata(signal_radial)
    return(signal_radial)

def _set_radial_scale(
        s_radial, 
        mrad_per_pixel=None, 
        background_range=None,
        gaussian_range=None,
        sto_region=None):
    """Set the scaling for radially integrated STEM diffraction
    dataset. Can either get a value as input, or fit a known
    Laue Zone peak with a Gaussian.
    
    Parameters
    ----------
    s_radial : HyperSpy spectrum signal
        The signal must be a 3-D dataset, with the two first
        axes being the spatial probe dimensions and the last
        the radially intergrated scattering angles.
    mrad_per_pixel : number, optional
        Calibration scale for the scattering angles, if a number
        is given this will be used as scale. If not, a Gaussian
        will be fitted to a known Laue Zone peak using the
        background_range, gaussian_range and sto_region
        parameters.
    background_range : (int, int), optional
        The index range for fitting the powerlaw background
        right before the Gaussian.
    gaussian_range : (int, int), optional
        Range for fitting a Gaussian to the Laue Zone peak.
    sto_region : (int, int, int, int), optional
        Calibration region with the known Laue Zone peak.

    Examples
    --------
    # Setting the scale directly 
    >>>> _set_radial_scale(s_radial, mrad_per_pixel=0.1)

    # Setting the scale by fitting a Gaussian:
    >>>> _set_radial_scale(
            s_radial, 
            background_range=(100, 120), 
            gaussian_range=(120,130),
            sto_region=(0,10,-2,-1))
    """
    if mrad_per_pixel == None:
        s_sto = s_radial.inav[
                sto_region[0]:sto_region[1],
                sto_region[2]:sto_region[3]].isig[
                        background_range[0]:gaussian_range[1]]
        mrad_per_pixel = _get_sto_calibration_gaussian(
            s_sto,
            background_range,
            gaussian_range)
    s_radial.axes_manager[-1].scale = mrad_per_pixel

def _get_sto_calibration_gaussian(
        signal_radial,
        background_range,
        gaussian_range):
    """Get scaling for radially integrated dataset using
    HyperSpy modelling with a powerlaw background and 
    Gaussian to fit the SrTiO3 Laue Zone peak."""
    m = lzp.model_radial_profile_data_with_gaussian(
            signal_radial,
            background_range,
            gaussian_range)
    centre_mean = m['Gaussian'].centre.as_signal().data.mean()
    if      (centre_mean < gaussian_range[0]) or\
            (centre_mean > gaussian_range[1]):
        raise Exception(
            "The Gaussian mean centre is outside the gaussian\
            fitting range. Something probably went very wrong\
            during the calibration fitting. Most likely due\
            to the wrong region being used as calibration,\
            or due to the background or Gaussian regions\
            being wrong")

    mrad_per_pixel = 95.378/centre_mean
    return(mrad_per_pixel)

def _get_sto_index_ranges_for_cameralengths(camera_length):
    """Returns indicies used for background subtraction 
    and Laue Zone fitting for SrTiO3 acquired on the Medipix
    for specific camera lengths."""

    if camera_length == 400:
        sto_background_range = (113, 140)
        sto_gaussian_range = (143, 153)
    else:
        raise Exception(
            "There is no calibration data for this camera length\
            find the indicies for background and Gaussian and\
            add them to _get_sto_index_ranges_for_cameralengths")
    calibration_dict = {}
    calibration_dict['background_range'] = sto_background_range
    calibration_dict['gaussian_range'] = sto_gaussian_range
    return(calibration_dict)

def save_fpd_dataset_as_radial_profile_signal(
        filename,
        mrad_per_pixel=None,
        background_range=None,
        gaussian_range=None,
        calibration_region=None):
    """Make a radially integrated HyperSpy HDF5 spectrum
    file from fpd 4-D diffraction HDF5. Uses center of mass
    to find the centre of the diffraction disk, and radial
    integration to reduce the 2-D diffraction images to 
    1-D scattering angle intensity. Calibrates the scattering
    angle either by a given value, or by fitting a Gaussian
    to the SrTiO3 Laue Zone peak.
    
    Parameters
    ----------
    filename : string
        Filename of fpd HDF5 file.
    mrad_per_pixel : number, optional
        Calibration scale for the scattering angles, if a number
        is given this will be used as scale. If not, a Gaussian
        will be fitted to a known Laue Zone peak using the
        background_range, gaussian_range and sto_region
        parameters.
    background_range : (int, int), optional
        The index range for fitting the powerlaw background
        right before the Gaussian. If not set, indicies from
        known camera lengths will be tried, but this might
        not work properly.
    gaussian_range : (int, int), optional
        Range for fitting a Gaussian to the Laue Zone peak.
        If not set, indicies from known camera lengths will 
        be tried, but this might not work properly.
    calibration_region : (int, int, int, int), optional
        Calibration region with the known Laue Zone peak.
        If this is not set, the highest y-axis row will be
        used, which will work for most datasets acquired 
        on the ARM200CF. However, this might go horribly
        wrong.

    Examples
    --------
    # Setting the scale directly 
    >>>> save_fpd_dataset_as_radial_profile_signal(
            "default.hdf5", 
            mrad_per_pixel=0.1)

    # Setting the scale by fitting a Gaussian:
    >>>> save_fpd_dataset_as_radial_profile_signal(
            "default.hdf5", 
            background_range=(100, 120), 
            gaussian_range=(120,130),
            sto_region=(0,10,-2,-1))
    """
    s_fpd = _load_fpd_dataset(filename)
    s_radial = _get_radial_profile_signal(s_fpd)
    if mrad_per_pixel == None:
        if background_range == None:
            calibration_dict = _get_sto_index_ranges_for_cameralengths(
                    s_fpd.metadata.Microscope['Camera length'])
            background_range = calibration_dict['background_range']
        if gaussian_range == None:
            calibration_dict = _get_sto_index_ranges_for_cameralengths(
                    s_fpd.metadata.Microscope['Camera length'])
            gaussian_range = calibration_dict['gaussian_range']

        if calibration_region == None:
            warnings.warn(
                "calibration_region is not set, and the last y-axis\
                row will be used. This might lead to the calibration\
                going horribly wrong.")
            calibration_region = (0,-1,-2,-1)

        _set_radial_scale(
                s_radial, 
                mrad_per_pixel=None, 
                background_range=background_range,
                gaussian_range=gaussian_range,
                sto_region=calibration_region)
    else:
        _set_radial_scale(
                s_radial, 
                mrad_per_pixel=mrad_per_pixel)
    
    new_filename = filename.replace(".hdf5","_radial.hdf5")
    s_radial.save(new_filename, overwrite=True)
