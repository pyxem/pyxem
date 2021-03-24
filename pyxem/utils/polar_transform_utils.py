import numpy as np
from numba import njit, prange, objmode
from skimage.transform import warp_polar
from pyxem.utils.expt_utils import find_beam_center_blur
from scipy import ndimage
from pyxem.utils.cuda_utils import is_cupy_array


try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndigpu
    CUPY_INSTALLED = True
except ImportError:
    CUPY_INSTALLED = False


""" These are designed to be fast and used for indexation, for data correction, see radial_utils"""


def get_template_polar_coordinates(
    simulation,
    in_plane_angle=0.0,
    delta_r=1,
    delta_theta=1,
    max_r=None,
):
    """
    Convert a single simulation to polar coordinates

    Parameters
    ----------
    simulation : diffsims.sims.diffraction_simulation.DiffractionSimulation
        The diffraction pattern template object
    in_plane_angle : float, optional
        In-plane angle in degrees that the template should be rotated by
    delta_r : float, optional
        Scaling factor for r in case the points should align with a rescaled
        polar image. If delta_r = 1 (default), then r is in units of pixels
        of the original image.
    delta_theta : float, optional
        Scaling factor for theta in case the points should align with a
        rescaled polar image. If delta_theta = 1 (default), then theta is in
        units of degrees.
    max_r : float, optional
        Maximum radial distance to consider. In units of pixels, not scaled
        by delta_r.

    Returns
    -------
    r : np.ndarray
        The r coordinates of the diffraction spots in the template scaled by
        delta_r
    theta : np.ndarray
        The theta coordinates of the diffraction spots in the template, scaled
        by delta_theta
    intensities : np.ndarray
        The intensities of the diffraction spots
    """
    x = simulation.calibrated_coordinates[:, 0]
    y = simulation.calibrated_coordinates[:, 1]
    intensities = simulation.intensities
    imag = x + 1j * y
    r = abs(imag)
    theta = np.rad2deg(np.angle(imag))
    theta = np.mod(theta + in_plane_angle, 360)
    if max_r is not None:
        condition = r < max_r
        r = r[condition]
        theta = theta[condition]
        intensities = intensities[condition]
    return r / delta_r, theta / delta_theta, intensities


def get_template_cartesian_coordinates(
    simulation, center=(0.0, 0.0), in_plane_angle=0.0, window_size=None
):
    """
    Get the cartesian coordinates of the diffraction spots in a template

    Parameters
    ----------
    simulation : diffsims.sims.diffraction_simulation.DiffractionSimulation
        The diffraction pattern
    center : 2-tuple of ints, optional
        The coordinate of the direct beam in pixel coordinates
    in_plane_angle : float, optional
        Angle in degrees representing an in-plane rotation of the template
        around the direct beam
    window_size : 2-tuple, optional
        Only return the reflections within the (width, height) in pixels
        of the image
    intensities : np.ndarray
        The intensities of the diffraction spots

    Returns
    -------
    x : np.ndarray
        x coordinates of the diffraction spots in the template in pixel units
    y : np.ndarray
        y coordinates of the diffraction spots in the template in pixel units
    """
    r, theta, intensities = get_template_polar_coordinates(
        simulation, in_plane_angle, 1, 1
    )
    x = r * np.cos(np.deg2rad(theta)) + center[0]
    y = r * np.sin(np.deg2rad(theta)) + center[1]
    if window_size is not None:
        condition = (x < window_size[0]) & (y < window_size[1]) & (y >= 0) & (x >= 0)
        x = x[condition]
        y = y[condition]
        intensities = intensities[condition]
    return x, y, intensities


def get_polar_pattern_shape(image_shape, delta_r, delta_theta, max_r=None):
    """
    Returns the shape of images if they would be transformed
    to polar coordinates.

    Parameters
    ----------
    image_shape: 2-Tuple
        (height, width) of the images in cartesian coordinates
    delta_r : float
        size of pixels in the r-direction, in units of cartesian pixels.
    delta_theta : float
        size of pixels in the theta-direction, in degrees
    max_r : float, optional
        maximum radius of the polar image, by default the cartesian distance
        in pixels from the center of the image to the corner, rounded up to
        the nearest integer.

    Returns
    -------
    theta_dim : int
        size of the theta dimension
    r_dim : int
        size of the r dimension
    """
    if max_r is None:
        half_y = image_shape[0] / 2
        half_x = image_shape[1] / 2
        r_dim = int(np.ceil(np.sqrt(half_x ** 2 + half_y ** 2)) / delta_r)
    else:
        r_dim = int(max_r / delta_r)
    theta_dim = int(round(360 / delta_theta))
    return (theta_dim, r_dim)


def _warp_polar_custom(image, center=None, radius=None, output_shape=None):
    """
    Function to emulate warp_polar in skimage.transform on both CPU and GPU. Not all
    parameters are supported
    
    Parameters
    ----------
    image: numpy.ndarray or cupy.ndarray
        Input image. Only 2-D arrays are accepted.         
    center: tuple (row, col), optional
        Point in image that represents the center of the transformation
        (i.e., the origin in cartesian space). Values can be of type float.
        If no value is given, the center is assumed to be the center point of the image.
    radius: float, optional
        Radius of the circle that bounds the area to be transformed.
    output_shape: tuple (row, col), optional

    Returns
    -------
    polar: numpy.ndarray or cupy.ndarray
        polar image

    Notes
    -----
    Speed gains on the GPU will depend on the size of your problem.
    On a Tesla V100 a 10x speed up was achieved with a 256x256 image:
    from 5 ms to 400 microseconds. For a 4000x4000 image a 1000x speed up
    was achieved: from 180 ms to 400 microseconds. However, this does not
    count the time to transfer data from the CPU to the GPU and back.
    """
    if is_cupy_array(image):
        dispatcher = cp
        map_function = ndigpu.map_coordinates
    else:
        dispatcher = np
        map_function = ndimage.map_coordinates
    if radius is None:
        radius = int(np.ceil(np.sqrt((image.shape[0] / 2)**2 + (image.shape[1] / 2)**2)))
    cx, cy = image.shape[1] // 2, image.shape[0] // 2
    if center is not None:
        cx, cy = center
    if output_shape is None:
        output_shape = (360, radius)
    delta_theta = 360 / output_shape[0]
    delta_r = radius / output_shape[1]
    t = dispatcher.arange(output_shape[0])
    r = dispatcher.arange(output_shape[1])
    R, T = dispatcher.meshgrid(r, t)
    X = R * delta_r * dispatcher.cos(dispatcher.deg2rad(T * delta_theta)) + cx
    Y = R * delta_r * dispatcher.sin(dispatcher.deg2rad(T * delta_theta)) + cy
    coordinates = dispatcher.stack([Y, X])
    polar = map_function(image, coordinates, order=1)
    return polar


def image_to_polar(
    image,
    delta_r=1.0,
    delta_theta=1.0,
    max_r=None,
    find_direct_beam=False,
    direct_beam_position=None,
):
    """Convert a single image to polar coordinates including the option to
    find the direct beam and take this as the center.

    Parameters
    ----------
    image : 2D numpy.ndarray or 2D cupy.ndarray
        Experimental image
    delta_r : float, optional
        The radial increment, determines how many columns will be in the polar
        image.
    delta_theta : float, optional
        The angular increment, determines how many rows will be in the polar
        image
    max_r : float, optional
        The maximum radial distance to include, in units of pixels. By default
        this is the distance from the center of the image to a corner in pixel
        units and rounded up to the nearest integer
    find_direct_beam : bool
        Whether to roughly find the direct beam using `find_beam_center_blur`
        using a gaussian smoothing with sigma = 1. If False then the middle of
        the image will be the center for the polar transform.
    direct_beam_position : 2-tuple
        (x, y) position of the central beam in pixel units. This overrides
        the automatic find_maximum parameter if set to True.

    Returns
    -------
    polar_image : 2D numpy.ndarray
        Array representing the polar transform of the image with shape
        (360/delta_theta, max_r/delta_r)
    """
    half_x, half_y = image.shape[1] / 2, image.shape[0] / 2
    if direct_beam_position is not None:
        c_x, c_y = direct_beam_position
    elif find_direct_beam:
        c_y, c_x = find_beam_center_blur(image, 1)
    else:
        c_x, c_y = half_x, half_y
    output_shape = get_polar_pattern_shape(
        image.shape, delta_r, delta_theta, max_r=max_r
    )
    return _warp_polar_custom(
        image,
        center=(c_y, c_x),
        output_shape=output_shape,
        radius=max_r,
    )



def chunk_to_polar(
    images,
    delta_r=1,
    delta_theta=1,
    max_r=None,
    find_direct_beam=False,
    direct_beam_positions=None,
):
    """
    Convert a chunk of images to polar coordinates

    Parameters
    ----------
    images : (scan_x, scan_y, x, y) np.ndarray or cp.ndarray
        diffraction patterns
    delta_r : float, optional
        size of pixels in r-direction of converted image, in units of
        cartesian pixels
    delta_theta : float, optional
        size of pixels in the theta direction of converted image, in degrees
    max_r : float, optional
        The maximum radial distance to include, in units of pixels. By default
        this is the distance from the center of the image to a corner in pixel
        units and rounded up to the nearest integer
    find_direct_beam : bool, optional
        optimize the direct beam position and take this as center for the
        transform. If false, the center of the image will be taken.
    direct_beam_positions : np.ndarray of shape (2,) or (scan_x, scan_y, 2)
        The (x, y) positions for the direct beam in each diffraction pattern.
        If only one (x, y) position is supplied, this will be mapped to all
        images. Overrides the find_maximum parameter

    Returns
    -------
    polar_chunk : (scan_x, scan_y, max_r/delta_theta, 360/delta_theta) np.ndarray or cp.ndarray
        diffraction patterns in polar coordinates. Returns a cuda or numpy array depending
        on the device
    """
    if is_cupy_array(images):
        dispatcher = cp
    else:
        dispatcher = np

    if direct_beam_positions is None:
        direct_beam_positions = np.empty(images.shape[:-2], dtype=object)
        direct_beam_positions.fill(None)
    elif len(direct_beam_positions) == 2:
        direct_beam_positions = np.array(direct_beam_positions).flatten()
        direct_beam_positions = np.tile(
            direct_beam_positions[np.newaxis, np.newaxis, ...], (*images.shape[:-2], 1)
        )
    else:
        if direct_beam_positions.ndim != 3:
            raise ValueError("direct_beam_positions must be 3-dimensional")

    pimage_shape = get_polar_pattern_shape(
        images.shape[-2:], delta_r, delta_theta, max_r=max_r
    )
    polar_chunk = dispatcher.empty(
        (images.shape[0], images.shape[1], pimage_shape[0], pimage_shape[1]),
        dtype=np.float32,
    )
    for idx, idy in np.ndindex(images.shape[:-2]):
        image = images[idx, idy]
        dbp = direct_beam_positions[idx, idy]
        polar_image = image_to_polar(
            image,
            delta_r=delta_r,
            delta_theta=delta_theta,
            max_r=max_r,
            find_direct_beam=find_direct_beam,
            direct_beam_position=dbp,
        )
        polar_chunk[idx, idy] = polar_image
    return polar_chunk
