import numpy as np
from pyxem.utils.expt_utils import find_beam_center_blur
from scipy import ndimage
from pyxem.utils.cuda_utils import get_array_module


try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndigpu

    CUPY_INSTALLED = True
except ImportError:
    CUPY_INSTALLED = False
    cp = None
    ndigpu = None


""" These are designed to be fast and used for indexation, for data correction, see radial_utils"""


def _cartesian_positions_to_polar_nonround(x, y, delta_r=1.0, delta_theta=1.0):
    r = np.sqrt(x ** 2 + y ** 2) / delta_r
    theta = np.mod(np.rad2deg(np.arctan2(y, x)), 360) / delta_theta
    return r, theta


def _cartesian_positions_to_polar(x, y, delta_r=1, delta_theta=1):
    """
    Convert 2D cartesian image coordinates to polar image coordinates

    Parameters
    ----------
    x : 1D numpy.ndarray
        x coordinates
    y : 1D numpy.ndarray
        y coordinates
    delta_r : float
        sampling interval in the r direction
    delta_theta : float
        sampling interval in the theta direction

    Returns
    -------
    r : 1D numpy.ndarray
        r coordinate or x coordinate in the polar image
    theta : 1D numpy.ndarray
        theta coordinate or y coordinate in the polar image
    """
    r, theta = _cartesian_positions_to_polar_nonround(x, y, delta_r, delta_theta)
    r = np.rint(r).astype(np.int32)
    theta = np.rint(theta).astype(np.int32)
    return r, theta


def get_template_polar_coordinates(
    simulation,
    in_plane_angle=0.0,
    delta_r=1,
    delta_theta=1,
    max_r=None,
    mirrored=False,
    rounded=True,
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
        by delta_r = the width of the polar image.
    mirrored : bool, optional
        Whether to mirror the template
    rounded: bool, optional
        Whether the coordinates should be rounded to integers

    Returns
    -------
    r : np.ndarray
        The r coordinates of the diffraction spots in the template scaled by
        delta_r
    theta : np.ndarray
        The theta coordinates of the diffraction spots in the template, scaled
        by delta_theta
    intensities : np.ndarray
        The intensities of the diffraction spots with dtype float
    """
    x = simulation.calibrated_coordinates[:, 0]
    y = simulation.calibrated_coordinates[:, 1]
    intensities = simulation.intensities.astype(np.float64)
    r, theta = _cartesian_positions_to_polar_nonround(x, y, delta_r, delta_theta)
    if max_r is not None:
        condition = r < max_r
        r = r[condition]
        theta = theta[condition]
        intensities = intensities[condition]
    theta = np.mod(theta + in_plane_angle / delta_theta, 360 / delta_theta)
    if mirrored:
        theta = 360 / delta_theta - theta
    if rounded:
        theta = theta.astype(np.int32)
        r = r.astype(np.int32)
    return r, theta, intensities


def get_template_cartesian_coordinates(
    simulation,
    center=(0.0, 0.0),
    in_plane_angle=0.0,
    window_size=None,
    mirrored=False,
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
    mirrored : bool, optional
        Whether to mirror the template

    Returns
    -------
    x : np.ndarray
        x coordinates of the diffraction spots in the template in pixel units
    y : np.ndarray
        y coordinates of the diffraction spots in the template in pixel units
    intensities: np.ndarray
        intensities of the spots
    """
    ox = simulation.calibrated_coordinates[:, 0]
    oy = simulation.calibrated_coordinates[:, 1]
    if mirrored:
        oy = -oy
    intensities = simulation.intensities
    c = np.cos(np.deg2rad(in_plane_angle))
    s = np.sin(np.deg2rad(in_plane_angle))
    # rotate it
    x = c * ox - s * oy + center[0]
    y = s * ox + c * oy + center[1]
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
        r_dim = int(round(max_r / delta_r))
    theta_dim = int(round(360 / delta_theta))
    return (theta_dim, r_dim)


def _get_map_function(dispatcher):
    return ndimage.map_coordinates if dispatcher == np else ndigpu.map_coordinates


def _warp_polar_custom(
    image, center, radius, output_shape, order=1, precision=np.float64
):
    """
    Function to emulate warp_polar in skimage.transform on both CPU and GPU. Not all
    parameters are supported

    Parameters
    ----------
    image: numpy.ndarray or cupy.ndarray
        Input image. Only 2-D arrays are accepted.
    center: tuple (row, col)
        Point in image that represents the center of the transformation
        (i.e., the origin in cartesian space). Values can be of type float.
    radius: float
        Radius of the circle that bounds the area to be transformed.
    output_shape: tuple (row, col)
        Shape of the output polar image
    order: int, optional
        Order of interpolation between pixels
    precision: np.float64 or np.float32
        Double or single precision output

    Returns
    -------
    polar: numpy.ndarray or cupy.ndarray
        polar image of dtype float64

    Notes
    -----
    Speed gains on the GPU will depend on the size of your problem.
    On a Tesla V100 a 10x speed up was achieved with a 256x256 image:
    from 5 ms to 400 microseconds. For a 4000x4000 image a 1000x speed up
    was achieved: from 180 ms to 400 microseconds. However, this does not
    count the time to transfer data from the CPU to the GPU and back.
    """
    dispatcher = get_array_module(image)
    cy, cx = center
    H = output_shape[0]
    W = output_shape[1]
    T = dispatcher.linspace(0, 2 * dispatcher.pi, H).reshape(H, 1)
    R = dispatcher.linspace(0, radius, W).reshape(1, W)
    X = R * dispatcher.cos(T) + cx
    Y = R * dispatcher.sin(T) + cy
    coordinates = dispatcher.stack([Y, X])
    map_function = _get_map_function(dispatcher)
    polar = map_function(image.astype(precision), coordinates, order=order)
    return polar


def image_to_polar(
    image,
    delta_r=1.0,
    delta_theta=1.0,
    max_r=None,
    find_direct_beam=False,
    direct_beam_position=None,
    order=1,
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
    order : int
        Order of interpolation for the polar transform

    Returns
    -------
    polar_image : 2D numpy.ndarray or cupy.ndarray
        Array representing the polar transform of the image with shape
        (360/delta_theta, max_r/delta_r)
    """
    half_x, half_y = image.shape[1] / 2, image.shape[0] / 2
    if direct_beam_position is not None:
        c_x, c_y = direct_beam_position
    elif find_direct_beam:
        c_x, c_y = find_beam_center_blur(image, 1)
    else:
        c_x, c_y = half_x, half_y
    output_shape = get_polar_pattern_shape(
        image.shape, delta_r, delta_theta, max_r=max_r
    )
    maximum_radius = output_shape[1] * delta_r
    return _warp_polar_custom(
        image,
        center=(c_y, c_x),
        output_shape=output_shape,
        radius=maximum_radius,
        order=order,
    )


def _chunk_to_polar(
    images,
    center,
    radius,
    output_shape,
    precision=np.float64,
):
    """
    Convert a chunk of images to polar coordinates

    Parameters
    ----------
    images : (scan_x, scan_y, x, y) np.ndarray or cp.ndarray
        diffraction patterns
    center : 2-Tuple of float
        center of the images in pixels (row, col) = (c_y, c_x)
    radius : float
        maximum radius to consider in the image. This gets mapped onto
        output_shape[1]
    output_shape : 2-Tuple of int
        (height, width) of the output polar images
    precision : np.float32 or np.float64

    Returns
    -------
    polar_chunk : (scan_x, scan_y, max_r/delta_theta, 360/delta_theta) np.ndarray or cp.ndarray
        diffraction patterns in polar coordinates. Returns a cuda or numpy array depending
        on the device
    """
    dispatcher = get_array_module(images)
    polar_chunk = dispatcher.empty(
        (images.shape[0], images.shape[1], output_shape[0], output_shape[1]),
        dtype=precision,
    )
    for idx, idy in np.ndindex(images.shape[:-2]):
        polar_chunk[idx, idy] = _warp_polar_custom(
            images[idx, idy],
            center=center,
            radius=radius,
            output_shape=output_shape,
        )
    return polar_chunk
