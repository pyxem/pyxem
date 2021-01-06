import numpy as np
from numba import njit, prange, objmode
from skimage.transform import warp_polar
from pyxem.utils.general_utils import get_direct_beam_center


def get_template_polar_coordinates(simulation, in_plane_angle=0., delta_r = 1, delta_theta=1):
    """
    Convert a single simulation to polar coordinates

    Parameters
    ----------
    simulation : diffsims.sims.diffraction_simulation.DiffractionSimulation
        The diffraction pattern
    in_plane_angle : float, optional
        Angle in degrees that the template should be rotated by
    delta_r : float, optional
        Scaling factor for r in case the points should align with a rescaled
        polar image.
    delta_that : float, optional
        Scaling factor for theta in case the points should align with a
        rescaled polar image.

    Returns
    -------
    r : np.ndarray
        The r coordinates of the diffraction spots in the template scaled by
        delta_r
    theta : np.ndarray
        The theta coordinates of the diffraction spots in the template, scaled
        by delta_theta
    """
    x = simulation.calibrated_coordinates[:,0]
    y = simulation.calibrated_coordinates[:,1]
    imag = x+1j*y
    r = abs(imag)
    theta = np.rad2deg(np.angle(imag))
    theta = np.mod(theta+in_plane_angle, 360)
    return r/delta_r, theta/delta_theta


def get_template_cartesian_coordinates(simulation, center=(0., 0.), in_plane_angle=0.):
    """
    Get the cartesian coordinates of the diffraction spots in a template

    Parameters
    ----------
    simulation : diffsims.sims.diffraction_simulation.DiffractionSimulation
        The diffraction pattern
    center : 2-tuple of ints, optional
        The coordinate of the direct beam in pixel coordinates
    in_plane_angle : float
        Angle in degrees representing an in-plane rotation of the template 
        around the direct beam

    Returns
    -------
    x : np.ndarray
        x coordinates of the diffraction spots in the template in pixel units
    y : np.ndarray
        y coordinates of the diffraction spots in the template in pixel units
    """
    r, theta = get_template_polar_coordinates(simulation, in_plane_angle, 1, 1)
    x = r*np.cos(np.deg2rad(theta)) + center[0]
    y = r*np.sin(np.deg2rad(theta)) + center[1]
    return x, y


def get_polar_pattern_shape(image_shape, delta_r, delta_theta):
    """
    Returns the (r, theta) shape of images if they would be transformed
    to polar coordinates.

    Parameters
    ----------
    image_shape: 2-Tuple
        (height, width) of the images in cartesian coordinates
    delta_r : float
        size of pixels in the r-direction, in units of cartesian pixels.
    delta_theta : float
        size of pixels in the theta-direction, in degrees
    """
    half_y = image_shape[0] // 2
    half_x = image_shape[1] // 2
    r_dim = int(np.ceil(np.sqrt(half_x**2 + half_y**2)) / delta_r)
    theta_dim = int(round(360 / delta_theta))
    return (theta_dim, r_dim)


def image_to_polar(image, delta_r=1., delta_theta=1, find_maximum=True, **kwargs):
    """Convert a single image to polar coordinates including the option to
    find the direct beam and take this as the center.

    Parameters
    ----------
    image : 2D numpy.ndarray
        Experimental image
    delta_r : float
        The radial increment, determines how many columns will be in the polar
        image.
    delta_theta : float
        The angular increment, determines how many rows will be in the polar
        image
    find_maximum : bool
        Whether to roughly find the direct beam in the center of the image. If
        false then the middle of the image will be the center for the polar
        transform.

    Returns
    -------
    polar_image : 2D numpy.ndarray
        Array representing the polar transform of the image with shape
        (360/delta_theta, r_max/delta_r) where r_max is the distance from the
        center of the image to the corner rounded up.
    """
    half_x, half_y = image.shape[1]//2, image.shape[0]//2
    if find_maximum:
        c_x, c_y = get_direct_beam_center(image, **kwargs)
    else:
        c_x, c_y = half_x, half_y
    output_shape = get_polar_pattern_shape(image.shape, delta_r, delta_theta)
    return warp_polar(image, center=(c_y, c_x), output_shape=output_shape)


@njit(nogil=True, parallel=True)
def _chunk_to_polar(images, delta_r, delta_theta, find_maximum):
    """
    Convert a chunk of images to polar coordinates

    The intensities in the polar images are scaled such that the mean is 0
    and the norm is 1.

    Parameters
    ----------
    images : (scan_x, scan_y, x, y) np.ndarray
        diffraction patterns
    delta_r : float
        size of pixels in r-direction of converted image, in units of
        cartesian pixels
    delta_theta : float
        size of pixels in the theta direction of converted image, in degrees
    find_maximum : bool
        optimize the direct beam position and take this as center for the 
        transform. If false, the center of the image will be taken.

    Returns
    -------
    polar_chunk : (scan_x, scan_y, r, theta) np.ndarray
        diffraction patterns in polar coordinates
    """
    half_x = images.shape[-1]//2
    half_y = images.shape[-2]//2
    r_dim = int(np.ceil(np.sqrt(half_x**2 + half_y**2))/delta_r)
    theta_dim = int(round(360/delta_theta))
    polar_chunk = np.empty((images.shape[0], images.shape[1], theta_dim, r_dim), dtype=np.float64)
    for idx in prange(images.shape[0]):
        for idy in prange(images.shape[1]):
            image = images[idx, idy]
            with objmode(polar_image='float64[:,:]'):
                polar_image = image_to_polar(image, delta_r=delta_r,
                                             delta_theta=delta_theta, find_maximum=find_maximum)
            polar_image = polar_image - np.mean(polar_image)
            polar_image = polar_image / np.sqrt(np.sum(polar_image**2))
            polar_chunk[idx, idy] = polar_image
    return polar_chunk
