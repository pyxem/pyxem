import numpy as np
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.units as units


def get_azimuthal_integrator(detector, detector_distance, shape, center=None, affine=None, mask=None, **kwargs):
    """ This is a basic method for creating a azimuthal integrator.

    This helps to deal with taking some of the pyXEM standards and apply them to pyFAI

    Parameters
    -----------
    detector: pyFAI.detectors.Detector
        The detector to be integrated
    detector_distance:
        distance sample - detector plan (orthogonal distance, not along the beam), in meter.
    shape: (int, int)
        The shape of the signal we are operating on. For the
    center: (float, float)
        The center of the diffraction pattern
    affine: (3x3)
        The affine transformation to apply to the data
    kwargs: dict
        Any additional arguments to the Azimuthal Integrator class
    """
    if center is None:
        center = shape/ 2  # Center is middle of the image
    if affine is not None:
        dx, dy = _get_displacements(center=center, shape=shape,affine=affine)  # creating spline
        detector.set_dx(dx)
        detector.set_dy(dy)
    ai = AzimuthalIntegrator(detector=detector, **kwargs)
    if mask is not None:
        ai.set_mask(None)
    ai.setFit2D(directDist=detector_distance, centerX=center[0], centerY=center[1])
    return ai


def _get_radial_extent(ai, shape=None, unit=None):
    """This method isn't perfect but it takes some Azimuthal Integrator and calculates the domain of the output

    Parameters
    -----------
    ai: AzimuthalIntegrator
        The integrator to operate on
    shape: (int, int)
        The shape of the detector.
    unit:
        The unit to calculate the radial extent with.
    """
    postions = ai.array_from_unit(shape=shape, unit=unit, typ="center")
    return [np.min(postions), np.max(postions)]


def _get_displacements(center, shape, affine):
    """ Gets the displacements for a set of points based on some affine transformation
    about some center point.

    Parameters
    -------------
    center: (tuple)
        The center to preform the affine transformation around
    shape: (tuple)
        The shape of the array
    affine: 3x3 array
        The affine transformation to apply to the image

    Returns
    ----------
    dx: np.array
        The displacement in the x direction of shape = shape
    dy: np.array
        The displacement in the y direction of shape = shape
    """
    difference=np.subtract(shape,center)
    xx,yy = np.mgrid[center[0]:difference[0],center[1]:difference[1]]  # all x and y coordinates on the grid
    coord = np.array([xx.flatten(), yy.flatten(), np.ones(shape[0]*shape[1])])
    corrected = np.matmul(coord.T, affine)
    dx = xx - corrected[:, 0]
    dy = yy - corrected[:, 1]
    return dx,dy