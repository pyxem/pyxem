import numpy as np
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.detectors import Detector


def get_azimuthal_integrator(
    detector,
    detector_distance,
    shape,
    center=None,
    affine=None,
    mask=None,
    wavelength=None,
    **kwargs
):
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
    mask: np.array
        A boolean array to be added to the integrator.
    wavelength: float
        The wavelength of the beam in meter. Needed to accounting for the
        Ewald sphere. 
    kwargs: dict
        Any additional arguments to the Azimuthal Integrator class
    """
    if center is None:
        center = np.divide(shape, 2)  # Center is middle of the image
    if affine is not None:
        # create spline representation with (dx,dy) displacements
        dx, dy = _get_displacements(center=center, shape=shape, affine=affine)
        detector.max_shape = shape
        detector.shape = shape
        detector.set_dx(dx)
        detector.set_dy(dy)
    ai = AzimuthalIntegrator(
        detector=detector, dist=detector_distance, wavelength=wavelength, **kwargs
    )
    if mask is not None:
        ai.set_mask(mask)
    if wavelength is not None:
        ai.wavelength = wavelength
    ai.setFit2D(
        directDist=detector_distance * 1000, centerX=center[1], centerY=center[0]
    )
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
    # all x and y coordinates on the grid
    shape_plus = np.add(shape, 1)
    x = range(shape_plus[0], 0, -1)
    y = range(shape_plus[1], 0, -1)
    xx, yy = np.meshgrid(x, y)
    xx = np.subtract(xx, center[1])
    yy = np.subtract(yy, center[0])
    coord = np.array(
        [xx.flatten(), yy.flatten(), np.ones((shape_plus[0]) * (shape_plus[1]))]
    )
    corrected = np.reshape(np.matmul(coord.T, affine), newshape=(*shape_plus, -1))
    dx = xx - corrected[:, :, 0]
    dy = yy - corrected[:, :, 1]
    return dx, dy


def _get_setup(wavelength, pyxem_unit, pixel_scale, radial_range=None):
    """Returns a generic set up for a flat detector with accounting for Ewald sphere effects
    """
    units_table = {
        "2th_deg": [None, 1, "2th_deg"],
        "2th_rad": [None, 1, "2th_rad"],
        "q_nm^-1": [1e-9, 1, "q_nm^-1"],
        "q_A^-1": [1e-10, 1, "q_A^-1"],
        "k_nm^-1": [1e-9, 2 * np.pi, "q_nm^-1"],  # add to pyFAI
        "k_A^-1": [1e-10, 2 * np.pi, "q_A^-1"],  # add to pyFAI
    }
    wavelength_scale = units_table[pyxem_unit][0]
    scale_factor = units_table[pyxem_unit][1]
    unit = units_table[pyxem_unit][2]
    detector_distance = 1
    if wavelength_scale is None:
        if pyxem_unit == "2th_deg":
            pixel_1_size = np.tan((pixel_scale[0] / 180) * np.pi)
            pixel_2_size = np.tan((pixel_scale[1] / 180) * np.pi)
        if pyxem_unit == "2th_rad":
            pixel_1_size = np.tan(pixel_scale[0])
            pixel_2_size = np.tan(pixel_scale[1])
    else:
        pixel_1_size = (
            pixel_scale[0] * (wavelength / wavelength_scale) * detector_distance
        )
        pixel_2_size = (
            pixel_scale[1] * (wavelength / wavelength_scale) * detector_distance
        )
    detector = Detector(pixel1=pixel_1_size, pixel2=pixel_2_size)
    if radial_range is not None:
        radial_range = [radial_range[0] * scale_factor, radial_range[1] * scale_factor]
    return detector, detector_distance, radial_range, unit, scale_factor
