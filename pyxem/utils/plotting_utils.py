import matplotlib.pyplot as plt
import numpy as np
from pyxem.utils.polar_transform_utils import (
        get_template_cartesian_coordinates,
        get_template_polar_coordinates,
        image_to_polar)
from pyxem.utils.general_utils import get_direct_beam_center


def plot_template(simulation, size_factor=1, units="real", **kwargs):
    """A quick-plot function for a simulated pattern"""
    _, ax = plt.subplots()
    ax.set_aspect("equal")
    if units == "real":
        coords = simulation.coordinates
    elif units == "pixel":
        coords = simulation.calibrated_coordinates
    else:
        raise NotImplementedError("Units not recognized, use real or pixel.")
    sp = ax.scatter(coords[:,0], coords[:,1],
               s = size_factor*np.sqrt(simulation.intensities), **kwargs)
    return ax, sp


def plot_template_over_pattern(pattern, simulation, in_plane_angle=0., find_direct_beam=True,
                               marker_color="red", marker_type="x", size_factor=1.,
                               coordinate_system="cartesian",
                               **kwargs):
    """
    A quick utility function to plot a simulated pattern over an experimental image

    Parameters
    ----------
    pattern : 2D np.ndarray
        The diffraction pattern
    simulation : diffsims.sims.diffraction_simulation.DiffractionSimulation
        The simulated diffraction pattern. It must be calibrated.
    in_plane_angle : float
        An in-plane rotation angle to apply to the template in degrees
    find_direct_beam: bool, optional
        Roughly find the optimal direct beam position if it is not centered.
    marker_color : str, optional
        Color of the spot markers
    marker_type : str, optional
        Type of marker used for the spots
    size_factor : float, optional
        Scaling factor for the spots. See notes on size.
    coordinate_system : str
        Type of coordinate system to plot the image and template in. Either
        `cartesian` or `polar`
    **kwargs : See imshow

    Returns
    -------
    ax : matplotlib.AxesSubplot
        The axes object
    im : matplotlib.image.AxesImage
        The representation of the image on the axes
    sp : matplotlib.collections.PathCollection
        The scatter plot representing the diffraction pattern

    Notes
    -----
    The spot marker sizes are scaled by the square root of their intensity
    """
    fig, ax = plt.subplots()
    if find_direct_beam:
        c_x, c_y = get_direct_beam_center(pattern)
    else:
        c_y, c_x = pattern.shape[0] // 2, pattern.shape[1] // 2
    if coordinate_system == "polar":
        pattern = image_to_polar(pattern, find_maximum=find_direct_beam)
        x, y = get_template_polar_coordinates(simulation, in_plane_angle=in_plane_angle)
    elif coordinate_system == "cartesian":
        x, y = get_template_cartesian_coordinates(simulation, center=(c_x, c_y), in_plane_angle=in_plane_angle)
    else:
        raise NotImplementedError("Only polar and cartesian are accepted coordinate systems")
    im = ax.imshow(pattern, **kwargs)
    sp = ax.scatter(x, y, s=size_factor * np.sqrt(simulation.intensities), marker=marker_type, color=marker_color)
    return (ax, im, sp)
