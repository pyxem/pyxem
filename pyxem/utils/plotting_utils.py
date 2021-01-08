import matplotlib.pyplot as plt
import numpy as np
from pyxem.utils.polar_transform_utils import (
    get_template_cartesian_coordinates,
    get_template_polar_coordinates,
    image_to_polar,
)
from pyxem.utils.expt_utils import find_beam_center_blur


def plot_template_over_pattern(
    pattern,
    simulation,
    ax=None,
    in_plane_angle=0.0,
    max_r=None,
    find_direct_beam=True,
    direct_beam_position=None,
    coordinate_system="cartesian",
    marker_color="red",
    marker_type="x",
    size_factor=1.0,
    **kwargs
):
    """
    A quick utility function to plot a simulated pattern over an experimental image

    Parameters
    ----------
    pattern : 2D np.ndarray
        The diffraction pattern
    simulation : diffsims.sims.diffraction_simulation.DiffractionSimulation
        The simulated diffraction pattern. It must be calibrated.
    axis : matplotlib.AxesSubplot, optional
        An axis object on which to plot. If None is provided, one will be created.
    in_plane_angle : float, optional
        An in-plane rotation angle to apply to the template in degrees
    max_r : float, optional
        Maximum radius to consider in the polar transform in pixel coordinates.
        Will only influence the result if `coordinate_system`="polar".
    find_direct_beam: bool, optional
        Roughly find the optimal direct beam position if it is not centered.
    direct_beam_position: 2-tuple
        The (x, y) position of the direct beam in pixel coordinates. Takes
        precedence over `find_direct_beam`
    coordinate_system : str, optional
        Type of coordinate system to plot the image and template in. Either
        `cartesian` or `polar`
    marker_color : str, optional
        Color of the spot markers
    marker_type : str, optional
        Type of marker used for the spots
    size_factor : float, optional
        Scaling factor for the spots. See notes on size.
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
    if ax is None:
        _, ax = plt.subplots()
    if direct_beam_position is not None:
        c_x, c_y = direct_beam_position
    elif find_direct_beam:
        c_y, c_x = find_beam_center_blur(pattern, 1)
    else:
        c_y, c_x = pattern.shape[0] // 2, pattern.shape[1] // 2
    if coordinate_system == "polar":
        pattern = image_to_polar(
            pattern,
            max_r=max_r,
            find_direct_beam=find_direct_beam,
            direct_beam_position=direct_beam_position,
        )
        x, y, intensities = get_template_polar_coordinates(
            simulation, in_plane_angle=in_plane_angle, max_r=max_r
        )
    elif coordinate_system == "cartesian":
        x, y, intensities = get_template_cartesian_coordinates(
            simulation,
            center=(c_x, c_y),
            in_plane_angle=in_plane_angle,
            window_size=pattern.shape[::-1],
        )
    else:
        raise NotImplementedError(
            "Only polar and cartesian are accepted coordinate systems"
        )
    im = ax.imshow(pattern, **kwargs)
    sp = ax.scatter(
        x,
        y,
        s=size_factor * np.sqrt(intensities),
        marker=marker_type,
        color=marker_color,
    )
    return (ax, im, sp)
