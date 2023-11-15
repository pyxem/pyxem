from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import hyperspy.api as hs
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
    mirrored_template=False,
    coordinate_system="cartesian",
    marker_color="red",
    marker_type="x",
    size_factor=1.0,
    **kwargs,
):
    """A quick utility function to plot a simulated pattern over an experimental image

    Parameters
    ----------
    pattern : 2D np.ndarray
        The diffraction pattern
    simulation : :class:`~diffsims.sims.diffraction_simulation.DiffractionSimulation`
        The simulated diffraction pattern. It must be calibrated.
    axis : matplotlib.AxesSubplot, optional
        An axis object on which to plot. If None is provided, one will be created.
    in_plane_angle : float, optional
        An in-plane rotation angle to apply to the template in degrees
    max_r : float, optional
        Maximum radius to consider in the polar transform in pixel coordinates.
        Will only influence the result if ``coordinate_system="polar"``.
    find_direct_beam: bool, optional
        Roughly find the optimal direct beam position if it is not centered.
    direct_beam_position: 2-tuple
        The (x, y) position of the direct beam in pixel coordinates. Takes
        precedence over `find_direct_beam`
    mirrored_template: bool, optional
        Whether to mirror the given template
    coordinate_system : str, optional
        Type of coordinate system to plot the image and template in. Either
        ``cartesian`` or ``polar``
    marker_color : str, optional
        Color of the spot markers
    marker_type : str, optional
        Type of marker used for the spots
    size_factor : float, optional
        Scaling factor for the spots. See notes on size.
    **kwargs :
        See :meth:`~matplotlib.pyplot.imshow`

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
    if coordinate_system == "polar":
        pattern = image_to_polar(
            pattern,
            max_r=max_r,
            find_direct_beam=find_direct_beam,
            direct_beam_position=direct_beam_position,
        )
        x, y, intensities = get_template_polar_coordinates(
            simulation,
            in_plane_angle=in_plane_angle,
            max_r=max_r,
            mirrored=mirrored_template,
        )
    elif coordinate_system == "cartesian":
        if direct_beam_position is not None:
            c_x, c_y = direct_beam_position
        elif find_direct_beam:
            c_x, c_y = find_beam_center_blur(pattern, 1)
        else:
            c_y, c_x = pattern.shape[0] / 2, pattern.shape[1] / 2
        x, y, intensities = get_template_cartesian_coordinates(
            simulation,
            center=(c_x, c_y),
            in_plane_angle=in_plane_angle,
            window_size=(pattern.shape[1], pattern.shape[0]),
            mirrored=mirrored_template,
        )
        y = pattern.shape[0] - y
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


def plot_templates_over_signal(
    signal,
    library,
    result: dict,
    phase_key_dict: dict,
    n_best: int = None,
    direct_beam_position: tuple[int, int] = None,
    marker_colors: list[str] = None,
    marker_type: str = "x",
    size_factor: float = 1.0,
    verbose: bool = True,
    **plot_kwargs,
):
    """
    Display an interactive plot of the diffraction signal,
    with simulated diffraction patterns corresponding to template matching results displayed on top.

    Parameters
    ----------
    signal : hyperspy.signals.Signal2D
        The 4D-STEM dataset.
    library : diffsims.libraries.diffraction_library.DiffractionLibrary
        The library of simulated diffraction patterns.
    result : dict
        Template matching results dictionary containing keys: phase_index, template_index,
        orientation, correlation, and mirrored_template.
        Returned from pyxem.utils.indexation_utils.index_dataset_with_template_rotation.
    phase_key_dict: dictionary
        A small dictionary to translate the integers in the phase_index array
        to phase names in the original template library.
        Returned from pyxem.utils.indexation_utils.index_dataset_with_template_rotation.
    n_best : int, optional
        Number of solutions to plot. If None, defaults to all solutions.
    find_direct_beam: bool, optional
        Roughly find the optimal direct beam position if it is not centered.
    direct_beam_position: 2-tuple
        The (x, y) position of the direct beam in pixel coordinates. Takes
        precedence over `find_direct_beam`
    marker_colors : list of str, optional
        Colors of the spot markers. Should be at least n_best long, otherwise colors will loop.
        Defaults to matplotlib's default color cycle
    marker_type : str, optional
        Type of marker used for the spots
    size_factor : float, optional
        Scaling factor for the spots. See notes on size.
    **plot_kwargs : Keyword arguments passed to signal.plot

    Notes
    -----
    The spot marker sizes are scaled by the square root of their intensity
    """

    n_best_indexed = result["template_index"].shape[-1]

    if n_best is None:
        n_best = n_best_indexed
    
    if n_best > n_best_indexed:
        raise ValueError("`n_best` cannot be larger than the amount of indexed solutions")

    if direct_beam_position is None:
        direct_beam_position = (
            signal.axes_manager[0].size // 2,
            signal.axes_manager[1].size // 2,
            )

    if marker_colors is None:
        marker_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if len(marker_colors) < n_best:
        print(
            "Warning: not enough colors in `marker_colors` for `n_best` different colored marks. Colors will loop"
        )

    # Fetch an array of the results, with the correct phases
    result_array = np.empty(
        (signal.axes_manager[0].size, signal.axes_manager[1].size, n_best_indexed),
        dtype=object,
    )

    for phase_ind, phase in phase_key_dict.items():
        # Mask of where the results used the current phase
        mask = result["phase_index"] == phase_ind

        # Flat array of the simulations of the current phase
        phase_library_simulations = library[phase]["simulations"]

        # 2D array of simulations, using the indices from template matching.
        # These might use the wrong phase
        phase_library_simulations[result["template_index"]]

        # Use the correct phase
        result_array[mask] = phase_library_simulations[mask]

    result_signal = hs.signals.Signal1D(result_array)
    orientation_signal = hs.signals.Signal2D(result["orientation"])
    mirrored_template_signal = hs.signals.Signal1D(result["mirrored_template"])
    

    def marker_func_generator(n: int):

        def marker_func(pattern, center, orientation, mirrored_template):
            angle = orientation[n, 0]
            pattern = pattern[n]
            mirrored_template = mirrored_template[n]
            x, y, _ = get_template_cartesian_coordinates(pattern, center=center, in_plane_angle=angle, mirrored=mirrored_template)
            y = signal.axes_manager.shape[1] - y # See https://github.com/pyxem/pyxem/issues/925
            return np.array((x, y)).T
        
        # The inputs gets squeezed, this ensures any 1D inputs gets correcly accessed
        if n_best_indexed == 1:
            def marker_func_1D(pattern, center, orientation, mirrored_template):
                orientation = orientation.reshape(-1, 3)
                mirrored_template = np.array([mirrored_template])
                return marker_func(pattern, center, orientation, mirrored_template)
            return marker_func_1D
        else:
            return marker_func

    signal.plot(**plot_kwargs)
    for i in range(n_best):
        markers = result_signal.map(
            marker_func_generator(i), 
            center=direct_beam_position, 
            orientation=orientation_signal, 
            mirrored_template=mirrored_template_signal, 
            inplace=False, 
            ragged=True, 
            lazy_output=True
            )
        color = marker_colors[i % len(marker_colors)]
        m = hs.plot.markers.Points.from_signal(markers, color=color)
        signal.add_marker(m)
    plt.gcf().legend()
