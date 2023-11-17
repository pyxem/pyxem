import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from hyperspy.utils.markers import point
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
    find_direct_beam: bool = False,
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

    if n_best is None:
        n_best = result["template_index"].shape[2]

    if direct_beam_position is None and not find_direct_beam:
        direct_beam_position = (0, 0)

    if marker_colors is None:
        marker_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if len(marker_colors) < n_best:
        print(
            "Warning: not enough colors in `marker_colors` for `n_best` different colored marks. Colors will loop"
        )

    # Add markers as iterable, recommended in hyperspy docs.
    # Using a genenerator will hopefully reduce memory.
    # To avoid scope errors, pass all variables as inputs
    def _get_markers_iter(
        signal,
        library,
        result,
        phase_key_dict,
        n_best,
        find_direct_beam,
        direct_beam_position,
        marker_colors,
        marker_type,
        size_factor,
        verbose,
    ):
        # Hyperspy wants one marker for all pixels in the navigation space,
        # so we generate all the data for a given solution and then yield them

        # Allocate space for all navigator pixels to potentially have the maximum amount of simulated diffraction spots
        max_marker_count = max(
            sim.intensities.size
            for lib in library.values()
            for sim in lib["simulations"]
        )

        shape = (
            signal.axes_manager[1].size,
            signal.axes_manager[0].size,
            max_marker_count,
        )

        # Explicit zeroes instead of empty, since we won't fill all elements in the final axis
        marker_data_x = np.zeros(shape)
        marker_data_y = np.zeros(shape)
        marker_data_i = np.zeros(shape)

        for n in range(n_best):
            color = marker_colors[n % len(marker_colors)]

            # Generate data for a given solution index.
            x_iter = range(signal.axes_manager[0].size)
            if verbose:
                x_iter = tqdm(x_iter)

            for px in x_iter:
                for py in range(signal.axes_manager[1].size):
                    sim_sol_index = result["template_index"][py, px, n]
                    mirrored_sol = result["mirrored_template"][py, px, n]
                    in_plane_angle = result["orientation"][py, px, n, 0]

                    phase_key = result["phase_index"][py, px, n]
                    phase = phase_key_dict[phase_key]
                    simulations = library[phase]["simulations"]
                    pattern = simulations[sim_sol_index]

                    if find_direct_beam:
                        x, y = find_beam_center_blur(signal.inav[px, py], 1)

                        # The result of `find_beam_center_blur` is in a corner.
                        # Move to center of image
                        x -= signal.axes_manager[2].size // 2
                        y -= signal.axes_manager[3].size // 2
                        direct_beam_position = (x, y)

                    x, y, intensities = get_template_cartesian_coordinates(
                        pattern,
                        center=direct_beam_position,
                        in_plane_angle=in_plane_angle,
                        mirrored=mirrored_sol,
                    )

                    x *= signal.axes_manager[2].scale
                    y *= signal.axes_manager[3].scale

                    marker_count = len(x)
                    marker_data_x[py, px, :marker_count] = x
                    marker_data_y[py, px, :marker_count] = y
                    marker_data_i[py, px, :marker_count] = intensities

            marker_kwargs = {
                "color": color,
                "marker": marker_type,
                "label": f"Solution index: {n}",
            }

            # Plot for the given solution index
            for i in range(max_marker_count):
                yield point(
                    marker_data_x[..., i],
                    marker_data_y[..., i],
                    size=4 * np.sqrt(marker_data_i[..., i]) * size_factor,
                    **marker_kwargs,
                )
                # We only need one set of labels per solution
                if i == 0:
                    marker_kwargs.pop("label")

    signal.plot(**plot_kwargs)
    signal.add_marker(
        _get_markers_iter(
            signal,
            library,
            result,
            phase_key_dict,
            n_best,
            find_direct_beam,
            direct_beam_position,
            marker_colors,
            marker_type,
            size_factor,
            verbose,
        )
    )
    plt.gcf().legend()
