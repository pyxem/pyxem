import numpy as np
import scipy
import skimage

import numpy.typing as npt

from pyxem.data import si_phase
from pyxem import signals


def create_diffraction_pattern(
    simulation,
    shape: tuple = (512, 512),
    direct_beam_position: tuple = None,
    radius: int = 20,
    num_electrons: int = None,
    in_plane_angle: float = 0,
    calibration: float = 0.01,
    mirrored: bool = False,
    transformation_matrix: npt.NDArray = None,
):
    """
    Create a simulated (spot) diffraction pattern based on the provided simulation.

    Parameters
    ----------
    simulation: SimulationGenerator
        An instance of SimulationGenerator that contains the diffraction simulation data.
    shape: tuple
        The shape of the output diffraction pattern, e.g. (512, 512).
    direct_beam_position: tuple, optional
        The position of the direct beam in the diffraction pattern. If None, it defaults to the center of the shape.
    radius: int
        The radius of the disk used to simulate the diffraction spots in pixels.
    num_electrons: int, optional
        The number of electrons to simulate in the diffraction pattern. If None, no Poisson noise is applied.
    in_plane_angle: float
        The in-plane angle for rotating the diffraction pattern.
    calibration:
        The calibration factor for the diffraction pattern, in units of Angstroms per pixel.
    mirrored: bool
        If True, the diffraction pattern will be mirrored.
    transformation_matrix:
        A transformation matrix to apply to the diffraction pattern coordinates.
        If None, no transformation is applied.

    Returns
    -------
    numpy.ndarray
        A 2D array representing the simulated diffraction pattern.

    """
    if direct_beam_position is None:
        direct_beam_position = (shape[1] // 2, shape[0] // 2)
    transformed = simulation._get_transformed_coordinates(
        in_plane_angle,
        direct_beam_position,
        mirrored,
        units="pixel",
        calibration=calibration,
    )
    in_frame = (
        (transformed.data[:, 0] >= 0)
        & (transformed.data[:, 0] < shape[1])
        & (transformed.data[:, 1] >= 0)
        & (transformed.data[:, 1] < shape[0])
    )
    spot_coords = np.round(transformed.data[in_frame]).astype(int)
    if transformation_matrix is not None:
        direct_beam_position = direct_beam_position + (0,)
        spot_coords = (
            (spot_coords - direct_beam_position) @ transformation_matrix
        ) + direct_beam_position
    spot_intens = transformed.intensity[in_frame]
    pattern = np.zeros(shape)
    # checks that we have some spots
    if spot_intens.shape[0] == 0:
        return pattern
    else:
        for cord, inten in zip(spot_coords, spot_intens):
            rr, cc = skimage.draw.disk(cord[:2], radius, shape=shape)
            pattern[rr, cc] = inten
    if num_electrons is not None:
        total = np.sum(spot_intens) * radius**2 * np.pi
        pattern = np.random.poisson((pattern / total) * num_electrons)
    return np.divide(pattern, np.max(pattern))


def simulated_strain(
    navigation_shape: tuple = (32, 32),
    signal_shape: tuple = (512, 512),
    disk_radius: int = 20,
    num_electrons: int = 1e5,
    strain_matrix: npt.NDArray = None,
    lazy: bool = False,
):
    """
    Create a simulated strain map from a simulated diffraction pattern and a strain matrix.

    Parameters
    ----------
    navigation_shape: tuple
        The shape of the navigation axes, e.g. (32, 32).
    signal_shape: tuple
        The shape of the signal axes, e.g. (512, 512).
    disk_radius: int
        The radius of the disk used to create the diffraction pattern.
    num_electrons:
        The number of electrons (per pixel) to simulate in the diffraction pattern.
    strain_matrix:
        A 3x3 matrix representing the strain to apply to the diffraction pattern.
        If None, a default strain matrix is used.
    lazy: bool
        If True, the returned signal will be lazy, otherwise it will be eager.
        Default is False.

    Returns
    -------
    Diffraction2D
        A simulated diffraction pattern with applied strain.
    """
    from diffsims.generators.simulation_generator import SimulationGenerator
    from orix.quaternion import Rotation

    if strain_matrix is None:
        strain_matrix = np.array([[0.1, 0.05, 0], [0.15, 0.2, 0], [0, 0, 1]])
    p = si_phase()
    gen = SimulationGenerator()
    rotations = Rotation.from_euler(
        [
            [0, 0, 0],
        ],
        degrees=True,
    )
    sim = gen.calculate_diffraction2d(
        phase=p, rotation=rotations, reciprocal_radius=1.5, max_excitation_error=0.1
    )

    precip = np.zeros(navigation_shape, dtype=float)
    r = navigation_shape[0] // 2
    c = navigation_shape[1] // 2
    r_rad = int(np.round(r * 0.5))
    c_rad = int(np.round(c * 0.7))
    rr, cc = skimage.draw.ellipse(r, c, r_radius=r_rad, c_radius=c_rad)
    precip[rr, cc] = 1

    scipy.ndimage.gaussian_filter(precip, sigma=3, output=precip)

    data = np.empty(precip.shape + signal_shape)

    for ind in np.ndindex(precip.shape):
        t = np.eye(3) + strain_matrix * precip[ind]
        data[ind] = create_diffraction_pattern(
            sim,
            shape=signal_shape,
            radius=disk_radius,
            num_electrons=num_electrons,
            transformation_matrix=t,
        )

    strained = signals.Diffraction2D(data)

    strained.axes_manager.signal_axes.set(
        name=("kx", "ky"), units=r"$\AA^{-1}$", scale=0.01
    )
    strained.axes_manager.navigation_axes.set(name=("x", "y"), units="nm", scale=1.0)

    strained.calibration.center = None

    if lazy:
        strained = strained.as_lazy()

    return strained
