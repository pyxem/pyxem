import numpy as np
from orix.quaternion import Rotation
from diffsims.generators.simulation_generator import SimulationGenerator
from pyxem.signals import Diffraction2D
from pyxem.data import si_phase
from skimage.draw import ellipse, disk
from scipy.ndimage import gaussian_filter


def create_diffraction_pattern(
    simulation,
    shape=(512, 512),
    direct_beam_position=None,
    radius=20,
    num_electrons=None,
    in_plane_angle=0,
    calibration=0.01,
    mirrored=False,
    transformation_matrix=None,
):
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
            rr, cc = disk(cord[:2], radius, shape=shape)
            pattern[rr, cc] = inten
    if num_electrons is not None:
        total = np.sum(spot_intens) * radius**2 * np.pi
        pattern = np.random.poisson((pattern / total) * num_electrons)
    return np.divide(pattern, np.max(pattern))


def simulated_strain(
    navigation_shape=(32, 32),
    signal_shape=(512, 512),
    disk_radius=20,
    num_electrons=1e5,
    strain_matrix=None,
    lazy=False,
):
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
    rr, cc = ellipse(r, c, r_radius=r_rad, c_radius=c_rad)
    precip[rr, cc] = 1

    gaussian_filter(precip, sigma=3, output=precip)

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

    strained = Diffraction2D(data)
    strained.axes_manager.signal_axes[0].name = "kx"
    strained.axes_manager.signal_axes[1].name = "kx"
    strained.axes_manager.signal_axes[0].units = r"$\AA^{-1}$"
    strained.axes_manager.signal_axes[1].units = r"$\AA^{-1}$"
    strained.axes_manager.signal_axes[0].scale = 0.01
    strained.axes_manager.signal_axes[1].scale = 0.01

    strained.axes_manager.navigation_axes[0].name = "x"
    strained.axes_manager.navigation_axes[1].name = "y"
    strained.axes_manager.navigation_axes[0].units = "nm"
    strained.axes_manager.navigation_axes[1].units = "nm"

    strained.calibration.center = None

    if lazy:
        strained = strained.as_lazy()

    return strained
