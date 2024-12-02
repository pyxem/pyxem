import numpy as np
from orix.crystal_map import Phase
from orix.quaternion import Rotation, Orientation
from diffpy.structure import Atom, Lattice, Structure
from diffsims.generators.simulation_generator import SimulationGenerator
from scipy.ndimage import gaussian_filter1d
from pyxem.signals import Diffraction1D, Diffraction2D
import skimage


def si_phase():
    """Create a silicon phase with space group 227. This is a diamond cubic structure with
    a lattice parameter of 5.431 Å.

    """
    a = 5.431
    latt = Lattice(a, a, a, 90, 90, 90)
    atom_list = []
    for coords in [[0, 0, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0]]:
        x, y, z = coords[0], coords[1], coords[2]
        atom_list.append(Atom(atype="Si", xyz=[x, y, z], lattice=latt))  # Motif part A
        atom_list.append(
            Atom(atype="Si", xyz=[x + 0.25, y + 0.25, z + 0.25], lattice=latt)
        )  # Motif part B
    struct = Structure(atoms=atom_list, lattice=latt)
    p = Phase(structure=struct, space_group=227)
    return p


def si_tilt():
    p = si_phase()
    gen = SimulationGenerator()
    rotations = Rotation.from_euler(
        [[0, 0, 0], [10, 0, 0]],
        degrees=True,
    )
    sim = gen.calculate_diffraction2d(
        phase=p, rotation=rotations, reciprocal_radius=1.5, max_excitation_error=0.1
    )
    dp1 = np.flipud(
        sim.get_diffraction_pattern(sigma=5, shape=(128, 128))
    )  # flip up/down to go from scatter to diffraction
    dp2 = np.flipud(sim.irot[1].get_diffraction_pattern(sigma=5, shape=(128, 128)))

    top = np.tile(
        dp1,
        (4, 2, 1, 1),
    )
    bottom = np.tile(
        dp2,
        (4, 2, 1, 1),
    )
    total = np.hstack((top, bottom))
    tilt = Diffraction2D(total)
    tilt.axes_manager.signal_axes[0].name = "kx"
    tilt.axes_manager.signal_axes[1].name = "kx"
    tilt.axes_manager.signal_axes[0].units = r"$\AA^{-1}$"
    tilt.axes_manager.signal_axes[1].units = r"$\AA^{-1}$"
    tilt.axes_manager.signal_axes[0].scale = 0.01
    tilt.axes_manager.signal_axes[1].scale = 0.01
    return tilt


def si_grains_from_orientations(
    oris: Orientation, seed: int = 2, size: int = 20, recip_pixels: int = 128
):
    p = si_phase()
    gen = SimulationGenerator()
    num_grains = oris.size

    sim = gen.calculate_diffraction2d(
        phase=p, rotation=oris, reciprocal_radius=1.5, max_excitation_error=0.1
    )
    dps = [
        np.flipud(
            sim.irot[i].get_diffraction_pattern(
                sigma=5, shape=(recip_pixels, recip_pixels)
            )
        )
        for i in range(num_grains)
    ]
    rng = np.random.default_rng(seed)
    x = rng.integers(0, size, size=num_grains)
    y = rng.integers(0, size, size=num_grains)
    navigator = np.zeros((size, size))
    for i in range(num_grains):
        navigator[x[i], y[i]] = i + 1
    navigator = skimage.segmentation.expand_labels(navigator, distance=size)
    grain_data = np.empty((size, size, recip_pixels, recip_pixels))
    for i in range(num_grains):
        grain_data[navigator == i + 1] = dps[i][np.newaxis]
    grains = Diffraction2D(grain_data)
    grains.axes_manager.signal_axes[0].name = "kx"
    grains.axes_manager.signal_axes[1].name = "kx"
    grains.axes_manager.signal_axes[0].units = r"$\AA^{-1}$"
    grains.axes_manager.signal_axes[1].units = r"$\AA^{-1}$"
    grains.axes_manager.signal_axes[0].scale = 0.01
    grains.axes_manager.signal_axes[1].scale = 0.01
    return grains


def si_grains(num_grains=4, seed=2, size=6, recip_pixels=128, return_rotations=False):
    """Generate a simulated dataset with grains in random orientations"""
    p = si_phase()
    rotations = Orientation.random(num_grains, symmetry=p.point_group)
    grains = si_grains_from_orientations(
        rotations, seed=seed, size=size, recip_pixels=recip_pixels
    )

    if return_rotations:
        return grains, rotations
    else:
        return grains


def si_grains_simple(seed=2, size=6, recip_pixels=128, return_rotations=False):
    """Generate a simulated dataset with low-index zone axes"""
    p = si_phase()
    rotations = Orientation.from_euler(
        [
            [0, 0, 0],  # [0 0 1]
            [0, 45, 0],  # [0 1 1]
            [0, 54.7, 45],  # [1 1 1]
            [0, 35, 45],  # [1 1 2]
        ],
        degrees=True,
        symmetry=p.point_group,
    )
    grains = si_grains_from_orientations(
        rotations, seed=seed, size=size, recip_pixels=recip_pixels
    )

    if return_rotations:
        return grains, rotations
    else:
        return grains


def si_rotations_line():
    from orix.sampling import get_sample_reduced_fundamental

    p = si_phase()
    gen = SimulationGenerator()
    rotations = get_sample_reduced_fundamental(resolution=3, point_group=p.point_group)
    sim = gen.calculate_diffraction2d(
        phase=p, rotation=rotations, max_excitation_error=0.1, reciprocal_radius=2
    )
    dps = []
    for i in range(rotations.size):
        dp = np.flipud(sim.irot[i].get_diffraction_pattern(sigma=5, shape=(256, 256)))
        dps.append(dp)
    line = Diffraction2D(np.array(dps))
    line.axes_manager.signal_axes[0].name = "kx"
    line.axes_manager.signal_axes[1].name = "kx"
    line.axes_manager.signal_axes[0].units = r"$\AA^{-1}$"
    line.axes_manager.signal_axes[1].units = r"$\AA^{-1}$"
    line.axes_manager.signal_axes[0].scale = 0.01
    line.axes_manager.signal_axes[1].scale = 0.01
    return line


def simulated1dsi(
    num_points=200,
    accelerating_voltage=200,
    reciporical_radius=1,
    sigma=2,
):
    p = si_phase()
    gen = SimulationGenerator(
        accelerating_voltage=accelerating_voltage,
    )
    x = np.zeros(num_points)
    sim = gen.calculate_diffraction1d(phase=p, reciprocal_radius=reciporical_radius)
    int_points = np.round(
        np.array(sim.reciprocal_spacing) * num_points / reciporical_radius
    ).astype(int)
    x[int_points] = sim.intensities
    x = gaussian_filter1d(x, sigma)
    s = Diffraction1D(x)
    s.axes_manager[0].name = "k"
    s.axes_manager[0].units = r"$\AA^{-1}$"
    s.axes_manager[0].scale = reciporical_radius / num_points
    return s
