import numpy as np
from orix.crystal_map import Phase
from orix.quaternion import Rotation
from diffpy.structure import Atom, Lattice, Structure
from diffsims.generators.simulation_generator import SimulationGenerator
from pyxem.signals import Diffraction2D
import skimage


def fe_fcc_phase():
    """Create a fe fcc phase with space group 225. This is a fcc structure with
    a lattice parameter of 3.571 Å.

    """
    a = 3.571
    latt = Lattice(a, a, a, 90, 90, 90)
    atom_list = []
    for coords in [[0, 0, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0]]:
        x, y, z = coords[0], coords[1], coords[2]
        atom_list.append(Atom(atype="Fe", xyz=[x, y, z], lattice=latt))  # Motif part A
    struct = Structure(atoms=atom_list, lattice=latt)
    p = Phase(structure=struct, space_group=225)
    return p


def fe_bcc_phase():
    """Create a fe bcc phase with space group 229. This is a bcc structure with
    a lattice parameter of 2.866 Å.

    """
    a = 2.866
    latt = Lattice(a, a, a, 90, 90, 90)
    atom_list = []
    for coords in [[0, 0, 0], [0.5, 0.5, 0.5]]:
        x, y, z = coords[0], coords[1], coords[2]
        atom_list.append(Atom(atype="Fe", xyz=[x, y, z], lattice=latt))  # Motif part A
    struct = Structure(atoms=atom_list, lattice=latt)
    p = Phase(structure=struct, space_group=229)
    return p


def fe_multi_phase_grains(num_grains=2, seed=2, size=20, recip_pixels=128):
    bcc = fe_bcc_phase()
    fcc = fe_fcc_phase()
    gen = SimulationGenerator()
    rotations1 = Rotation.random(num_grains)
    rotations2 = Rotation.random(num_grains)

    sim = gen.calculate_diffraction2d(
        phase=[bcc, fcc],
        rotation=[rotations1, rotations2],
        reciprocal_radius=2.5,
        max_excitation_error=0.1,
    )
    dps_bcc = [
        sim.iphase[0]
        .irot[i]
        .get_diffraction_pattern(
            sigma=3, shape=(recip_pixels, recip_pixels), calibration=0.02
        )
        for i in range(num_grains)
    ]
    dps_fcc = [
        sim.iphase[1]
        .irot[i]
        .get_diffraction_pattern(
            sigma=3, shape=(recip_pixels, recip_pixels), calibration=0.02
        )
        for i in range(num_grains)
    ]
    rng = np.random.default_rng(seed)
    x = rng.integers(0, size, size=num_grains * 2)
    y = rng.integers(0, size, size=num_grains * 2)
    navigator = np.zeros((size, size))
    for i in range(num_grains * 2):
        navigator[x[i], y[i]] = i + 1
    navigator = skimage.segmentation.expand_labels(navigator, distance=size)
    grain_data = np.empty((size, size, recip_pixels, recip_pixels))
    for i in range(num_grains * 2):
        if i < num_grains:
            grain_data[navigator == i + 1] = dps_bcc[i][np.newaxis]
        else:
            grain_data[navigator == i + 1] = dps_fcc[i % num_grains][np.newaxis]
    grains = Diffraction2D(grain_data)
    grains.axes_manager.signal_axes[0].name = "kx"
    grains.axes_manager.signal_axes[1].name = "kx"
    grains.axes_manager.signal_axes[0].units = r"$\AA^{-1}$"
    grains.axes_manager.signal_axes[1].units = r"$\AA^{-1}$"
    grains.axes_manager.signal_axes[0].scale = 0.02
    grains.axes_manager.signal_axes[1].scale = 0.02
    return grains
