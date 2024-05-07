import numpy as np
from orix.crystal_map import Phase
from orix.quaternion import Rotation
from diffpy.structure import Atom, Lattice, Structure
from diffsims.generators.simulation_generator import SimulationGenerator
from scipy.ndimage import gaussian_filter1d
from pyxem.signals import Diffraction1D, Diffraction2D


def si_phase():
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
    rotations = Rotation.from_euler([[0, 0, 0], [10, 0, 0]], degrees=True)
    sim = gen.calculate_diffraction2d(
        phase=p, rotation=rotations, reciprocal_radius=1.5, max_excitation_error=0.1
    )
    dp1 = sim.get_diffraction_pattern(
        sigma=5,
    )
    dp2 = sim.irot[1].get_diffraction_pattern(sigma=5)

    top = np.tile(
        dp1,
        (10, 5, 1, 1),
    )
    bottom = np.tile(
        dp2,
        (10, 5, 1, 1),
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
