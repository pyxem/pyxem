from pyxem.data.dummy_data import CrystalSTEMSimulation
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from pyxem.data.simulated_fe import fe_fcc_phase


def simulated_overlap(
    real_space_pixels=128,
    num_crystals=50,
    noise_level=0.1,
    num_electrons=10,
    radius=5,
    recip_space_pixels=64,
    k_range=1,
):
    """
    Create a simulated diffraction pattern with overlapping nanocrystals.

    Returns
    -------
    diffraction_pattern : Signal2D
        A simulated diffraction pattern with overlapping nanocrystals.
    """
    fcc_fe = fe_fcc_phase()
    sim = CrystalSTEMSimulation(
        fcc_fe,
        real_space_pixels=real_space_pixels,
        num_crystals=num_crystals,
        recip_space_pixels=recip_space_pixels,
        k_range=k_range,
    )
    arr = sim.make_4d_stem(
        num_electrons=num_electrons, noise_level=noise_level, radius=radius
    )
    signal = ElectronDiffraction2D(arr)
    signal.axes_manager.signal_axes[1].scale = k_range / (recip_space_pixels / 2)
    signal.axes_manager.signal_axes[0].scale = k_range / (recip_space_pixels / 2)
    signal.calibration.center = None
    return signal
