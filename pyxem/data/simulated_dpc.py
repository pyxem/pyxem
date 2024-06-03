import pyxem as pxm
from skimage.draw import disk
import numpy as np


def simulated_stripes(beam_shifts=None):
    """
    Create a simulated diffraction pattern with stripes for showing magnetic DPC.

    Parameters
    ----------
    shifts : array_like, optional
        The shifts to apply to the direct beam.  The default is None which
        corresponds to a shift of [[3, 3], [-3,3]] for the alternating stripes.
    Returns
    -------
    diffraction_pattern : Signal2D
        A simulated diffraction pattern with the direct beam shifted by .
    """
    if beam_shifts is None:
        beam_shifts = np.array([[3, 3], [-3, 3]])
    data = np.zeros((60, 60, 64, 64))
    rr, cc = disk(center=(32, 32), radius=16)
    data[:, :, rr, cc] = 1
    s = pxm.signals.ElectronDiffraction2D(data)
    shifts = np.zeros((60, 60, 2))
    shifts[20:40, 10:20] = beam_shifts[0]
    shifts[20:40, 20:30] = beam_shifts[1]
    shifts[20:40, 30:40] = beam_shifts[0]
    shifts[20:40, 40:50] = beam_shifts[1]
    bs = pxm.signals.BeamShift(shifts)
    s.center_direct_beam(shifts=bs)  # Shift the direct beam
    s.axes_manager[2].scale = 0.1
    s.axes_manager[3].scale = 0.1
    return s
