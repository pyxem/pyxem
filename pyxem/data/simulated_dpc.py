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


def simulated_constant_shift_magnitude(beam_radius=10):
    """
    Create a simulated diffraction pattern with a direct beam that is shifted by a constant magnitude across
    the entire pattern. In the presence of electromagnetic fields in the entire sample area, the plane fitting
    can fail, however by minimizing the magnitude variance, we can fit a plane to the shifts


    Parameters
    ----------
    beam_radius : int, optional
        The radius of the direct beam in pixels. The default is 10.

    Returns
    -------
    probes : Diffraction2D
        A simulated diffraction pattern with a direct beam that is shifted by a constant
        magnitude across the entire pattern.

    """
    bright_field_disk = np.zeros((128, 128), dtype=np.int16)
    bright_field_disk[
        np.sum((np.mgrid[:128, :128] - 64) ** 2, axis=0) < beam_radius**2
    ] = 500

    probes = np.zeros((20, 20), dtype=int)
    probes = bright_field_disk[np.newaxis][probes]
    probes = pxm.signals.Diffraction2D(probes)

    p = [0.5] * 6  # Plane parameters
    x, y = np.meshgrid(np.arange(20), np.arange(20))
    base_plane_x = p[0] * x + p[1] * y + p[2]
    base_plane_y = p[3] * x + p[4] * y + p[5]

    base_plane = np.stack((base_plane_x, base_plane_y)).T
    data = base_plane.copy()

    shifts = np.zeros_like(data)
    shifts[:10, 10:] = (10, 10)
    shifts[:10, :10] = (10, -10)
    shifts[10:, 10:] = (-10, 10)
    shifts[10:, :10] = (-10, -10)
    data += shifts
    data = pxm.signals.BeamShift(data)
    probes.center_direct_beam(shifts=-data)
    return probes
