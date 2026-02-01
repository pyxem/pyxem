import numpy as np
from pyxem.data.dummy_data import make_diffraction_test_data as mdtd
import hyperspy.api as hs
import hyperspy.misc.utils as hs_utils


def tilt_boundary_data(correct_pivot_point=True, spacing=20, spot_radius=7):
    """
    Create an ElectronDiffraction2D signal across a grain boundary.

    Parameters
    ----------
    correct_pivot_point : bool, optional
        If True, the direct beam will moved similarly to residual pivot
        point that can be observed experimentally.
    spacing : int or tuple, optional
        The spacing between the disks in the diffraction pattern. If a single
        integer is provided, it will be used for both x and y directions.
    spot_radius : int, optional
        The radius of the disks in the diffraction pattern.

    Returns
    -------
    ElectronDiffraction2D
        A signal of diffraction patterns across a grain boundary.
    """
    if not hs_utils.isiterable(spacing):
        spacing = (spacing, spacing)

    di = mdtd.DiffractionTestImage(disk_r=spot_radius, intensity_noise=False)
    di.add_disk(x=128, y=128, intensity=10.0)  # Add a zero beam disk at the center
    di.add_cubic_disks(vx=spacing[0], vy=spacing[1], intensity=2.0, n=5)
    di.add_background_lorentz()
    di_rot = di.copy()
    di_rot.rotation = 10
    dtd = mdtd.DiffractionTestDataset(10, 10, 256, 256)
    position_array = np.ones((10, 10), dtype=bool)
    position_array[:5] = False
    dtd.add_diffraction_image(di, position_array)
    dtd.add_diffraction_image(di_rot, np.invert(position_array))
    s = dtd.get_signal()
    if not correct_pivot_point:
        # Shifting the zero beam away from the center
        xx, yy = np.meshgrid(range(10), range(10))
        shifts = np.stack([xx * 0.5, yy * 0.5], axis=-1)
        s.center_direct_beam(shifts=hs.signals.Signal1D(shifts))
    s.axes_manager.signal_axes.set(scale=0.3, offset=-0.3 * 128)
    return s
