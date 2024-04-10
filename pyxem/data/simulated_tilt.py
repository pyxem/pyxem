import numpy as np
from pyxem.data.dummy_data import make_diffraction_test_data as mdtd
import hyperspy.api as hs


def tilt_boundary_data(correct_pivot_point=True):
    di = mdtd.DiffractionTestImage(intensity_noise=False)
    di.add_disk(x=128, y=128, intensity=10.0)  # Add a zero beam disk at the center
    di.add_cubic_disks(vx=20, vy=20, intensity=2.0, n=5)
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
    s.axes_manager[2].scale = 0.3
    s.axes_manager[3].scale = 0.3
    s.axes_manager[2].offset = -0.3 * 128
    s.axes_manager[3].offset = -0.3 * 128
    return s
