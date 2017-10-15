import numpy as np
import fpd_data_processing.make_diffraction_test_data as mdtd
from fpd_data_processing.pixelated_stem_class import DPCSignal2D


def get_disk_shift_simple_test_signal():
    """Get HyperSpy 2D signal with 2D navigation dimensions for DPC testing.

    Probe size x/y (20, 20), and image size x/y (50, 50).
    Disk moves from 22-28 x/y.
    """
    disk_x, disk_y = np.mgrid[22:28:20j, 22:28:20j]
    s = mdtd.generate_4d_data(
            probe_size_x=20, probe_size_y=20,
            image_size_x=50, image_size_y=50,
            disk_x=disk_x, disk_y=disk_y, disk_r=2,
            ring_x=None, add_noise=True)
    return(s)


def get_holz_simple_test_signal():
    """Get HyperSpy 2D signal with 2D navigation dimensions for HOLZ testing.

    Probe size x/y (20, 20), and image size x/y (50, 50).
    Contains a disk and a ring. The disk stays at x, y = 25, 25, with radius 2.
    The ring has a radius of 20, and moves from x, y = 24-26, 24-26.
    """
    ring_x, ring_y = np.mgrid[24:26:20j, 24:26:20j]
    s = mdtd.generate_4d_data(
            probe_size_x=20, probe_size_y=20,
            image_size_x=50, image_size_y=50,
            disk_x=25, disk_y=25, disk_r=2, disk_I=20,
            ring_x=ring_x, ring_y=ring_y, ring_r=15, ring_I=10,
            add_noise=True)
    return(s)


def get_single_ring_diffraction_signal():
    """Get HyperSpy 2D signal with a single ring with centre position.

    The ring has a centre at x=105 and y=67, and radius=40.
    """
    data = mdtd.MakeTestData(size_x=200, size_y=150, default=False, blur=True)
    x, y = 105, 67
    data.add_ring(x, y, r=40)
    s = data.signal
    s.axes_manager[0].offset, s.axes_manager[1].offset = -x, -y
    return(s)


def get_simple_dpc_signal():
    """Get a simple DPCSignal2D with a zero point in the centre.

    Example
    -------
    >>> import fpd_data_processing.api as fp
    >>> s = fp.dummy_data.get_simple_dpc_signal()

    """
    data = np.mgrid[-5:5:100j, -5:5:100j]
    s = DPCSignal2D(data)
    return s
