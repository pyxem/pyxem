from hyperspy.api import load as hsload

from .diffraction_signal import ElectronDiffraction

__version__ = '0.2'


def load(*args, **kwargs):
    signal = hsload(*args, **kwargs)
    return ElectronDiffraction(signal)

