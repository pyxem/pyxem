from hyperspy.api import load as hsload

from pymatgen import Lattice, Structure

from .diffraction_signal import ElectronDiffraction
from .diffraction_generator import ElectronDiffractionCalculator

__version__ = '0.2'


def load(*args, **kwargs):
    signal = hsload(*args, **kwargs)
    return ElectronDiffraction(signal)

