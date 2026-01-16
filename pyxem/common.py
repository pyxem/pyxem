import importlib
import numpy

if numpy.__version__ >= "1.25.0":
    from numpy.exceptions import VisibleDeprecationWarning
else:
    from numpy import VisibleDeprecationWarning


CUPY_INSTALLED = importlib.util.find_spec("cupy") is not None


__all__ = ["CUPY_INSTALLED", "VisibleDeprecationWarning"]


def __dir__():
    return sorted(__all__)
