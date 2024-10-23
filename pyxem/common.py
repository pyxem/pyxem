import numpy

if numpy.__version__ >= "1.25.0":
    from numpy.exceptions import VisibleDeprecationWarning
else:
    from numpy import VisibleDeprecationWarning
