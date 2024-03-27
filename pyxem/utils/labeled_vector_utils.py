import itertools
from copy import deepcopy

import numpy as np
from scipy.spatial import ConvexHull

warnings.warn(
    "This module has been renamed and should now be imported as `pyxem.utils.vectors`",
    FutureWarning,
)
from pyxem.utils.vectors import (
    column_mean,
    vectors2image,
    points_to_polygon,
    convert_to_markers,
    points_to_poly_collection,
)
