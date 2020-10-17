# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from hyperspy.signals import BaseSignal

from transforms3d.euler import euler2quat, quat2axangle, euler2axangle
from transforms3d.quaternions import qmult, qinverse

from pyxem.utils.indexation_utils import get_phase_name_and_index
from pyxem.signals import transfer_navigation_axes_to_signal_axes

"""Signal class for crystallographic phase and orientation maps."""


def _metric_from_dict(metric_dict, metric):
    """Utility function for retrieving an entry in a dictionary. Used to map
    over dicts in signal space.

    Parameters
    ----------
    metric_dict : dict
        Dictionary to retrieve entry from
    metrics : str
        Name of the entry

    Returns
    -------
    entry
        Dictionary entry specified by metric.

    """
    return metric_dict[0][metric]
