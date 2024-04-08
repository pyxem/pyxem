# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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

"""Additional utility functions for processing signals.

.. currentmodule:: pyxem.utils

.. rubric:: Modules

.. autosummary::
    :toctree: ../generated/
    :template: custom-module-template.rst

    plotting
    ransac_ellipse_tools
    vectors
    diffraction
    calibration

"""


from pyxem.utils.ransac_ellipse_tools import determine_ellipse
from pyxem.utils.calibration import find_diffraction_calibration
from pyxem.utils.plotting import plot_template_over_pattern

from pyxem.utils import vectors
from pyxem.utils import plotting
from pyxem.utils import ransac_ellipse_tools
from pyxem.utils import calibration
from pyxem.utils import diffraction

__all__ = [
    "find_diffraction_calibration",
    "plot_template_over_pattern",
    "determine_ellipse",
    "ransac_ellipse_tools",
    "vectors",
    "calibration",
    "plotting",
    "diffraction",
]
