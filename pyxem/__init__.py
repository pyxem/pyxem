# -*- coding: utf-8 -*-
# Copyright 2016-2025 The pyXem developers
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


import importlib.util
import logging

try:
    import cupy as cp

    CUPY_INSTALLED = True
except ImportError:
    CUPY_INSTALLED = False
# Checked without importing, which takes seconds: onnxruntime runs the neural
# spot finder, torch the ``device=`` template matching.
ONNXRUNTIME_INSTALLED = importlib.util.find_spec("onnxruntime") is not None
TORCH_INSTALLED = importlib.util.find_spec("torch") is not None
from pyxem import components
from pyxem import signals
from pyxem import generators
from pyxem import data

from . import release_info


_logger = logging.getLogger(__name__)


__all__ = [
    "components",
    "generators",
    "signals",
    "data",
    "CUPY_INSTALLED",
    "ONNXRUNTIME_INSTALLED",
    "TORCH_INSTALLED",
]

__version__ = release_info.version
__author__ = release_info.author
__copyright__ = release_info.copyright
__credits__ = release_info.credits
__license__ = release_info.license
__maintainer__ = release_info.maintainer
__email__ = release_info.email
__status__ = release_info.status
