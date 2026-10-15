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

"""Neural diffraction-spot detector (SpotUNet) and its centre networks.

``unet``, ``preprocess``, ``preprocess_torch``, ``centre``,
``centre_network``, ``centre_refine``, ``infer`` and ``refine`` are vendored
from SpyDE (https://github.com/directelectron/spyde, ``spyde/models/``): the
architectures, preprocessing and decoding must stay faithful to how the
checkpoints were trained. Each file is SpyDE's apart from black formatting
and these rewrites: imports of SpyDE's device lock and model registry point
at :mod:`.device_lock` and :mod:`.weights`, and ``SPYDE_`` environment
variables are named ``PYXEM_``. Change the maths there, not here.

:mod:`.weights` (em-database names and files, model caching) and
:mod:`.device_lock` are pyxem's. Requires torch.
"""

from .weights import (
    DEFAULT_DETECTOR,
    DETECTORS,
    clear_model_cache,
    detector_path,
    get_model,
    get_refiner,
)

__all__ = [
    "DEFAULT_DETECTOR",
    "DETECTORS",
    "clear_model_cache",
    "detector_path",
    "get_model",
    "get_refiner",
]
