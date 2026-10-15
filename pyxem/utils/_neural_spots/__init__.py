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

"""Neural diffraction-spot detector (SpotUNet).

A small U-Net with a heat-map head and a subpixel-offset head. The
architecture, preprocessing and decoding are vendored from SpyDE
(https://github.com/directelectron/spyde) and must stay faithful to how the
checkpoints were trained. Requires torch; the weights come from em-database
(see :mod:`.weights`).
"""

from .weights import EMDATABASE_ENTRY, clear_model_cache, get_model, resolve_model_path

__all__ = ["EMDATABASE_ENTRY", "clear_model_cache", "get_model", "resolve_model_path"]
