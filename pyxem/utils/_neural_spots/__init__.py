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

"""Direct Electron's neural diffraction-spot networks, run with ONNX Runtime:
the DE SpotNet detectors and the DE CenterNet and DE FriedelNet center
networks.

The networks are SpyDE's (https://github.com/directelectron/spyde,
``spyde/models/`` at 68dfa52038; there the SpotUNet detectors anyscale8,
dynamical-mask8, production and base16 and the F5 and P4 center networks),
exported to ONNX with BatchNorm folded in;
each ``.onnx`` file carries the settings the pipeline needs as metadata. The
code around them reproduces SpyDE's:

* ``preprocess``, ``center_refine`` and ``refine`` are SpyDE's numpy modules,
  verbatim apart from black formatting, US spelling (SpyDE's ``centre*.py``
  are ``center*.py`` here; every ``centre``, ``neighbour``, ``normalise`` and
  ``-ise`` is ``center``, ``neighbor``, ``normalize`` and ``-ize``), ``SPYDE_``
  environment variables named ``PYXEM_``, the model registry replaced by
  :mod:`.weights`, and ``center_refine``'s torch GPU window path removed;
* ``center``, ``networks`` and ``infer`` are numpy ports of SpyDE's torch
  ``centre``, ``centre_network`` and ``infer``, computed in float32 as the
  originals are.

Change the math in SpyDE first, then here. :mod:`.weights` (em-database names
and files, model caching) is pyxem's. Requires onnxruntime.
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
