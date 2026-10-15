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

"""Where the trained weights come from, and loading them once per process.

The weights are not shipped with pyxem. Each model is its own em-database
entry (https://github.com/electronmicroscopy/emdatabase), which pins it by
dated version and checks it against its md5 on download. Every entry and
file name the neural spot finder uses is in this module and nowhere else.

A local checkpoint can always be used instead: ``model_path=`` for the
detector, or a directory holding the files below named by the
``PYXEM_SPOTUNET_WEIGHTS_DIR`` environment variable, which is looked in first.
"""

import os
import threading
from pathlib import Path

#: Environment variable naming a local directory of weight files.
LOCAL_DIRECTORY_VARIABLE = "PYXEM_SPOTUNET_WEIGHTS_DIR"

#: Detector models (``get_diffraction_vectors(method="neural", model=...)``):
#: the em-database entry and the file each one is. ``None``: no entry, so the
#: model is only available from the local directory.
DETECTORS = {
    "anyscale8": ("SpotUNetDiffraction", "spotunet_anyscale8_peaked_sharpened.pt"),
    "dynamical-mask8": ("SpotUNetDynamical", "spotunet_dynamical_mask8.pt"),
    "production": ("SpotUNetProduction", "spotunet_production.pt"),
    "base16": (None, "spotunet_base16.pt"),
}
#: SpyDE's default detector.
DEFAULT_DETECTOR = "anyscale8"

#: The center steps, by the ids the vendored ``center_refine`` asks for.
REFINERS = {
    "center-fast-f5-v1": ("SpotUNetCenterRefiner", "center_fast_f5.pt"),
    "center-partner-p4-v1": ("SpotUNetFriedelPartner", "center_partner_p4.pt"),
}

_detector_cache = {}
_refiner_cache = {}
_cache_lock = threading.Lock()


def _weights_file(entry_name, file_name, version=None):
    """A local path to ``file_name``: from the local directory when it is
    there, else the md5-checked download of the em-database entry
    ``entry_name``, which must publish exactly that file."""
    directory = os.environ.get(LOCAL_DIRECTORY_VARIABLE)
    if directory and (Path(directory) / file_name).is_file():
        return str(Path(directory) / file_name)
    if entry_name is None:
        raise FileNotFoundError(
            f"{file_name} is not distributed through em-database; put it in a "
            f"directory named by {LOCAL_DIRECTORY_VARIABLE}."
        )
    try:
        import emdatabase.data
    except ImportError as error:
        raise ImportError(
            "The neural spot finder downloads its weights with em-database. "
            "Install it with `pip install em-database` (or `pip install "
            "pyxem[neural]`), or put the weight files in a directory named by "
            f"{LOCAL_DIRECTORY_VARIABLE}."
        ) from error
    entry_class = getattr(emdatabase.data, entry_name, None)
    if entry_class is None:
        raise ImportError(
            f"The installed em-database has no `{entry_name}` entry yet. Upgrade "
            f"em-database, or put {file_name} in a directory named by "
            f"{LOCAL_DIRECTORY_VARIABLE}."
        )
    entry = entry_class()
    if entry.file != file_name:
        raise ValueError(
            f"em-database entry `{entry_name}` publishes {entry.file}, not "
            f"{file_name}; this pyxem expects {file_name}."
        )
    return str(entry.download(version=version, background=False))


def detector_path(model=None, model_path=None, version=None):
    """Local checkpoint of a detector: ``model_path`` when given, else the
    weights of ``model`` (a key of :data:`DETECTORS`)."""
    if model_path is not None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"No spot-detector checkpoint at {path}")
        return str(path)
    model = model or DEFAULT_DETECTOR
    if model not in DETECTORS:
        raise ValueError(f"model must be one of {sorted(DETECTORS)}, not {model!r}")
    entry_name, file_name = DETECTORS[model]
    return _weights_file(entry_name, file_name, version)


def _device(device):
    import torch

    from .infer import _default_device

    return torch.device(device) if device is not None else _default_device()


def get_model(model=None, model_path=None, version=None, device=None):
    """The detector as ``(network, device)``, loaded once per process."""
    from .infer import load_model

    path = detector_path(model, model_path, version)
    device = _device(device)
    key = (path, str(device))
    with _cache_lock:
        if key not in _detector_cache:
            _detector_cache[key] = load_model(path, device)
        return _detector_cache[key]


def get_refiner(model_id, device=None):
    """A center-step network (a key of :data:`REFINERS`), loaded once per process."""
    from .center_network import load_refiner

    entry_name, file_name = REFINERS[model_id]
    device = _device(device)
    key = (model_id, str(device))
    with _cache_lock:
        if key not in _refiner_cache:
            _refiner_cache[key] = load_refiner(
                _weights_file(entry_name, file_name), device
            )
        return _refiner_cache[key]


def demote_cached_models_to_cpu():
    """After a device failure, move every cached detector to the CPU so the
    next caller does not select the failed device again."""
    import torch

    cpu = torch.device("cpu")
    with _cache_lock:
        for key, (network, _) in list(_detector_cache.items()):
            _detector_cache[key] = (network.to(cpu), cpu)


def clear_model_cache():
    with _cache_lock:
        _detector_cache.clear()
        _refiner_cache.clear()
