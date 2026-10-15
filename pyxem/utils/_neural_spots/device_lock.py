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

"""One process-wide lock around Apple MPS submissions.

Two threads submitting to Metal at once corrupt its command encoder and crash
the process with a native fault no ``except`` catches. Every torch call in
this package that touches an MPS device takes this lock. CUDA is thread safe
and its concurrency is wanted, so the lock is a null context off MPS.
"""

import contextlib
import threading

DEVICE_LOCK = threading.RLock()


def accelerator_lock(device):
    """The lock for ``device``: :data:`DEVICE_LOCK` on MPS, else a null context."""
    if device is not None and getattr(device, "type", str(device)) == "mps":
        return DEVICE_LOCK
    return contextlib.nullcontext()
