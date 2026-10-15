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

import math

import numpy as np
import traits.api as t
from hyperspy.ui_registry import add_gui_method
from scipy.ndimage import gaussian_filter

from pyxem.utils import _find_vectors


@add_gui_method(toolkey="pyxem.Diffraction2D.get_diffraction_vectors")
class DiffractionVectorFinder(t.HasTraits):
    """The settings of ``get_diffraction_vectors(method="neural")``, tuned on
    one pattern at a time.

    Created by ``get_diffraction_vectors(method="neural", interactive=True)``,
    which also opens its GUI. Every change of a setting or of
    :attr:`position` re-runs the detector on the pattern at that position;
    :meth:`compute` then searches every pattern with the settings as left.

    Attributes
    ----------
    pattern : numpy.ndarray
        The pattern at :attr:`position`, navigation-blurred as the full
        search will blur it.
    peaks : numpy.ndarray
        ``(N, 3)`` ``[row, column, intensity]`` found in :attr:`pattern`.
    response : numpy.ndarray
        Each detection's confidence painted at its pixel.
    vectors : DiffractionVectors or None
        The result of the last :meth:`compute`.
    """

    threshold = t.Range(0.0, 1.0, value=0.3, desc="Network confidence threshold")
    min_distance = t.Range(1, 64, value=4, desc="Minimum spot separation (px)")
    spot_radius = t.Range(0.0, 64.0, value=0.0, desc="Spot radius (px), 0 = auto")
    navigation_sigma = t.Range(0.0, 10.0, value=0.0, desc="Navigation blur")
    mask_beam_stop = t.Bool(False, desc="Exclude a beam stop")
    mask_dilation = t.Range(0, 32, value=5, desc="Beam-stop dilation (px)")
    show_response = t.Bool(False, desc="Show where the network fired")
    #: Navigation index (array order) of the displayed pattern.
    position = t.Tuple()
    #: Fired after every detection, for a GUI to redraw.
    detected = t.Event()

    def __init__(
        self,
        signal,
        finder,
        mask=None,
        mask_automatic=False,
        mask_dilation=_find_vectors.DEFAULT_MASK_DILATION,
        navigation_sigma=0.0,
        neighbour_refine=False,
        center=None,
        calibration=None,
        column_names=None,
        units=None,
        get_intensity=True,
    ):
        super().__init__()
        self.signal = signal
        self.finder = finder
        self.neighbour_refine = neighbour_refine
        self._output_settings = dict(
            center=center,
            calibration=calibration,
            column_names=column_names,
            units=units,
            get_intensity=get_intensity,
        )
        self._navigation_dimension = signal.axes_manager.navigation_dimension
        self._user_mask = None if mask_automatic else mask
        self._detected_stop = None
        self.vectors = None
        self.widgets = None
        self.peaks = _find_vectors.empty_peaks()
        self.response = None
        self._holding = True
        self.trait_set(
            threshold=finder.threshold,
            min_distance=finder.min_distance,
            spot_radius=finder.spot_radius or 0.0,
            navigation_sigma=float(navigation_sigma),
            mask_beam_stop=mask is not None,
            mask_dilation=int(mask_dilation),
            position=tuple(
                size // 2 for size in signal.data.shape[: self._navigation_dimension]
            ),
        )
        self._holding = False
        self.pattern = self._read_pattern()
        self.detect()

    @t.observe(
        [
            "threshold",
            "min_distance",
            "spot_radius",
            "mask_beam_stop",
            "mask_dilation",
            "show_response",
        ]
    )
    def _setting_changed(self, event):
        if not self._holding:
            self.detect()

    @t.observe(["position", "navigation_sigma"])
    def _pattern_changed(self, event):
        if not self._holding:
            self.pattern = self._read_pattern()
            self.detect()

    def _update_finder(self):
        self.finder.threshold = float(self.threshold)
        self.finder.min_distance = int(self.min_distance)
        self.finder.spot_radius = float(self.spot_radius) or None
        return self.finder

    @property
    def mask(self):
        """The beam-stop mask in use, or None."""
        if not self.mask_beam_stop:
            return None
        if self._user_mask is not None:
            return self._user_mask
        if self._detected_stop is None:
            stop = _find_vectors.automatic_mask(
                self.signal.data, self._navigation_dimension, dilation=0
            )
            self._detected_stop = False if stop is None else stop
        if self._detected_stop is False:
            return None
        return _find_vectors.dilate_mask(self._detected_stop, self.mask_dilation)

    @property
    def parameters(self):
        """The current settings, as keyword arguments for
        ``get_diffraction_vectors``."""
        parameters = dict(self._update_finder().parameters)
        parameters["navigation_sigma"] = float(self.navigation_sigma)
        if not self.mask_beam_stop:
            parameters["mask"] = None
        elif self._user_mask is not None:
            parameters["mask"] = self._user_mask
        else:
            parameters["mask"] = "auto"
            parameters["mask_dilation"] = int(self.mask_dilation)
        return parameters

    def _read_pattern(self):
        """The pattern at :attr:`position`, navigation-blurred like the full
        search. Only the neighbourhood the blur reaches is read, so a lazy
        signal computes a handful of patterns rather than a chunk."""
        data = self.signal.data
        position = tuple(self.position)
        if self._navigation_dimension == 0:
            pattern = data
        elif self.navigation_sigma <= 0:
            pattern = data[position]
        else:
            reach = int(math.ceil(3 * self.navigation_sigma))
            window = []
            for axis, index in enumerate(position):
                size = data.shape[axis]
                offsets = np.arange(index - reach, index + reach + 1)
                # Half-sample symmetric reflection, as the full search pads.
                offsets = np.where(offsets < 0, -offsets - 1, offsets)
                offsets = np.where(offsets >= size, 2 * size - offsets - 1, offsets)
                window.append(np.clip(offsets, 0, size - 1))
            slab = data[tuple(slice(w.min(), w.max() + 1) for w in window)]
            if hasattr(slab, "compute"):
                slab = slab.compute()
            neighbourhood = np.asarray(slab)[np.ix_(*[w - w.min() for w in window])]
            blurred = gaussian_filter(
                neighbourhood.astype(np.float32),
                sigma=(self.navigation_sigma,) * self._navigation_dimension + (0, 0),
                mode="reflect",
                truncate=3.0,
            )
            pattern = blurred[(reach,) * self._navigation_dimension]
        if hasattr(pattern, "compute"):
            pattern = pattern.compute()
        return np.asarray(pattern, dtype=np.float32)

    def detect(self):
        """Re-run the detector on :attr:`pattern`."""
        self.response, self.peaks = self._update_finder().detect(
            self.pattern, mask=self.mask
        )
        self.detected = True

    def marker_radius(self):
        """Radius in pixels at which a detected spot is drawn."""
        return self._update_finder().marker_radius()

    def compute(self):
        """Find the vectors in every pattern with the current settings.

        Returns
        -------
        DiffractionVectors
            Lazy if the signal is lazy; also kept on :attr:`vectors`.
        """
        self.vectors = self.signal._vectors_from_finder(
            self._update_finder(),
            mask=self.mask,
            navigation_sigma=float(self.navigation_sigma),
            neighbour_refine=self.neighbour_refine,
            **self._output_settings,
        )
        return self.vectors
