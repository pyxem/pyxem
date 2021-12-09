# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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

import pytest
import numpy as np
from pyxem.generators import PDFGenerator1D
from pyxem.signals.reduced_intensity1d import ReducedIntensity1D
from pyxem.signals.pair_distribution_function1d import PairDistributionFunction1D


@pytest.fixture
def reduced_intensity1d():
    data = np.ones((1, 10)) * np.arange(4).reshape(4, 1)
    data = data.reshape(2, 2, 10)
    ri = ReducedIntensity1D(data)
    return ri


def test_pdf_gen_init(reduced_intensity1d):
    pdfgen = PDFGenerator1D(reduced_intensity1d)
    assert isinstance(pdfgen, PDFGenerator1D)


def test_get_pdf(reduced_intensity1d):
    pdfgen = PDFGenerator1D(reduced_intensity1d)
    pdf = pdfgen.get_pdf(s_min=0, s_max=9)
    assert isinstance(pdf, PairDistributionFunction1D)


def test_s_limits(reduced_intensity1d):
    pdfgen = PDFGenerator1D(reduced_intensity1d)
    pdf = pdfgen.get_pdf(s_min=0)
    pdf2 = pdfgen.get_pdf(s_min=0, s_max=10)
    assert np.array_equal(pdf.data, pdf2.data)


def test_s_limit_failure(reduced_intensity1d):
    pdfgen = PDFGenerator1D(reduced_intensity1d)

    with pytest.raises(
        ValueError, match="User specified s_max is larger than the maximum"
    ):
        pdf3 = pdfgen.get_pdf(s_min=0, s_max=15)


def test_signal_size():
    spectrum = np.array([5.0, 4.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    ri = ReducedIntensity1D(spectrum)
    pdfgen = PDFGenerator1D(ri)
    pdf = pdfgen.get_pdf(s_min=0, s_max=10)
    assert isinstance(pdf, PairDistributionFunction1D)

    ri = ReducedIntensity1D([spectrum])
    pdfgen = PDFGenerator1D(ri)
    pdf = pdfgen.get_pdf(s_min=0, s_max=10)
    assert isinstance(pdf, PairDistributionFunction1D)

    ri = ReducedIntensity1D([[spectrum]])
    pdfgen = PDFGenerator1D(ri)
    pdf = pdfgen.get_pdf(s_min=0, s_max=10)
    assert isinstance(pdf, PairDistributionFunction1D)

    ri = ReducedIntensity1D([[[spectrum]]])
    pdfgen = PDFGenerator1D(ri)
    pdf = pdfgen.get_pdf(s_min=0, s_max=10)
    shape = pdf.data.shape
    assert shape == (1, 1, 1, 2000)

    ri = ReducedIntensity1D([[[spectrum]]])
    pdfgen = PDFGenerator1D(ri)
    pdf = pdfgen.get_pdf(s_min=0, s_max=10, r_min=0, r_max=8)
    shape = pdf.data.shape
    assert shape == (1, 1, 1, 800)
