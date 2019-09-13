# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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
from pyxem.generators.pdf_generator import PDFGenerator
from pyxem.signals.reduced_intensity_profile import ReducedIntensityProfile
from pyxem.signals.pdf_profile import PDFProfile


@pytest.fixture
def reduced_intensity_profile():
    data = np.ones((1, 10)) * np.arange(4).reshape(4, 1)
    data = data.reshape(2, 2, 10)
    ri = ReducedIntensityProfile(data)
    return ri


def test_pdf_gen_init(reduced_intensity_profile):
    pdfgen = PDFGenerator(reduced_intensity_profile)
    assert isinstance(pdfgen, PDFGenerator)


def test_get_pdf(reduced_intensity_profile):
    pdfgen = PDFGenerator(reduced_intensity_profile)
    pdf = pdfgen.get_pdf(s_cutoff=[0, 9])
    assert isinstance(pdf, PDFProfile)


def test_s_limits(reduced_intensity_profile):
    pdfgen = PDFGenerator(reduced_intensity_profile)
    pdf = pdfgen.get_pdf(s_cutoff=[0, 12])
    pdf2 = pdfgen.get_pdf(s_cutoff=[0, 10])
    assert np.array_equal(pdf.data, pdf2.data)


def test_signal_size():
    spectrum = np.array([5., 4., 3., 2., 2., 1., 1., 1., 0., 0.])
    ri = ReducedIntensityProfile(spectrum)
    pdfgen = PDFGenerator(ri)
    pdf = pdfgen.get_pdf(s_cutoff=[0, 10])
    assert isinstance(pdf, PDFProfile)

    ri = ReducedIntensityProfile([spectrum])
    pdfgen = PDFGenerator(ri)
    pdf = pdfgen.get_pdf(s_cutoff=[0, 10])
    assert isinstance(pdf, PDFProfile)

    ri = ReducedIntensityProfile([[spectrum]])
    pdfgen = PDFGenerator(ri)
    pdf = pdfgen.get_pdf(s_cutoff=[0, 10])
    assert isinstance(pdf, PDFProfile)

    ri = ReducedIntensityProfile([[[spectrum]]])
    pdfgen = PDFGenerator(ri)
    pdf = pdfgen.get_pdf(s_cutoff=[0, 10])
    assert pdf is None