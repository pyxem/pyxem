"""
Finding Diffraction Vectors
===========================
"""

# %%
# This example shows how to find the diffraction vectors for a given
# signal and then plot them using hyperspy's markers.


import pyxem as pxm
import hyperspy.api as hs

s = pxm.data.tilt_boundary_data()
# %%

s.find_peaks(iteractive=True)  # find the peaks using the interactive peak finder

# %%

"""
Template Matching
=================

The best method for finding peaks is usually through template matching.  In this case a disk with
some radius is used as the template.  The radius of the disk should be chosen to be the same size
as the diffraction spots.  The template matching is done using the :meth:`template_match_disk` method.
"""


temp_small = s.template_match_disk(disk_r=3)  # Too small
temp = s.template_match_disk(disk_r=5)  # Just right
"""
Finding Diffraction Vectors
===========================
"""

# %%
# This example shows how to find the diffraction vectors for a given
# signal and then plot them using hyperspy's markers.


import pyxem as pxm
import hyperspy.api as hs

s = pxm.data.tilt_boundary_data()
# %%

# s.find_peaks(iteractive=True)  # find the peaks using the interactive peak finder

# %%

"""
Template Matching
=================

The best method for finding peaks is usually through template matching.  In this case a disk with
some radius is used as the template.  The radius of the disk should be chosen to be the same size
as the diffraction spots.  The template matching is done using the :meth:`template_match_disk` method.
"""


temp_small = s.template_match_disk(disk_r=3, subtract_min=False)  # Too small
temp = s.template_match_disk(disk_r=5, subtract_min=False)  # Just right
temp_large = s.template_match_disk(disk_r=7, subtract_min=False)  # Too large
ind = (5, 5)
hs.plot.plot_images(
    [temp_small.inav[ind], temp.inav[ind], temp_large.inav[ind]],
    label=["Too Small", "Just Right", "Too Large"],
)

pks = temp.find_peaks(interactive=False, threshold_abs=0.4, min_distance=5)
# %%
"""
Plotting Peaks
==============
We can plot the peaks using hyperSpy's markers and DiffractionVectors.
"""
vectors = pxm.signals.DiffractionVectors.from_peaks(
    pks
)  # calibration is automatically set
s.plot()

s.add_marker(vectors.to_markers(color="red", sizes=10, alpha=0.5))

"""
Subpixel Peak Fitting
=====================

The template matching is done on the pixel grid.  To find the peak position more accurately the correlation
can be upsampled using the :class:`pyxem.generators.SubpixelrefinementGenerator` class.  This class takes a
`DiffractionSignal2D` object and a `DiffractionVector` object as input.
"""
"""
from pyxem.generators.subpixelrefinement_generator import SubpixelrefinementGenerator
subpixel_gen = SubpixelrefinementGenerator(s, pks)
refined_peaks_com = subpixel_gen.center_of_mass_method(square_size=5)
# square_size is the size of the square around the peak to use for the center of mass calculation
refined_peaks_xc= subpixel_gen.conventional_xc(square_size=5,
                                                 disc_radius=5,
                                                 upsample_factor=3)
markers1 = pks.as_markers(colors='red',
                          sizes=.1,
                          alpha=0.5)
markers2 = refined_peaks_com.as_markers(colors='blue',
                                        sizes=.1,
                                        alhpa=0.5)

markers3 = refined_peaks_xc.as_markers(colors='green',
                                       sizes=.1,
                                       alhpa=0.5)

s.plot()
s.add_marker([markers1, markers2, markers3])
"""
# %%
