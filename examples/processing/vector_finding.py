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

s.find_peaks(interactive=True)  # find the peaks using the interactive peak finder

# %%

"""
Template Matching
=================

The best method for finding peaks is usually through template matching.  In this case a disk with
some radius is used as the template.  The radius of the disk should be chosen to be the same size
as the diffraction spots.  The template matching is done using the :meth:`template_match_disk` method.

This can also be done lazy, including the plotting of the markers!
"""
s.axes_manager[2].scale = 0.3


temp_small = s.template_match_disk(disk_r=3, subtract_min=False)  # Too small
temp = s.template_match_disk(disk_r=5, subtract_min=False)  # Just right
temp_large = s.template_match_disk(disk_r=7, subtract_min=False)  # Too large
ind = (5, 5)
hs.plot.plot_images(
    [temp_small.inav[ind], temp.inav[ind], temp_large.inav[ind]],
    label=["Too Small", "Just Right", "Too Large"],
)

vectors = temp.get_diffraction_vectors(threshold_abs=0.4, min_distance=5)

# %%
# Plotting Peaks
# ==============
# We can plot the peaks using hyperSpy's markers and DiffractionVectors.

s.plot()
s.add_marker(vectors.to_markers(color="red", sizes=10, alpha=0.5))

# %%
# Subpixel Peak Fitting
# =====================
#
# The template matching is done on the pixel grid.  To find the peak position more accurately the correlation
# can be up-sampled using the :func:`pyxem.signals.DiffractionVectors.subpixel_refine` method.  This method takes a
# `DiffractionSignal2D` object and uses that to refine the peak positions.
#
# This only really works up to up-sampling of 2-4. There is little improvement with increased up-sampling while
# it greatly increases the computation time.

refined_peaks_com = vectors.subpixel_refine(s, "center-of-mass", square_size=20)
refined_peaks_xc = vectors.subpixel_refine(
    s, "cross-correlation", square_size=20, upsample_factor=2, disk_r=5
)

markers2 = refined_peaks_com.to_markers(color="blue", sizes=10, alpha=0.25)
markers3 = refined_peaks_xc.to_markers(color="green", sizes=10, alpha=0.25)


s.plot()
s.add_marker(vectors.to_markers(color="red", sizes=10, alpha=0.25))
s.add_marker(markers2)
s.add_marker(markers3)

# %%

# sphinx_gallery_thumbnail_number = 3
