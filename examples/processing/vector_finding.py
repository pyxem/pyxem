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
can be upsampled using the :func:`pyxem.signals.DiffractionVectors.subpixel_refine` method.  This method takes a
`DiffractionSignal2D` object and uses that to refine the peak positions.
"""
refined_peaks_com = vectors.subpixel_refine("center_of_mass", square_size=5)
refined_peaks_xc = vectors.subpixel_refine(
    "cross-correlation", disk_r=5, upsample_factor=2, square_size=5
)
refined_peaks_pxc = vectors.subpixel_refine(
    "phase_cross_correlation", disk_r=5, upsample_factor=2, square_size=5
)
markers1 = refined_peaks_com.as_markers(colors="red", sizes=0.1, alpha=0.5)
markers2 = refined_peaks_xc.as_markers(colors="blue", sizes=0.1, alhpa=0.5)

markers3 = refined_peaks_pxc.as_markers(colors="green", sizes=0.1, alhpa=0.5)

s.plot()
s.add_marker([markers1, markers2, markers3])
# %%
