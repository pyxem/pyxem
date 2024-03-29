"""
Template Matching
=================

This example shows how the template matching is done in pyxem to find peaks in a diffraction pattern.
"""

from skimage.morphology import disk
import numpy as np
import hyperspy.api as hs

import pyxem as pxm
import pyxem.dummy_data.make_diffraction_test_data as mdtd

s = pxm.data.tilt_boundary_data()

# How Template Matching Works
# ===========================

# Pyxem uses a window-normalized cross-correlation to find the peaks.  This is much better for finding
# both strongly and weakly scattering peaks but sometimes if the window is too small, too large, or the
# wrong shape, the behavior can be unexpected.

template = disk(5)
padded_template = np.pad(template, 3)  # padding a template increased the window size
padded_template_large = np.pad(
    template, 20
)  # padding a template increased the window size

template_normal = s.template_match(template)
template_padded = s.template_match(padded_template)
template_padded_large = s.template_match(padded_template_large)

ind = (5, 5)
hs.plot.plot_images(
    [
        s.inav[ind],
        template_normal.inav[ind],
        template_padded.inav[ind],
        template_padded_large.inav[ind],
    ],
    label=["Signal", "Normal Window", "Large Window", "Very Large Window"],
    tight_layout=True,
    per_row=2,
)

# %%

# In the very large window case you can see that in the middle the strong zero beam is __included__ in the window
# and the intensity for those pixels is suppressed in the template matching. We also see this occurs in a
# square pattern because even though our template is a disk, the window is a square!

# Template Matching with an Amorphous Halo
# =========================================

# Sometimes the template matching can be thrown off by an amorphous halo around the diffraction spots.  This
# can be seen in the following example.  In this case we can use a dilated (circular) window to reduce the
# effect of the imposed square window.
data = mdtd.generate_4d_data(
    image_size_x=256,
    image_size_y=256,
    disk_x=128,
    disk_y=128,
    ring_r=45,
    ring_x=128,
    ring_y=128,
    disk_I=10,
    ring_lw=6,
    ring_I=2,
)
amorphous_data = data + s


template_normal = amorphous_data.template_match_disk(disk_r=6)
template_circular = s.template_match_disk(disk_r=6, dilated_template_window=True)


ind = (5, 5)
hs.plot.plot_images(
    [amorphous_data.inav[ind], template_normal.inav[ind], template_circular.inav[ind]],
    label=["Signal", "Square Window", "Large Window"],
    tight_layout=True,
    per_row=3,
)
