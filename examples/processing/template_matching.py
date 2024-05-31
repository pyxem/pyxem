"""
Template Matching
=================

This example shows how the template matching is done in pyxem to find peaks in a diffraction pattern.
"""

import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt
from skimage.morphology import disk


import pyxem as pxm
import pyxem.data.dummy_data.make_diffraction_test_data as mdtd

s = pxm.data.tilt_boundary_data()

# %%
# How Template Matching Works
# ===========================
#
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
#
# Template Matching with an Amorphous Halo
# =========================================
#
# Sometimes the template matching can be thrown off by an amorphous halo around the diffraction spots.  This
# can be seen in the following example.  In this case we can use a dilated (circular) window to reduce the
# effect of the imposed square window. This can be seen in the intensity of the diffraction vectors at
# +y,+x and -y,-x.

data = mdtd.generate_4d_data(
    image_size_x=s.axes_manager.signal_axes[0].size,
    image_size_y=s.axes_manager.signal_axes[1].size,
    disk_x=s.axes_manager.signal_axes[0].size // 2,
    disk_y=s.axes_manager.signal_axes[0].size // 2,
    ring_r=45 / 128 * s.axes_manager.signal_axes[0].size // 2,
    ring_x=s.axes_manager.signal_axes[0].size // 2,
    ring_y=s.axes_manager.signal_axes[0].size // 2,
    disk_I=10,
    ring_lw=8,
    ring_I=2,
)
amorphous_data = data + s

template_normal = amorphous_data.template_match_disk(disk_r=6, subtract_min=False)
template_circular = amorphous_data.template_match_disk(
    disk_r=6, circular_background=True, template_dilation=5, subtract_min=False
)
mask = template_normal.get_direct_beam_mask(35)

mask2 = ~template_normal.get_direct_beam_mask(55)

template_normal.data[:, :, mask] = 0
template_circular.data[:, :, mask] = 0

template_normal.data[:, :, mask2] = 0
template_circular.data[:, :, mask2] = 0

f = plt.figure(figsize=(15, 5))
ind = (5, 5)
hs.plot.plot_images(
    [amorphous_data.inav[ind], template_normal.inav[ind], template_circular.inav[ind]],
    label=["Signal", "Square Window", "Circular Window"],
    tight_layout=True,
    per_row=3,
    vmin=[0, 0.7, 0.7],
    fig=f,
)

# %%
