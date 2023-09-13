"""
Centering the Zero Beam
=======================
"""

# %%
# Making a Dummy Dataset with a Zero Beam that systematically deviates from the center
import numpy as np
from pyxem.dummy_data import make_diffraction_test_data as mdtd
import hyperspy.api as hs
di = mdtd.DiffractionTestImage(intensity_noise=False)
di.add_disk(x=128, y=128, intensity=10.)  # Add a zero beam disk at the center
di.add_cubic_disks(vx=20, vy=20, intensity=2., n=5)
di.add_background_lorentz()
di_rot = di.copy()
di_rot.rotation = 10
dtd = mdtd.DiffractionTestDataset(10, 10, 256, 256)
position_array = np.ones((10, 10), dtype=bool)
position_array[:5] = False
dtd.add_diffraction_image(di, position_array)
dtd.add_diffraction_image(di_rot, np.invert(position_array))
s = dtd.get_signal()

# Shifting the zero beam away from the center
xx, yy = np.meshgrid(range(10), range(10))
shifts = np.stack([xx*0.5, yy*0.5], axis=-1)
s.center_direct_beam(shifts=hs.signals.Signal1D(shifts))


# Getting the Position of the Zero beam
# -------------------------------------
# The zero beam position can be obtained using the :meth:`get_direct_beam_position` method.

shifts = s.get_direct_beam_position(method="blur", sigma=5, half_square_width=20)
hs.plot.plot_images(shifts.T)  # Plotting the shifts
# %%

# Making a Linear Plane
# ---------------------
# In many instances the zero beam position will vary systematically with the scan position.
# This can be corrected by fitting a linear plane to the zero beam position using the
# :meth:`make_linear_plane` method.
shifts.make_linear_plane()  # Making a linear plane to remove the systematic shift
hs.plot.plot_images(shifts.T)  # Plotting the shifts after making a linear plane

# Centering the Zero Beam
# -----------------------
# The zero beam can be centered using the :meth:`center_direct_beam` method.

centered = s.center_direct_beam(shifts=shifts, inplace=False)
pacbed_centered = centered.sum()  # Plotting the sum of the dataset to check that the zero beam is centered
pacbed = s.sum()  # Plotting the sum of the dataset to check that the zero beam is centered

hs.plot.plot_images([pacbed, pacbed_centered], label=["Original", "Centered"])
# %%

