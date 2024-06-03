"""
Centering the Zero Beam
=======================
"""

# %%
# Making a Dummy Dataset with a Zero Beam that systematically deviates from the center
import hyperspy.api as hs
import pyxem as pxm

s = pxm.data.tilt_boundary_data(correct_pivot_point=False)

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
shifts.get_linear_plane()  # Making a linear plane to remove the systematic shift
hs.plot.plot_images(shifts.T)  # Plotting the shifts after making a linear plane

# Centering the Zero Beam
# -----------------------
# The zero beam can be centered using the :meth:`center_direct_beam` method.

centered = s.center_direct_beam(shifts=shifts, inplace=False)
pacbed_centered = (
    centered.sum()
)  # Plotting the sum of the dataset to check that the zero beam is centered
pacbed = (
    s.sum()
)  # Plotting the sum of the dataset to check that the zero beam is centered

hs.plot.plot_images([pacbed, pacbed_centered], label=["Original", "Centered"])
# %%
