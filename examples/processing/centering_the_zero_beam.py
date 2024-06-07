"""
Centering the Zero Beam
=======================
"""

# %%
# Making a Dummy Dataset with a Zero Beam that systematically deviates from the center
import hyperspy.api as hs
import pyxem as pxm

s = pxm.data.tilt_boundary_data(correct_pivot_point=False)

# %%
# Getting the Position of the Zero beam
# -------------------------------------
# The zero beam position can be obtained using the :meth:`get_direct_beam_position` method.

s_shifts = s.get_direct_beam_position(method="blur", sigma=5, half_square_width=20)
s_shifts.plot()  # Plotting the shifts

# Getting a Linear Plane
# ---------------------
# In many instances the zero beam position will vary systematically with the scan position.
# This can be corrected by fitting a linear plane to the zero beam position using the
# :meth:`get_linear_plane` method.
s_linear_plane = s_shifts.get_linear_plane()  # Making a linear plane to remove the systematic shift
s_linear_plane.plot()  # Plotting the shifts after making a linear plane

# %%
# Centering the Zero Beam
# -----------------------
# The zero beam can be centered using the :meth:`center_direct_beam` method.

s_centered = s.center_direct_beam(shifts=s_linear_plane, inplace=False)
s_pacbed_centered = s_centered.sum()  # Plotting the sum of the dataset to check that the zero beam is centered
s_pacbed = s.sum()  # Plotting the sum of the non-centered dataset to check that the zero beam is centered

hs.plot.plot_images([s_pacbed, s_pacbed_centered], label=["Original", "Centered"])
# %%
