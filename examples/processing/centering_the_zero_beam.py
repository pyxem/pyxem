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
s_shifts.plot()

# Getting a Linear Plane
# ---------------------
# In many instances the zero beam position will vary systematically with the scan position.
# This can be corrected by fitting a linear plane to the zero beam position using the
# :meth:`get_linear_plane` method.
s_linear_plane = s_shifts.get_linear_plane()
s_linear_plane.plot()

# %%
# Centering the Zero Beam
# -----------------------
# The zero beam can be centered using the :meth:`center_direct_beam` method.
# Then we sum all the diffraction patterns for the both the centered beam,
# and the non-centered one, to compare them.

s_centered = s.center_direct_beam(shifts=s_linear_plane, inplace=False)
s_pacbed_centered = s_centered.sum()
s_pacbed = s.sum()

hs.plot.plot_images([s_pacbed, s_pacbed_centered], label=["Original", "Centered"])
# %%
