"""
Magnetic DPC
============

This example shows a basic workflow for doing magnetic DPC on a simple simulated test
dataset. The dataset consists of alternating stripes of switching magnetic fields resulting
in reversals of the direction of the magnetic field.

This is a generalization of some of the real datasets which are more completely analyzed in
the longer Demo Notebooks.
"""

import pyxem as pxm

s = pxm.data.simulated_stripes()

beam_pos = s.get_direct_beam_position(
    method="center_of_mass", half_square_width=20
)  # Just the center of the DP

beam_pos.plot()

# %%
# Visualizing the Shifts
# -----------------------
# The shifts can be visualized using the :meth:`plot` method which shows both the
# x and y shifts in a single plot.

s.plot()

# %%
# Plotting Magnitude and Direction
# --------------------------------
