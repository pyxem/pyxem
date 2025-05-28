"""
Multi Phase Orientation Mapping
===============================
You can also calculate the orientation of the grains for multiple phases using the
:meth:`pyxem.signals.PolarSignal2D.get_orientation` method. This requires that you
simulate the entire S2 space for the phase and then compare to the simulated diffraction.

For more information on the orientation mapping process see :cite:`pyxemorientationmapping2022`
"""

import pyxem as pxm
from pyxem.data import fe_multi_phase_grains, fe_bcc_phase, fe_fcc_phase
from diffsims.generators.simulation_generator import SimulationGenerator
from orix.quaternion import Rotation
from orix.sampling import get_sample_reduced_fundamental
import hyperspy.api as hs

multi_phase = fe_multi_phase_grains()

# %%
# First we center the diffraction patterns and get a polar signal
# Increasing the number of npt_azim with give better polar sampling
# but will take longer to compute the orientation map
# The mean=True argument will return the mean pixel value in each bin rather than the sum
# this makes the high k values more visible

multi_phase.calibration.center = None
polar_multi = multi_phase.get_azimuthal_integral2d(
    npt=100, npt_azim=360, inplace=False, mean=True
)
polar_multi.plot()

# %%
# Now we can get make a simulation. In this case we want to set a minimum_intensity which removes the
# low intensity reflections. We also sample the S2 space using the :func:`orix.sampling.get_sample_reduced_fundamental`
# We have two phases here so we can make a simulation object with both of the phases.

bcc = fe_bcc_phase()
fcc = fe_fcc_phase()
bcc.name = "BCC Phase"
fcc.name = "FCC Phase"
fcc.color = "red"
bcc.color = "blue"

generator = SimulationGenerator(200, minimum_intensity=0.05)
rotations_bcc = get_sample_reduced_fundamental(
    resolution=1, point_group=bcc.point_group
)
rotations_fcc = get_sample_reduced_fundamental(
    resolution=1, point_group=fcc.point_group
)

sim = generator.calculate_diffraction2d(
    [bcc, fcc],
    rotation=[rotations_bcc, rotations_fcc],
    max_excitation_error=0.1,
    reciprocal_radius=2,
    with_direct_beam=False,
)
# %%
# Orientation Mapping
# -------------------
# Now we can calculate the orientation map using the polar signal and the simulated diffraction pattern
# Sometimes a gamma correction helps to re-normalize the intensity of the simulated diffraction pattern
# and increase the intensity of the high k diffraction spots. Unfortunately this is quite dependent on the
# detector response/ sample thickness etc. As a result there is no good rule of thumb.  As such it is important
# to play with this value/ use the background correction to get the best results.
# Additionally, here we have set the n_best to -1 which means that we will return all the
# orientations in the simulation. This is not always the best option as it can lead to
# memory spikes. It does allow us to plot a very nice heatmap of the orientation.
# Which is useful for understanding the orientation mapping process and mis-orientations.


polar_multi = polar_multi**0.5  # gamma correction
orientation_map = polar_multi.get_orientation(sim, n_best=-1, frac_keep=1)


# %%
# Plotting The Orientation Mapping
# --------------------------------
# We can then plot the orientation map using the :meth:`pyxem.signals.OrientationMap.plot_over_signal` method.
# Here we have set add_ipf_correlation_heatmap=True which will add a heatmap of the IPF correlation.

orientation_map.plot_over_signal(
    multi_phase, vmax="99th", add_ipf_correlation_heatmap=True
)

# %%
# Getting the Crystal Map
# -----------------------
# Now we can calculate the orientation map using the polar signal and the simulated diffraction pattern
# Sometimes a gamma correction helps to re-normalize the intensity of the simulated diffraction pattern
# and increase the intensity of the high k diffraction spots. Unfortunately this is quite dependent on the
# detector response/ sample thickness etc. As a result there is no good rule of thumb.  As such it is important
# to play with this value/ use the background correction to get the best results.

cmap = orientation_map.to_crystal_map()
cmap.plot()
# %%
# sphinx_gallery_thumbnail_number = 4
