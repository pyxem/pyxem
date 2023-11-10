"""
Determining Elliptic Distortion
===============================
This example shows how to determine the elliptic distortion of an diffraction pattern from uncorrected
2-fold astigmatism.  Determining the exact elliptic distortion is important for tasks such as

- Orientation mapping
- Angular Correlation (AC)
- Fluctuation Electron Microscopy (FEM)
- And many other 4-D STEM techniques
"""


########################################################################################
# First, we import the necessary packages and make a single test diffraction pattern.
import pyxem.dummy_data.make_diffraction_test_data as mdtd
import pyxem as pxm

test_data = mdtd.MakeTestData(200, 200, default=False)
test_data.add_disk(x0=100, y0=100, r=5, intensity=30)
test_data.add_ring_ellipse(x0=100, y0=100, semi_len0=63, semi_len1=70, rotation=45)
s = test_data.signal
s.set_signal_type("electron_diffraction")
s.plot()

# %%

"""
Using RANSAC Ellipse Determination
----------------------------------
The RANSAC algorithm is a robust method for fitting an ellipse to points with outliers. Here we
use it to determine the elliptic distortion of the diffraction pattern and exclude the zero
beam as "outliers".  We can also manually exclude other points from the fit by using the
mask parameter.
"""

center, affine, params, pos, inlier = pxm.utils.ransac_ellipse_tools.determine_ellipse(
    s, use_ransac=True, return_params=True
)

el, in_points, out_points = pxm.utils.ransac_ellipse_tools.ellipse_to_markers(
    ellipse_array=params, points=pos, inlier=inlier
)

s.plot()
s.add_marker(in_points, plot_marker=True)
s.add_marker(el, plot_marker=True)
s.add_marker(out_points, plot_marker=True)
# %%

"""
Using Manual Ellipse Determination
----------------------------------
Sometimes it is useful to force the ellipse to fit certain points.  For example, here we
can force the ellipse to fit the first ring by masking the zero beam.
"""

mask = s.get_direct_beam_mask(radius=20)

center, affine, params, pos = pxm.utils.ransac_ellipse_tools.determine_ellipse(
    s,
    return_params=True,
    mask=mask.data,
    num_points=500,
    use_ransac=False,
)
el, in_points = pxm.utils.ransac_ellipse_tools.ellipse_to_markers(
    ellipse_array=params,
    points=pos,
)

s.plot()
s.add_marker(in_points, plot_marker=True)
s.add_marker(el, plot_marker=True)

# %%
