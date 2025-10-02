"""
=====================
Other Virtual Imaging
=====================
"""

import pyxem as pxm
import hyperspy.api as hs

vectors = pxm.utils.virtual_images_utils.get_vectors_mesh(
    10, 10, 40, angle=0.0, shear=0.0
)
rois, labels = pxm.signals.DiffractionVectors2D(vectors).to_roi(
    radius=4, include_labels=True
)
c = pxm.data.dummy_data.get_cbed_signal()
c.calibration(center=None)
c.plot()
for r in rois:
    r.add_widget(c, axes=(2, 3))
c.add_marker(labels)
# %%

vdfs = c.get_virtual_image(rois)

vdfs.plot()

# %%
# We can also use multiple different ROIs and combine them into a virtual image.

r1 = hs.roi.RectangularROI(-35, -35, 35, 35)
c1 = hs.roi.CircleROI(cx=25.5, cy=24.5, r=7, r_inner=0)
c2 = hs.roi.CircleROI(cx=0, cy=0, r=30.5, r_inner=10.0)

c.plot()
for r in [r1, c1, c2]:
    r.add_widget(c, axes=(2, 3))

vdfs = c.get_virtual_image([r1, c1, c2])

vdfs.plot()
# %%
