"""
Operations on vectors
=====================

This example shows how to perform some basic operations slicing and selecting vectors.
This is designed to be very flexible and powerful.  Many operations such as slicing with a
boolean array are supported.

Additionally, lazy operations are supported and can be chained together. These are often faster
than their non-lazy counterparts as ``dask`` very effectively prunes the computation graph.
"""
import pyxem as pxm
import hyperspy.api as hs

hs.set_log_level("ERROR")

s = pxm.data.tilt_boundary_data()
temp = s.template_match_disk(disk_r=5, subtract_min=False)
pks = temp.find_peaks(
    interactive=False, threshold_abs=0.4, min_distance=5, get_intensity=True
)
vectors = pxm.signals.DiffractionVectors.from_peaks(
    pks
)  # calibration is automatically set

# Plotting all the vectors

s.plot()
all_vectors = vectors.to_markers(color="red", sizes=10, alpha=0.5)
s.add_marker(all_vectors)
# %%

slic_vectors = (vectors.ivec[:, vectors.ivec[0] < 10]).to_markers(
    color="green", sizes=5, alpha=0.5
)

s.plot()
s.add_marker([all_vectors, slic_vectors])

# %%
slic_vectors = (
    vectors.ivec[:, (vectors.ivec[0] > 0) * (vectors.ivec[0] < 10)]
).to_markers(color="w", sizes=5, alpha=0.5)
s.plot()
s.add_marker([all_vectors, slic_vectors])
# %%

vect_magnitudes = (vectors.ivec[0] ** 2 + vectors.ivec[1] ** 2) ** 0.5
slic_vectors = vectors.ivec[:, vect_magnitudes < 20].to_markers(
    color="w", sizes=5, alpha=0.5
)
s.plot()
s.add_marker([all_vectors, slic_vectors])
s.add_marker([all_vectors, slic_vectors])
# %%


slic_vectors = (vectors.ivec[:, vectors.ivec["intensity"] < 0.5]).to_markers(
    color="w", sizes=5, alpha=0.5
)
s.plot()
s.add_marker([all_vectors, slic_vectors])
s.add_marker([all_vectors, slic_vectors])
# %%
