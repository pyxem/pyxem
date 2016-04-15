import stem_diffraction_radial_integration as sdri
import laue_zone_processing as lzp
import hyperspy.api as hs

#sdri.save_fpd_dataset_as_radial_profile_signal("default1.hdf5")

s_radial = hs.load("default1_radial.hdf5")
s_lfo = s_radial.isig[47.:78.]
m_lfo_one_gaussian = lzp.model_lfo_with_one_gaussian(s_lfo)
m_lfo_two_gaussian = lzp.model_lfo_with_two_gaussians(s_lfo)
