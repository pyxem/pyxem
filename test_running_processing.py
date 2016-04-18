import stem_diffraction_radial_integration as sdri
import laue_zone_processing as lzp
import hyperspy.api as hs

s_radial = hs.load("default1_radial.hdf5")
s_radial = s_radial.inav[:,14:]
s_lfo = s_radial.isig[47.:78.]
s_sto = s_radial.isig[77.:107.]
m_lfo_one_gaussian = lzp.model_lfo_with_one_gaussian(s_lfo)
m_sto_gaussian = lzp.model_sto(s_sto)
m_lfo_one_gaussian.save("model_lfo_one_gaussian.hdf5")
m_sto_gaussian.save("model_sto_gaussian.hdf5")
