import hyperspy.api as hs
import laue_zone_plotting as lzp

s_lfo = hs.load("model_lfo_one_gaussian.hdf5")
m_lfo = s_lfo.models['a'].restore()
s_sto = hs.load("model_sto_gaussian.hdf5")
m_sto = s_sto.models['a'].restore()

s_radial = hs.load("default1_radial.hdf5")

lzp.plot_lfo_sto_laue_zone_report(
    m_lfo,
    m_sto,
    s_radial)
