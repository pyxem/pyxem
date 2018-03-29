# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

class DiffractionLibrary(dict):
    """Maps crystal structure (phase) and orientation (Euler angles or
    axis-angle pair) to simulated diffraction data.
    """

    def get_pattern(self,phase=False,angle=False):
        """ Extracts a single pattern for viewing,
        unspecified layers of dict are selected randomly and so this method
        is not entirely repeatable
        
        
        Parameters
        ----------
        
        Phase : Label for the Phase you are interested in. Randomly chosen
        if False
        Angle : Label for the Angle you are interested in. Randomly chosen 
        if False
        
        """
        
        if phase:
            if angle:
                return self[phase][angle]['Sim']
            else:
                diff_lib = self[phase]
                for diffraction_pattern in diff_lib.values():
                    return diffraction_pattern['Sim']
        else:
            for key in self.keys():
                diff_lib = self[key]
                for diffraction_pattern in diff_lib.values():
                    return diffraction_pattern['Sim']

    def plot(self):
        """Plots the library interactively. This is a helper method and it is
        not tested.
        """
        from pyxem.signals.electron_diffraction import ElectronDiffraction
        sim_diff_dat = []
        for key in self.keys():
            for ori in self[key].keys():
                dpi = self[key][ori]['Sim'].as_signal(128, 0.03, 1)
                sim_diff_dat.append(dpi.data)
        ppt_test = ElectronDiffraction(sim_diff_dat)
        ppt_test.plot()
