import numpy as np
import pymatgen as pmg

def generate_pyprismatic_input(unit_cell_structure,scaling_for_supercell=1,comment="Default Comment"):
        """ Generates a .xyz file on which Pyprismatic can run

        Args:

        unit_cell_structure: pymatgen.Structure object

        scaling: int or (3,1) 
            
        Goes into pymatgen's 'make_supercell' function, can be int or (3,1) array,
        refer to their docs for details    
            
        comment: str
        
        Be away that this cannot wrap to the next line, and should not contain '\n' characters

        """
        lattice_params = unit_cell_structure.lattice.abc
        
        line_1 = comment
        line_2 = lattice_params

        #TODO figure out a better code layout for this
        unit_cell_structure.make_supercell(scaling_for_supercell)
        
        # TODO vectorise this
        Z_list = []
        for element in unit_cell_structure.species:
                Z_list.append(int(element.Z))

        # TODO Get dw from Element's names
        dw = 0.076
        atomic_numbers = np.asarray(Z_list).reshape(len(unit_cell_structure.species),1)
        cart_coords = unit_cell_structure.cart_coords
        percent_occupied  = np.ones_like(atomic_numbers)
        debeye_waller = dw*np.ones_like(atomic_numbers)
        # combines the rest of what needs to come out
        printing_array = np.hstack([atomic_numbers,cart_coords,percent_occupied,debeye_waller])

        print(line_1)
        print('    {0:.3g}   {1:.3g}   {2:.3g}'.format(line_2[0],line_2[1],line_2[2]))
        for row in printing_array:
                print('{0:.3g} {1:.4f} {2:.4f} {3:.4f} {4:.3f} {5:.3f}'.format(row[0],row[1],row[2],row[3],row[4],row[5]))
        print("-1")
        return None
