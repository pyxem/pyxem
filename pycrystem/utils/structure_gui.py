import ipywidgets as ipw
from IPython.display import display, clear_output
from pymatgen import Element, Lattice, Structure

lattices = ['cubic', 'hexagonal', 'rhombohedral', 'tetragonal', 'orthorhombic',
            'monoclinic', 'triclinic']
functions = [Lattice.cubic, Lattice.hexagonal, Lattice.rhombohedral,
             Lattice.tetragonal, Lattice.orthorhombic, Lattice.monoclinic,
             Lattice.from_parameters]
parameters = [('a',), ('a', 'c'), ('a', 'alpha'), ('a', 'c'), ('a', 'b', 'c'),
              ('a', 'b', 'c', 'beta'),
              ('a', 'b', 'c', 'alpha', 'beta', 'gamma')]

lattice_choices = dict(zip(lattices, zip(functions, parameters)))


class Site:

    def __init__(self, element, x, y, z):
        self.element = element
        self.x = x
        self.y = y
        self.z = z


class StructureUI:

    def __init__(self):

        self.lattice = lattice_choices['cubic']
        self.parameters = {'a': 3.}
        self.sites = []
        self.widgets = self.create_widgets()
        self.display()

    def create_widgets(self):
        widgets = [
            self.create_lattice_choice_widget(),
            self.create_param_widgets(),
            self.create_site_widgets(),
            self.create_save_widget(),
        ]
        return widgets

    def display(self):
        for widget in self.widgets:
            widget.close()
        clear_output()
        self.widgets = self.create_widgets()
        for widget in self.widgets:
            display(widget)

    def create_lattice_choice_widget(self):

        lattice_choice_widget = ipw.Dropdown(
            options=lattice_choices,
            description='Lattice:',
            value=self.lattice
        )

        def on_lattice_choice_change(change):
            self.lattice = lattice_choice_widget.value
            self.parameters = {}
            self.display()

        lattice_choice_widget.observe(on_lattice_choice_change, names='value')

        return lattice_choice_widget

    def create_param_widgets(self):

        container = []

        for parameter in self.lattice[1]:
            container.append(self.create_param_widget(parameter))

        param_container = ipw.HBox(container)
        return param_container

    def create_param_widget(self, parameter):
        value_default = 0.
        value_max = None
        if parameter in ('a', 'b', 'c'):
            value_default = 3.
            value_max = 1e6
        elif parameter in ('alpha', 'beta', 'gamma'):
            value_default = 90.
            value_max = 180.
        if parameter in self.parameters:
            value = self.parameters[parameter]
        else:
            value = value_default
        parameter_widget = ipw.BoundedFloatText(
            value=value,
            min=0.,
            max=value_max,
            description=parameter
        )

        self.parameters[parameter] = value

        def on_parameter_change(change):
            self.parameters[parameter] = parameter_widget.value

        parameter_widget.observe(on_parameter_change, names='value')

        return parameter_widget

    def create_site_widgets(self):

        container = []

        for site in self.sites:
            container.append(self.create_site_widget(site))

        add_site_widget = ipw.Button(
            description='Add site',
            button_style='info',
            icon='plus',
        )

        def on_add_site(change):
            if len(self.sites) > 0:
                last_site = self.sites[-1]
            else:
                last_site = Site(Element.H, 0, 0, 0)
            new_site = Site(last_site.element, last_site.x, last_site.y,
                            last_site.z)
            self.sites.append(new_site)
            self.display()

        add_site_widget.on_click(on_add_site)

        container.append(add_site_widget)

        return ipw.VBox(container)

    def create_site_widget(self, site=None):

        if site is None:
            site = Site(Element.H, 0, 0, 0)

        container = []

        elements = Element.__members__
        element_names = sorted(elements.keys(),
                               key=lambda element: elements[element].number)
        elements = sorted(elements.values(), key=lambda element: element.number)

        element_choice_widget = ipw.Dropdown(
            options=dict(zip(element_names, elements)),
            value=site.element,
            description='Element:'
        )

        def on_element_choice(change):
            site.element = element_choice_widget.value

        element_choice_widget.observe(on_element_choice, names='value')

        container.append(element_choice_widget)

        x_axis_widget = ipw.BoundedFloatText(value=getattr(site, 'x'), min=0.0,
                                             max=1.0, description='x')
        y_axis_widget = ipw.BoundedFloatText(value=getattr(site, 'y'), min=0.0,
                                             max=1.0, description='y')
        z_axis_widget = ipw.BoundedFloatText(value=getattr(site, 'z'), min=0.0,
                                             max=1.0, description='z')

        def x_axis_change(change):
            site.x = x_axis_widget.value

        def y_axis_change(change):
            site.y = y_axis_widget.value

        def z_axis_change(change):
            site.z = z_axis_widget.value

        x_axis_widget.observe(x_axis_change, names='value')
        y_axis_widget.observe(y_axis_change, names='value')
        z_axis_widget.observe(z_axis_change, names='value')

        container.extend([x_axis_widget, y_axis_widget, z_axis_widget])

        remove_site_widget = ipw.Button(
            button_style='danger',
            icon='times',
        )

        def on_remove_site(change):
            self.sites.remove(site)
            self.display()

        remove_site_widget.on_click(on_remove_site)

        container.append(remove_site_widget)

        return ipw.HBox(container)

    def create_save_widget(self):

        filename_widget = ipw.Text(placeholder='e.g. nickel',
                                   description='File name:')
        save_widget = ipw.Button(description="Save", button_style="success",
                                 icon="floppy-o")

        def on_save(change):
            if filename_widget.value is "":
                raise ValueError("Please supply a file name.")
            self.get_structure().to(
                filename="{}.cif".format(filename_widget.value))

        save_widget.on_click(on_save)

        return ipw.HBox([filename_widget, save_widget])

    def get_structure(self):
        lattice = self.lattice[0](**self.parameters)
        species = [site.element for site in self.sites]
        coordinates = [[site.x, site.y, site.z] for site in self.sites]
        structure = Structure(lattice, species, coordinates)
        return structure
