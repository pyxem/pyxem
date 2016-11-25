from setuptools import setup
setup(
        name = 'higher_order_laue_zone_calculation',
        packages = [
            'higher_order_laue_zone_calculation',
            ],
        version = '0.1.0',
        description = 'Library for analysing higher order laue zones from 4D datasets',
        keywords = [
            'STEM',
            'data analysis',
            'microscopy',
            ],
        install_requires = [
            'scipy',
            'numpy>=1.10',
            'h5py',
            'ipython>=2.0',
            'matplotlib>=1.2',
            'hyperspy>=1.0.1',
            ],
)
