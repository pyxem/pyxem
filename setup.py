from setuptools import setup
setup(
        name = 'fpd_data_processing',
        packages = [
            'fpd_data_processing',
            ],
        version = '0.2.0',
        description = 'Library for processing scanning transmission electron microscopy data acquired using a fast pixelated detector',
        author = 'Magnus Nord',
        author_email = 'magnunor@gmail.com',
        license = 'GPL v3',
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
            'matplotlib>=2.0',
            'hyperspy>=1.3',
            ],
)
