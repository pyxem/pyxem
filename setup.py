from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
        name = 'pixstem',
        packages = [
            'pixstem',
            ],
        version = '0.3.1',
        description = 'Library for processing scanning transmission electron microscopy data acquired using a pixelated detector',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author = 'Magnus Nord',
        author_email = 'magnunor@gmail.com',
        license = 'GPL v3',
        url = 'http://pixstem.org/',
        download_url = 'https://gitlab.com/pixstem/pixstem/repository/archive.tar?ref=0.3.1',
        keywords = [
            'STEM',
            'data analysis',
            'microscopy',
            ],
        install_requires = [
            'scipy',
            'numpy>=1.13',
            'h5py',
            'ipython>=2.0',
            'matplotlib>=2.0',
            'hyperspy>=1.3',
            'dask',
            ],
)
