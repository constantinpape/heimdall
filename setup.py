import runpy
import itertools
from setuptools import setup, find_packages

__version__ = runpy.run_path('heimdall/version.py')['__version__']

requires = [
    "napari",
    "numpy",
]

extras = {
    "hdf5": ["h5py"],
}

extras["all"] = list(itertools.chain.from_iterable(extras.values()))

setup(
    name='heimdall',
    packages=find_packages(include='heimdall'),
    version=__version__,
    author='Constantin Pape',
    install_requires=requires,
    extras_require=extras,
    url='https://github.com/constantinpape/heimdall',
    license='MIT',
    entry_points={
        "console_scripts": ["view_container = heimdall.scripts.view_container:main"]
    },
)
