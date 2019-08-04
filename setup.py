import runpy
from setuptools import setup

__version__ = runpy.run_path('heimdall/version.py')['__version__']
setup(name='heimdall',
      version=__version__,
      author='Constantin Pape',
      url='https://github.com/constantinpape/heimdall',
      license='MIT',
      scripts='scrips/view_contanier')
