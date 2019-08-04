import runpy
from setuptools import setup

__version__ = runpy.run_path('heimdall/version.py')['__version__']
setup(name='heimdall',
      version=__version__,
      author='Constantin Pape',
      url='',
      license='MIT')
      # TODO
      # scripts='scrips/view_contanier')
