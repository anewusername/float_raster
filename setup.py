#!/usr/bin/env python3

from setuptools import setup
import float_raster

setup(name='float_raster',
      version=float_raster.version,
      description='High-precision anti-aliasing polygon rasterizer',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/gogs/jan/float_raster',
      py_modules=['float_raster'],
      install_requires=[
            'numpy',
            'scipy',
      ],
      )
