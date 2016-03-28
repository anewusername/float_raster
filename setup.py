#!/usr/bin/env python

from setuptools import setup

setup(name='float_raster',
      version='0.1',
      description='High-precision anti-aliasing polygon rasterizer',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/gogs/jan/float_raster',
      py_modules=['float_raster'],
      install_requires=[
            'numpy'
      ],
      )
