#!/usr/bin/env python3

from setuptools import setup
import float_raster

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='float_raster',
      version=float_raster.version,
      description='High-precision anti-aliasing polygon rasterizer',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/code/jan/float_raster',
      py_modules=['float_raster'],
      install_requires=[
            'numpy',
            'scipy',
      ],
      classifiers=[
            'Programming Language :: Python :: 3',
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Information Technology',
            'Intended Audience :: Manufacturing',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU Affero General Public License v3',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Multimedia :: Graphics :: Graphics Conversion',
      ],
      )
