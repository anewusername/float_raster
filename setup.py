#!/usr/bin/env python3

from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    long_description = f.read()

with open('float_raster/VERSION.py', 'rt') as f:
    version = f.readlines()[2].strip()

setup(name='float_raster',
      version=version,
      description='High-precision anti-aliasing polygon rasterizer',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/code/jan/float_raster',
      packages=find_packages(),
      package_data={
          'float_raster': ['py.typed']
      },
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
