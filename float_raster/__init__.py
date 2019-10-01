"""
Module for rasterizing polygons, with float-precision anti-aliasing on
 a non-uniform rectangular grid.

See the documentation for float_raster.raster(...) for details.
"""

import pathlib

from .float_raster import *

__author__ = 'Jan Petykiewicz'

with open(pathlib.Path(__file__).parent / 'VERSION', 'r') as f:
    __version__ = f.read().strip()

