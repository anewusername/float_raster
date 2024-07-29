"""
Module for rasterizing polygons, with float-precision anti-aliasing on
 a non-uniform rectangular grid.

See the documentation for float_raster.raster(...) for details.
"""

from .float_raster import (
    raster as raster,
    find_intersections as find_intersections,
    create_vertices as create_vertices,
    clip_vertices_to_window as clip_vertices_to_window,
    get_raster_parts as get_raster_parts,
    )


__author__ = 'Jan Petykiewicz'
__version__ = '0.8'
