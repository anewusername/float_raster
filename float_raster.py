"""
Module for rasterizing polygons, with float-precision anti-aliasing on
 a non-uniform rectangular grid.

See the documentation for raster(...) for details.
"""

from typing import Tuple
import numpy
from numpy import logical_and, diff, floor, ceil, ones, zeros, hstack, full_like, newaxis
from scipy import sparse

__author__ = 'Jan Petykiewicz'

version = '0.4'


def raster(vertices: numpy.ndarray,
           grid_x: numpy.ndarray,
           grid_y: numpy.ndarray
           ) -> numpy.ndarray:
    """
    Draws a polygon onto a 2D grid of pixels, setting pixel values equal to the fraction of the
     pixel area covered by the polygon. This implementation is written for accuracy and works with
     double precision, in contrast to most other implementations which are written for speed and
     usually only allow for 256 (and often fewer) possible pixel values without performing (very
     slow) super-sampling.

    Polygons are assumed to have clockwise vertex order; reversing the vertex order is equivalent
     to multiplying the result by -1.

    :param vertices: 2xN ndarray containing x,y coordinates for each vertex of the polygon
    :param grid_x: x-coordinates for the edges of each pixel (ie, the leftmost two columns span
        x=grid_x[0] to x=grid_x[1] and x=grid_x[1] to x=grid_x[2])
    :param grid_y: y-coordinates for the edges of each pixel (see grid_x)
    :return: 2D ndarray with pixel values in the range [0, 1] containing the anti-aliased polygon
    """
    vertices = numpy.array(vertices)
    grid_x = numpy.array(grid_x)
    grid_y = numpy.array(grid_y)

    min_bounds = floor(vertices.min(axis=1))
    max_bounds = ceil(vertices.max(axis=1))

    keep_x = logical_and(grid_x >= min_bounds[0],
                         grid_x <= max_bounds[0])
    keep_y = logical_and(grid_y >= min_bounds[1],
                         grid_y <= max_bounds[1])

    if not (keep_x.any() and keep_y.any()):  # polygon doesn't overlap grid
        return zeros((grid_x.size - 1, grid_y.size - 1))

    vertices = create_vertices(vertices, grid_x, grid_y)
    parts_grid = get_raster_parts(vertices, grid_x, grid_y).toarray()
    result_grid = numpy.real(parts_grid) + numpy.imag(parts_grid).cumsum(axis=0)
    return result_grid


def find_intersections(
        vertices: numpy.ndarray,
        grid_x: numpy.ndarray,
        grid_y: numpy.ndarray
        ) -> Tuple[numpy.ndarray]:
    """
    Find intersections between a polygon and grid lines
    """
    if vertices.shape[0] != 2:
        raise Exception('vertices must be 2xN')

    min_bounds = floor(vertices.min(axis=1))
    max_bounds = ceil(vertices.max(axis=1))

    keep_x = logical_and(grid_x >= min_bounds[0],
                         grid_x <= max_bounds[0])
    keep_y = logical_and(grid_y >= min_bounds[1],
                         grid_y <= max_bounds[1])

    if not (keep_x.any() or keep_y.any()):  # polygon doesn't overlap grid
        mat_shape = (vertices.shape[1], grid_x.size + grid_y.size)
        return zeros(mat_shape), zeros(mat_shape), zeros(mat_shape, dtype=bool)

    y_seg_xs = hstack((min_bounds[0], grid_x[keep_x], max_bounds[0])).T
    x_seg_ys = hstack((min_bounds[1], grid_y[keep_y], max_bounds[1])).T

    '''
    Calculate intersections between polygon and grid line segments
    '''
    xy1b = numpy.roll(vertices, -1, axis=1)

    # Lists of initial/final coordinates for polygon segments
    xi1 = vertices[0, :, newaxis]
    yi1 = vertices[1, :, newaxis]
    xf1 = xy1b[0, :, newaxis]
    yf1 = xy1b[1, :, newaxis]

    # Lists of initial/final coordinates for grid segments
    xi2 = hstack((full_like(x_seg_ys, min_bounds[0]), y_seg_xs))
    xf2 = hstack((full_like(x_seg_ys, max_bounds[0]), y_seg_xs))
    yi2 = hstack((x_seg_ys, full_like(y_seg_xs, min_bounds[0])))
    yf2 = hstack((x_seg_ys, full_like(y_seg_xs, max_bounds[1])))

    # Perform calculation
    dxi = xi1 - xi2
    dyi = yi1 - yi2
    dx1 = xf1 - xi1
    dx2 = xf2 - xi2
    dy1 = yf1 - yi1
    dy2 = yf2 - yi2

    numerator_a = dx2 * dyi - dy2 * dxi
    numerator_b = dx1 * dyi - dy1 * dxi
    denominator = dy2 * dx1 - dx2 * dy1

    # Avoid warnings since we may multiply eg. NaN*False
    with numpy.errstate(invalid='ignore', divide='ignore'):
        u_a = numerator_a / denominator
        u_b = numerator_b / denominator

        # Find the adjacency matrix
        int_x = xi1 + dx1 * u_a
        int_y = yi1 + dy1 * u_a
        adjacency = logical_and.reduce((u_a >= 0, u_a <= 1, u_b >= 0, u_b <= 1))

        # Arrange output.
        # adjacency[i, j] tells us if polygon segment i intersects with grid line j
        # xy[i, j] tells us the x,y coordinates of the intersection in the form x+iy
        # normalized_distance[i, j] tells us the fraction of the segment i
        #   we have to traverse in order to reach the intersection
        xy = (int_x + 1j * int_y) * adjacency
        normalized_distance = u_a

    # print('sparsity', adjacency.astype(int).sum() / adjacency.size)
    return normalized_distance, xy, adjacency


def create_vertices(
        vertices: numpy.ndarray,
        grid_x: numpy.ndarray,
        grid_y: numpy.ndarray,
        new_vertex_data: Tuple[numpy.ndarray] = None
        ) -> sparse.coo_matrix:
    """
    Create additional vertices where a polygon crosses gridlines
    """
    if vertices.shape[0] != 2:
        raise Exception('vertices must be 2xN')
    if grid_x.size < 1 or grid_y.size < 1:
        raise Exception('Grid must contain at least one line in each direction?')

    num_poly_vertices = vertices.shape[1]

    if new_vertex_data is None:
        new_vertex_data = find_intersections(vertices, grid_x, grid_y)
    normalized_distance, xy_matrix, adjacency_matrix = new_vertex_data

    '''
    Insert any polygon-grid intersections as new polygon vertices
    '''
    # Figure out how to sort each row of the intersection matrices
    #  based on distance from (xi1, yi1) (the polygon segment's first point)
    # This lets us insert them as new vertices in the proper order
    sortix = normalized_distance.argsort(axis=1)
    sortix_paired = (numpy.arange(num_poly_vertices)[:, newaxis], sortix)
    assert(normalized_distance.shape[0] == num_poly_vertices)

    # if any new points fall outside the window, shrink them back onto it
    xy_shrunken = (numpy.real(xy_matrix).clip(grid_x[0], grid_x[-1]) + 1j *
                   numpy.imag(xy_matrix).clip(grid_y[0], grid_y[-1]))

    # Use sortix to sort adjacency matrix and the intersection (x, y) coordinates,
    #  and hstack the original points to the left of the new ones
    xy_with_original = hstack((vertices[0, :, newaxis] + 1j * vertices[1, :, newaxis],
                               xy_shrunken[sortix_paired]))
    has_intersection = hstack((ones((vertices.shape[1], 1), dtype=bool),
                               adjacency_matrix[sortix_paired]))

    # Now remove all extra entries which don't correspond to new vertices
    #  (ie, no intersection happened), and then flatten, creating our
    #  polygon-with-extra-vertices, though some redundant vertices are included,
    #  which we must later remove manually.
    vertices = xy_with_original[has_intersection]

    return vertices

def clip_vertices_to_window(
        vertices: numpy.ndarray,
        min_x: float = -numpy.inf,
        max_x: float = numpy.inf,
        min_y: float = -numpy.inf,
        max_y: float = numpy.inf
        ) -> numpy.ndarray:
    """
    """
    # Remove points outside the window (these will only be original points)
    #  Since the boundaries of the window are also pixel boundaries, this just
    #  makes the polygon boundary proceed along the window edge
    inside = logical_and.reduce((numpy.real(vertices) <= max_x,
                                 numpy.real(vertices) >= min_x,
                                 numpy.imag(vertices) <= max_y,
                                 numpy.imag(vertices) >= min_y))
    vertices = vertices[inside]

    # Remove consecutive duplicate vertices
    consecutive = numpy.ediff1d(vertices, to_begin=[1 + 1j]).astype(bool)
    vertices = vertices[consecutive]
    return vertices


def get_raster_parts(
        vertices: numpy.ndarray,
        grid_x: numpy.ndarray,
        grid_y: numpy.ndarray
        ) -> sparse.coo_matrix:
    """
    This function performs the same task as raster(...), but instead of returning a dense array
    of pixel values, it returns a sparse array containing the value
        (-area + 1j * cover)
    for each pixel which contains a line segment, where
        cover   is the fraction of the pixel's y-length that is traversed by the segment,
                 multiplied by the sign of (y_final - y_initial)
        area    is the fraction of the pixel's area covered by the trapezoid formed by
                 the line segment's endpoints (clipped to the cell edges) and their projections
                 onto the pixel's left (i.e., lowest-x) edge, again multiplied by
                 the sign of (y_final - y_initial)
        Note that polygons are assumed to be wound clockwise.

    The result from raster(...) can be obtained with
        raster_result = numpy.real(lines_result) + numpy.imag(lines_result).cumsum(axis=0)

    :param vertices: 2xN ndarray containing x,y coordinates for each point in the polygon
    :param grid_x: x-coordinates for the edges of each pixel (ie, the leftmost two columns span
        x=grid_x[0] to x=grid_x[1] and x=grid_x[1] to x=grid_x[2])
    :param grid_y: y-coordinates for the edges of each pixel (see grid_x)
    :return: Complex sparse COO matrix containing area and cover information
    """
    if grid_x.size < 2 or grid_y.size < 2:
        raise Exception('Grid must contain at least one full pixel')

    num_xy_px = numpy.array([grid_x.size, grid_y.size]) - 1

    vertices = clip_vertices_to_window(vertices,
                                       grid_x[0], grid_x[-1],
                                       grid_y[0], grid_y[-1])

    # If the shape fell completely outside our area, just return a blank grid
    if vertices.size == 0:
        return sparse.coo_matrix(shape=num_xy_px)

    '''
    Calculate area, cover
    '''
    # Calculate segment cover, area, and corresponding pixel's subscripts
    poly = hstack((vertices, vertices[0]))
    endpoint_avg = (poly[:-1] + poly[1:]) * 0.5

    # Remove segments along the right,top edges
    #  (they correspond to outside pixels, but couldn't be removed until now
    #  because 'vertices' stored points, not segments, and the edge points are needed
    #  when creating endpoint_avg)
    non_edge = numpy.logical_and(numpy.real(endpoint_avg) < grid_x[-1],
                                 numpy.imag(endpoint_avg) < grid_y[-1])

    endpoint_avg_final = endpoint_avg[non_edge]
    x_sub = numpy.digitize(numpy.real(endpoint_avg_final), grid_x) - 1
    y_sub = numpy.digitize(numpy.imag(endpoint_avg_final), grid_y) - 1

    cover = diff(numpy.imag(poly), axis=0)[non_edge] / diff(grid_y)[y_sub]
    area = (numpy.real(endpoint_avg_final) - grid_x[x_sub]) * cover / diff(grid_x)[x_sub]

    # Use coo_matrix(...).toarray() to efficiently convert from (x, y, v) pairs to ndarrays.
    #  We can use v = (-area + 1j * cover) followed with calls to numpy.real() and numpy.imag() to
    #  improve performance (Otherwise we'd have to call coo_matrix() twice. It's really inefficient
    #  because it involves lots of random memory access, unlike real() and imag()).
    poly_grid = sparse.coo_matrix((-area + 1j * cover, (x_sub, y_sub)), shape=num_xy_px)
    return poly_grid

