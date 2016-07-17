"""
Module for rasterizing polygons, with float-precision anti-aliasing on
 a non-uniform rectangular grid.

See the documentation for raster(...) for details.
"""

import numpy
from numpy import logical_and, diff, floor, ceil, ones, zeros, hstack, full_like, newaxis
from scipy import sparse

__author__ = 'Jan Petykiewicz'


def raster(poly_xy: numpy.ndarray,
           grid_x: numpy.ndarray,
           grid_y: numpy.ndarray
           ) -> numpy.ndarray:
    """
    Draws a polygon onto a 2D grid of pixels, setting pixel values equal to the fraction of the
     pixel area covered by the polygon. This implementation is written for accuracy and works with
     double precision, in contrast to most other implementations which are written for speed and
     usually only allow for 256 (and often fewer) possible pixel values without performing (very
     slow) super-sampling.

    :param poly_xy: 2xN ndarray containing x,y coordinates for each point in the polygon
    :param grid_x: x-coordinates for the edges of each pixel (ie, the leftmost two columns span
        x=grid_x[0] to x=grid_x[1] and x=grid_x[1] to x=grid_x[2])
    :param grid_y: y-coordinates for the edges of each pixel (see grid_x)
    :return: 2D ndarray with pixel values in the range [0, 1] containing the anti-aliased polygon
    """
    poly_xy = numpy.array(poly_xy)
    grid_x = numpy.array(grid_x)
    grid_y = numpy.array(grid_y)

    if poly_xy.shape[0] != 2:
        raise Exception('poly_xy must be 2xN')
    if grid_x.size < 1 or grid_y.size < 1:
        raise Exception('Grid must contain at least one full pixel')

    num_xy_px = numpy.array([grid_x.size, grid_y.size]) - 1

    min_bounds = floor(poly_xy.min(axis=1))
    max_bounds = ceil(poly_xy.max(axis=1))

    keep_x = logical_and(grid_x >= min_bounds[0],
                         grid_x <= max_bounds[0])
    keep_y = logical_and(grid_y >= min_bounds[1],
                         grid_y <= max_bounds[1])

    if not (keep_x.any() and keep_y.any()):  # polygon doesn't overlap grid
        return zeros(num_xy_px)

    y_seg_xs = hstack((min_bounds[0], grid_x[keep_x], max_bounds[0])).T
    x_seg_ys = hstack((min_bounds[1], grid_y[keep_y], max_bounds[1])).T

    num_poly_vertices = poly_xy.shape[1]

    '''
    Calculate intersections between polygon and grid line segments
    '''
    xy1b = numpy.roll(poly_xy, -1, axis=1)

    # Lists of initial/final coordinates for polygon segments
    xi1 = poly_xy[0, :, newaxis]
    yi1 = poly_xy[1, :, newaxis]
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

        # Find the adjacency matrix A of intersecting lines.
        int_x = xi1 + dx1 * u_a
        int_y = yi1 + dy1 * u_a
        int_b = logical_and.reduce((u_a >= 0, u_a <= 1, u_b >= 0, u_b <= 1))

        # Arrange output.
        # int_adjacency_matrix[i, j] tells us if polygon segment i intersects with grid line j
        # int_xy_matrix[i, j] tells us the x,y coordinates of the intersection in the form x+iy
        # int_normalized_distance_1to2[i, j] tells us the fraction of the segment i
        #   we have to traverse in order to reach the intersection
        int_adjacency_matrix = int_b
        int_xy_matrix = (int_x + 1j * int_y) * int_b
        int_normalized_distance_1to2 = u_a

    # print('sparsity', int_adjacency_matrix.astype(int).sum() / int_adjacency_matrix.size)

    '''
    Insert any polygon-grid intersections as new polygon vertices
    '''
    # Figure out how to sort each row of the intersection matrices
    #  based on distance from (xi1, yi1) (the polygon segment's first point)
    # This lets us insert them as new vertices in the proper order
    sortix = int_normalized_distance_1to2.argsort(axis=1)
    sortix_paired = (numpy.arange(num_poly_vertices)[:, newaxis], sortix)
    assert(int_normalized_distance_1to2.shape[0] == num_poly_vertices)

    # If any new points fall outside the window, shrink them back onto it
    xy_shrunken = (numpy.real(int_xy_matrix).clip(grid_x[0], grid_x[-1]) + 1j *
                   numpy.imag(int_xy_matrix).clip(grid_y[0], grid_y[-1]))

    # Use sortix to sort adjacency matrix and the intersection (x, y) coordinates,
    #  and hstack the original points to the left of the new ones
    xy_with_original = hstack((poly_xy[0, :, newaxis] + 1j * poly_xy[1, :, newaxis],
                               xy_shrunken[sortix_paired]))
    has_intersection = hstack((ones((poly_xy.shape[1], 1), dtype=bool),
                               int_adjacency_matrix[sortix_paired]))

    # Now remove all extra entries which don't correspond to new vertices
    #  (ie, no intersection happened), and then flatten, creating our
    #  polygon-with-extra-vertices, though some extra vertices are included,
    #  which we must remove manually.
    vertices = xy_with_original[has_intersection]

    # Remove points outside the window (these will only be original points)
    #  Since the boundaries of the window are also pixel boundaries, this just
    #  makes the polygon boundary proceed along the window edge
    inside = logical_and.reduce((numpy.real(vertices) <= grid_x[-1],
                                 numpy.real(vertices) >= grid_x[0],
                                 numpy.imag(vertices) <= grid_y[-1],
                                 numpy.imag(vertices) >= grid_y[0]))
    vertices = vertices[inside]

    # Remove consecutive duplicate vertices
    consecutive = numpy.ediff1d(vertices, to_begin=[1 + 1j]).astype(bool)
    vertices = vertices[consecutive]

    # If the shape fell completely outside our area, just return a blank grid
    if vertices.size == 0:
        return zeros(num_xy_px)

    '''
    Calculate area, cover
    '''
    # Calculate segment cover, area, and corresponding pixel's subscripts
    poly = hstack((vertices, vertices[0]))
    endpoint_avg = (poly[:-1] + poly[1:]) * 0.5

    # Remove segments along the right,top edges
    #  (they correspond to outside pixels, but couldn't be removed until now
    #  because poly_xy stores points, not segments, and the edge points are needed
    #  when creating endpoint_avg)
    non_edge = numpy.logical_and(numpy.real(endpoint_avg) < grid_x[-1],
                                 numpy.imag(endpoint_avg) < grid_y[-1])

    endpoint_final = endpoint_avg[non_edge]
    x_sub = numpy.digitize(numpy.real(endpoint_final), grid_x) - 1
    y_sub = numpy.digitize(numpy.imag(endpoint_final), grid_y) - 1

    cover = diff(numpy.imag(poly), axis=0)[non_edge] / diff(grid_y)[y_sub]
    area = (numpy.real(endpoint_final) - grid_x[x_sub]) * cover / diff(grid_x)[x_sub]

    # Use coo_matrix(...).toarray() to efficiently convert from (x, y, v) pairs to ndarrays.
    #  We can use v = (-area + 1j * cover) followed with calls to numpy.real() and numpy.imag() to
    #  improve performance (Otherwise we'd have to call coo_matrix() twice. It's really inefficient
    #  because it involves lots of random memory access, unlike real() and imag()).
    poly_grid = sparse.coo_matrix((-area + 1j * cover, (x_sub, y_sub)), shape=num_xy_px).toarray()
    result_grid = numpy.real(poly_grid) + numpy.imag(poly_grid).cumsum(axis=0)

    return result_grid
