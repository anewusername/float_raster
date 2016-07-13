"""
Module for rasterizing polygons, with float-precision anti-aliasing on
 a non-uniform rectangular grid.

See the documentation for raster(...) for details.
"""

import numpy
from numpy import r_, c_, logical_and, diff, floor, ceil, ones, zeros, vstack, hstack,\
    full_like, newaxis
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

    # ## Calculate intersections
    xy1b = numpy.roll(poly_xy, -1, axis=1)

    xi1 = poly_xy[0, :, newaxis]
    yi1 = poly_xy[1, :, newaxis]
    xf1 = xy1b[0, :, newaxis]
    yf1 = xy1b[1, :, newaxis]

    xi2 = hstack((full_like(x_seg_ys, min_bounds[0]), y_seg_xs))
    xf2 = hstack((full_like(x_seg_ys, max_bounds[0]), y_seg_xs))
    yi2 = hstack((x_seg_ys, full_like(y_seg_xs, min_bounds[0])))
    yf2 = hstack((x_seg_ys, full_like(y_seg_xs, max_bounds[1])))

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
        int_adjacency_matrix = int_b
        int_matrix_x = int_x * int_b
        int_matrix_y = int_y * int_b
        int_normalized_distance_1to2 = u_a

    # ## Insert intersection points as vertices
    # If new points fall outside the window, shrink them back onto it
    int_matrix_x = int_matrix_x.clip(grid_x[0], grid_x[-1])
    int_matrix_y = int_matrix_y.clip(grid_y[0], grid_y[-1])

    # sort intersections based on distance from first vertex, to add in order
    sortix = int_normalized_distance_1to2.argsort(axis=1)
    sortix_paired = (numpy.arange(num_poly_vertices)[:, newaxis], sortix)
    assert(int_normalized_distance_1to2.shape[0] == num_poly_vertices)

    # Use sortix to sort adjacency matrix and the intersection (x, y) coordinates,
    #  and vstack the original points on top of the top row
    xs = vstack((poly_xy[0, :], int_matrix_x[sortix_paired].T))
    ys = vstack((poly_xy[1, :], int_matrix_y[sortix_paired].T))
    has_intersection = r_[ones((1, poly_xy.shape[1]), dtype=bool),
                          int_adjacency_matrix[sortix_paired].T]

    # Now use has_intersection to index the intersection coordinates, thus creating a 2-column
    #  array which holds the [[x, y], ...] for the polygon with added vertices at pixel-boundary
    #  intersections
    poly_xy_xy = c_[xs.T[has_intersection.T], ys.T[has_intersection.T]]

    # Remove points outside the window (these will only be original points)
    #  Since the boundaries of the window are also pixel boundaries, this just
    #  makes the polygon boundary proceed along the window edge
    inside_window = logical_and.reduce((poly_xy_xy[:, 1] <= grid_y[-1],
                                        poly_xy_xy[:, 1] >= grid_y[0],
                                        poly_xy_xy[:, 0] <= grid_x[-1],
                                        poly_xy_xy[:, 0] >= grid_x[0]))
    poly_xy_xy = poly_xy_xy[inside_window, :]

    # Remove consecutive duplicate entries
    consecutive = diff(poly_xy_xy, axis=0).any(axis=1)  # use any() as !=0
    poly_xy_xy = poly_xy_xy[r_[True, consecutive], :]

    # If the shape fell completely outside our area, just return a blank grid
    if poly_xy_xy.size == 0:
        # for matlab:
        # rg = array.array('d', numpy.nditer(zeros(num_xy_px), order='F'))
        # return rg
        return zeros(num_xy_px)

    # ## Calculate area, cover
    # Calculate segment cover, area, and corresponding pixel's subscripts
    poly = vstack((poly_xy_xy,
                   poly_xy_xy[0, :]))
    endpoint_avg = (poly[:-1, :] + poly[1:, :]) / 2

    # Remove segments along the right,top edges
    #  (they correspond to outside pixels, but couldn't be removed until now
    #  because poly_xy stores points, not segments, and the edge points are needed
    #  when creating endpoint_avg)
    non_edge = numpy.logical_and(endpoint_avg[:, 0] < grid_x[-1],
                                 endpoint_avg[:, 1] < grid_y[-1])

    x_sub = numpy.digitize(endpoint_avg[non_edge, 0], grid_x) - 1
    y_sub = numpy.digitize(endpoint_avg[non_edge, 1], grid_y) - 1

    cover = diff(poly[:, 1], axis=0)[non_edge] / diff(grid_y)[y_sub]
    area = (endpoint_avg[non_edge, 0] - grid_x[x_sub]) * cover / diff(grid_x)[x_sub]

    poly_grid = sparse.coo_matrix((-area, (x_sub, y_sub)), shape=num_xy_px).toarray()
    cover_grid = sparse.coo_matrix((cover, (x_sub, y_sub)), shape=num_xy_px).toarray()
    poly_grid = poly_grid + cover_grid.cumsum(axis=0)

    return poly_grid
