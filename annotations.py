"""Turning QuPath point annotations into tiles.

The annotations are drawn as loose MultiPoint sets, so before anything can be
done with them the points have to be put into an order that traces the outline
they were meant to describe, and the outline has to be filled in on the tile
grid.

Both 01_write_tiles_to_qupath.py (which writes the tiles back into the QuPath project)
and 03_extract_ground_truth.py (which scores the algorithm against them) need
exactly this, so it lives here rather than in either one.
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import binary_fill_holes

from config import TILE_SIZE


def order_points_annotation(points_xy):
    """Order a loose set of annotation points into a single traced path.

    Greedy nearest-neighbour: start at the topmost point, then repeatedly
    step to the closest point not yet visited.
    """
    pts = np.array(points_xy, dtype=float)        # converts the points to np.array
    n = len(pts)
    tree = KDTree(pts)
    visited = np.zeros(n, dtype=bool)             # form an array of false values
    start = int(np.argmin(pts[:, 1]))             # find the topmost point (start point)
    order = [start]
    visited[start] = True
    for _ in range(n - 1):                        # loop through the points, sorted by distance
        last = order[-1]
        _, neighbours = tree.query(pts[last], k=n)
        for nb in neighbours:                     # choose the point that is not yet visited
            if not visited[nb]:
                order.append(int(nb))
                visited[nb] = True
                break
    return pts[order]


def two_opt_cleanup(tour, max_passes=30, verbose_name=None):
    """Shorten a traced path by un-crossing it.

    Plain nearest-neighbour tours can leave a cluster of points unvisited
    until the very end, forcing a long jump to reach them and another long
    jump to get back.

    2-opt fixes this without changing the tracing approach: for any two
    edges (i, i+1) and (j, j+1) in the tour, if reversing the segment
    between them produces a shorter total path, do it.
    """
    tour = tour.copy()
    n = len(tour)

    def edge_len(a, b):
        return np.hypot(tour[a, 0] - tour[b, 0], tour[a, 1] - tour[b, 1])

    for p in range(max_passes):
        improved = False
        for i in range(n - 1):
            a, b = i, i + 1
            d_ab = edge_len(a, b)
            # search over all valid j for this i
            js = np.arange(i + 2, n)
            js = js[js != n - 1] if i == 0 else js   # don't touch the closing edge on the first edge
            if len(js) == 0:
                continue
            c = js
            d = (js + 1) % n
            d_cd = np.hypot(tour[c, 0] - tour[d, 0], tour[c, 1] - tour[d, 1])
            d_ac = np.hypot(tour[a, 0] - tour[c, 0], tour[a, 1] - tour[c, 1])
            d_bd = np.hypot(tour[b, 0] - tour[d, 0], tour[b, 1] - tour[d, 1])
            gains = (d_ab + d_cd) - (d_ac + d_bd)
            best_idx = np.argmax(gains)
            if gains[best_idx] > 1e-6:
                j = int(js[best_idx])
                tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                improved = True
        if verbose_name:
            print(f"  [{verbose_name}] 2-opt pass {p+1}: "
                  f"{'improved' if improved else 'no change, stopping'}")
        if not improved:
            break
    return tour


def get_tiles_inside_boundary(ordered_points, tile_size=TILE_SIZE):
    """Every tile enclosed by the traced outline, as level-0 pixel origins.

    The outline is rasterised onto a small local grid, filled, and the filled
    cells are mapped back to slide coordinates.
    """
    pts = (np.array(ordered_points) / tile_size).astype(int)
    col_min, row_min = pts.min(axis=0)
    col_max, row_max = pts.max(axis=0)
    # binary_fill_holes only fills enclosed regions, so a border is added to
    # guarantee the traced outline is closed on every side.
    border = 1
    # distance between min and max, plus 1 to include the last tile, plus the
    # border on both sides
    H = row_max - row_min + 2 * border + 1
    W = col_max - col_min + 2 * border + 1
    grid = np.zeros((H, W), dtype=bool)
    n = len(pts)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]                    # %n joins the last point back to the first
        num_steps = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1])) + 1
        cols = np.round(np.linspace(p1[0], p2[0], num_steps)).astype(int) - col_min + border
        rows = np.round(np.linspace(p1[1], p2[1], num_steps)).astype(int) - row_min + border
        for c, r in zip(cols, rows):
            if 0 <= r < H and 0 <= c < W:
                grid[r, c] = True
    filled = binary_fill_holes(grid)
    row_idx, col_idx = np.where(filled)
    return [
        {"tile_x": int((col_idx[i] - border + col_min) * tile_size),
         "tile_y": int((row_idx[i] - border + row_min) * tile_size)}
        for i in range(len(row_idx))
    ]
