import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


import sys
import matplotlib.tri
import time


def circumcircle2(T):
    P1, P2, P3 = T[:, 0], T[:, 1], T[:, 2]
    b = P2 - P1
    c = P3 - P1
    d = 2 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    center_x = (c[:, 1] * (np.square(b[:, 0]) + np.square(b[:, 1])) - b[:, 1] * (
                np.square(c[:, 0]) + np.square(c[:, 1]))) / d + P1[:, 0]
    center_y = (b[:, 0] * (np.square(c[:, 0]) + np.square(c[:, 1])) - c[:, 0] * (
                np.square(b[:, 0]) + np.square(b[:, 1]))) / d + P1[:, 1]
    return np.array((center_x, center_y)).T


def check_outside(point, bbox):
    point = np.round(point, 4)
    return point[0] < bbox[0] or point[0] > bbox[2] or point[1] < bbox[1] or point[1] > bbox[3]


def move_point(start, end, bbox):
    vector = end - start
    c = calc_shift(start, vector, bbox)
    print(start, vector, bbox, c)
    if c > 0 and c < 1:
        start = start + c * vector
        return start


def calc_shift(point, vector, bbox):
    c = sys.float_info.max
    for l, m in enumerate(bbox):
        a = (float(m) - point[l % 2]) / vector[l % 2]
        if a > 0 and not check_outside(point + a * vector, bbox):
            if abs(a) < abs(c):
                c = a
    return c if c < sys.float_info.max else None


def voronoi2(P, bbox=None):
    if not isinstance(P, np.ndarray):
        P = np.array(P)
    if not bbox:
        xmin = P[:, 0].min()
        xmax = P[:, 0].max()
        ymin = P[:, 1].min()
        ymax = P[:, 1].max()
        xrange = (xmax - xmin) * 0.3333333
        yrange = (ymax - ymin) * 0.3333333
        bbox = (xmin - xrange, ymin - yrange, xmax + xrange, ymax + yrange)
    bbox = np.round(bbox, 4)

    D = matplotlib.tri.Triangulation(P[:, 0], P[:, 1])
    T = D.triangles
    n = T.shape[0]
    C = circumcircle2(P[T])

    segments = []
    for i in range(n):
        for j in range(3):
            k = D.neighbors[i][j]
            if k != -1:
                # cut segment to part in bbox
                start, end = C[i], C[k]
                if check_outside(start, bbox):
                    start = move_point(start, end, bbox)
                    if start is None:
                        continue
                if check_outside(end, bbox):
                    end = move_point(end, start, bbox)
                    if end is None:
                        continue
                segments.append([start, end])
            else:
                # ignore center outside of bbox
                if check_outside(C[i], bbox):
                    continue
                first, second, third = P[T[i, j]], P[T[i, (j + 1) % 3]], P[T[i, (j + 2) % 3]]
                edge = np.array([first, second])
                vector = np.array([[0, 1], [-1, 0]]).dot(edge[1] - edge[0])
                line = lambda p: (p[0] - first[0]) * (second[1] - first[1]) / (second[0] - first[0]) - p[1] + first[1]
                orientation = np.sign(line(third)) * np.sign(line(first + vector))
                if orientation > 0:
                    vector = -orientation * vector
                c = calc_shift(C[i], vector, bbox)
                if c is not None:
                    segments.append([C[i], C[i] + c * vector])
    return segments


seed = 777
max_iter = 1000
prec = 1E-2

np.random.seed(seed)
nCells = 10

#np.random.rand(nbCells, 2)
y1 = [0.68785741, 0.49219261, 0.34511557, 0.99504991, 0.69526717, 0.01070004, 0.34501585, 0.17204948, 0.94936067, 0.24919264]
y2 = [0.7327903, 0.6602892, 0.5803169, 0.5947815, 0.8662715, 0.1039026, 0.4183077, 0.8675228, 0.3523569, 0.3898254]
points = np.asmatrix([[i, j] for i, j in zip(y1, y2)])
vtilde = np.repeat(0, nCells)
q = np.repeat(1/nCells, nCells)
demand = np.repeat(0, nCells)

vor = Voronoi(points)
#lines=voronoi2(points, (0, 0, 1, 1))
#plt.scatter(points[:,0], points[:,1], color="blue")
#lines = matplotlib.collections.LineCollection(lines, color='red')
#plt.gca().add_collection(lines)
#plt.show()
#fig = voronoi_plot_2d(vor)
#plt.show()

regions, vertices = voronoi_finite_polygons_2d(vor)

min_x = 0
max_x = 1
min_y = 0
max_y = 1

mins = np.tile((min_x, min_y), (vertices.shape[0], 1))
bounded_vertices = np.max((vertices, mins), axis=0)
maxs = np.tile((max_x, max_y), (vertices.shape[0], 1))
bounded_vertices = np.min((bounded_vertices, maxs), axis=0)

box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

# colorize
for region in regions:
    polygon = vertices[region]
    # Clipping polygon
    poly = Polygon(polygon)
    poly = poly.intersection(box)
    polygon = [p for p in poly.exterior.coords]

    plt.fill(*zip(*polygon), alpha=0.4)

plt.plot(points[:, 0], points[:, 1], 'ko')
plt.axis('equal')
plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

plt.savefig('voro.png')
plt.show()
#if __name__=='__main__':
#    points=np.random.rand(100,2)*100
#    lines=voronoi2(points, (0,0, 1, 1))
#    plt.scatter(points[:,0], points[:,1], color="blue")
#    lines = matplotlib.collections.LineCollection(lines, color='red')
#    plt.gca().add_collection(lines)
#    plt.axis((-20,120, -20,120))
#    plt.show()