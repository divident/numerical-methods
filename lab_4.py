import math

import matplotlib.pyplot as plt
import numpy as np

# points = [
#     (1, 0.6),
#     (2, 1),
#     (2.5, 2),
#     (2, 3),
#     (0, 3),
#     (-2, 2.5),
#     (-2.5, 1.5),
#     (-3, -1),
#     (-2, -2),
#     (0, -2.5)
# ]
#
points = [
    (1, 1),
    (2, 3),
    (3, 2),
    (4, 4),
    (5, 1)
]


def transform_polar(points):
    polar_points = []
    for x, y in points:
        r = math.sqrt(x ** 2 + y ** 2)
        if x > 0 and y >= 0:
            thetha = math.atan(y / x)
        elif x > 0 and y < 0:
            thetha = math.atan(y / x) + 2 * math.pi
        elif x < 0:
            thetha = math.atan(y / x) + math.pi
        elif x == 0 and y > 0:
            thetha = math.pi / 2
        else:
            thetha = (3 * math.pi) / 2

        polar_points.append((thetha, r))
    return polar_points


def a(i):
    xi, yi = points[i]
    xj, yj = points[i + 1]
    return (yj - yi) / (xj - xi)


def spline_first():
    tmp_points = []
    step = 0.1
    n = len(points)
    for i in range(n - 1):
        xj, _ = points[i + 1]
        xi, yi = points[i]
        beg, end = min(xi, xj), max(xi, xj)
        for k in np.arange(beg, end, step):
            tmp_points.append((k, a(i) * (k - xi) + yi))
    return tmp_points


def h(i):
    return points[i + 1][0] - points[i][0]


def h_matrix():
    h_row = []
    h_matrix = []
    n = len(points)
    for i in range(1, n - 1):
        h_row.append(h(i - 1))
        h_row.append(2 * (h(i - 1) + h(i)))
        h_row.append(h(i))
        h_matrix.append(h_row)
        h_row = []
    return h_matrix


def x_matrix():
    n = len(points)
    x_matrix = []
    for i in range(1, n - 1):
        yi, yj, yk = points[i - 1][1], points[i][1], points[i + 1][1]
        x_matrix.append([6.0 / h(i) * (yk - yj) - 6.0 / h(i - 1) * (yj - yi)])
    return x_matrix


def spline_third():
    n = len(points)
    first_row = [1] + [0 for i in range(n - 1)]
    last_row = first_row[::-1]
    right = 0
    left = n - 3
    matrix = h_matrix()
    z_matrix = []
    z_matrix.append(first_row)
    for i in range(n - 2):
        row = [0 for i in range(right)] + matrix[i] + [0 for i in range(left)]
        z_matrix.append(row)
        right += 1
        left -= 1
    z_matrix.append(last_row)
    X = [[0, ]] + x_matrix() + [[0, ]]
    Z, X = np.array(z_matrix), np.array(X)
    Y = np.linalg.solve(Z, X)
    print(Z)
    print(X)
    print(Y)
    return Y


def calculate_points(z):
    ret_points = []
    step = 0.1
    n = len(points)
    for i in range(n - 1):
        xi, yi = points[i]
        xj, yj = points[i + 1]
        val1 = z[i][0] / (6.0 * h(i))
        val2 = z[i + 1][0] / (6.0 * h(i))
        val3 = (yj / h(i) - (z[i + 1][0] * h(i)) / 6.0)
        val4 = (yi / h(i) - (z[i][0] * h(i)) / 6.0)
        fun = lambda x: val1 * (xj - x) ** 3 + val2 * (x - xi) ** 3 + val3 * (x - xi) + val4 * (xj - x)
        # beg, end = min(xi, xj), max(xi, xj)
        for k in np.arange(xi, xj + 2 * step, step):
            ret_points.append((k, fun(k)))

    return ret_points


def f(x):
    return math.e ** (-x / 2) * math.sin(4 * x)


def generate_points():
    ret_points = []
    step = 1
    for k in np.arange(-1, 7 + 2 * step, step):
        ret_points.append((k, f(k)))
    return ret_points


def main():
    global points
    points = generate_points()
    xpoints = [x[0] for x in points]
    ypoints = [y[1] for y in points]

    # spline_points = spline_first()
    # xspline = [x for x, _ in spline_points]
    # yspline = [y for _, y in spline_points]
    #
    # plt.plot(xspline, yspline, 'bo')
    # plt.plot(xpoints, ypoints, 'ro')
    # plt.show()
    Y = spline_third()
    spline_points = calculate_points(Y)
    xspline = [x for x, _ in spline_points]
    yspline = [y for _, y in spline_points]

    plt.plot(xspline, yspline, 'bo')
    plt.plot(xpoints, ypoints, 'ro')
    plt.show()


if __name__ == "__main__":
    main()
