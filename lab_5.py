import math
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return math.e ** (-x / 2) * math.sin(4 * x)

h = 0.5


def generate_points():
    ret_points = []
    step = 0.5
    for k in np.arange(-1, 7 + 2 * step, step):
        ret_points.append((k, f(k)))
    return ret_points


def B(x0, h, x):
    if x <= x0 - 2 * h:
        return 0
    elif x0 - 2 * h <= x <= x0 - h:
        return 1 / 6 * (2 * h + (x - x0)) ** 3
    elif x0 - h <= x <= x0:
        return 2 / 3 * h ** 3 - 1 / 2 * (x - x0) ** 2 * (2 * h + (x - x0))
    elif x0 <= x < x0 + h:
        return 2 / 3 * h ** 3 - 1 / 2 * (x - x0) ** 2 * (2 * h - (x - x0))
    elif x0 + h <= x <= x0 + 2 * h:
        return 1 / 6 * (2 * h - (x - x0)) ** 3
    else:
        return 0


def b_matrix(n):
    matrix = []
    first_row = [1] + [0 for _ in range(n - 1)]
    matrix.append(first_row)
    last_row = first_row[::-1]
    for i in range(n - 2):
        row = [0 for _ in range(i)] + [1, 4, 1] + [0 for _ in range(n - 3 - i)]
        matrix.append(row)
    matrix.append(last_row)
    return matrix


def a_matrix(n, b_matrix, x_points):
    b_matrix = np.array([np.array(xi) for xi in b_matrix])
    x_matrix = []
    x_matrix.append([1/h**3 * f(x_points[0]), ])
    for i in range(1, n - 1):
        x_matrix.append([6/h**3 * f(x_points[i]), ])
    x_matrix.append([1/h**3 * f(x_points[n - 1]), ])
    x_matrix = np.array(x_matrix)
    a = np.linalg.solve(b_matrix, x_matrix)

    #print(a)
    a = np.vstack((a, [2 * a[n - 1] - a[n - 2]]))
    a = np.vstack(([2 * a[0] - a[1]], a))
    #print(a)
    # a = np.concatenate([2 * a[0] - a[1], ], a, [2 * a[n-2] - a[n-3], ])
    return a


def b_spline(a_matrix, x_points):
    step = 0.1
    ret_points = []
    n = len(x_points)

    for i in range(n - 1):
        xi, xj = x_points[i], x_points[i + 1]
        for k in np.arange(xi, xj+2*step, step):
            a1 = a_matrix[i]
            a2 = a_matrix[i+1]
            a3 = a_matrix[i+2]
            a4 = a_matrix[i+3]
            y = a1 * B(x_points[i] - h, h, k) + a2 * B(x_points[i], h, k) + \
                a3 * B(x_points[i] + h, h, k) + a4 * B(x_points[i] + 2*h, h, k)
            ret_points.append((k, y))
    return ret_points


def main():
    points = generate_points()
    #points = [(1,1), (2,3), (3,2)]
    xpoints = [x[0] for x in points]
    ypoints = [y[1] for y in points]
    n = len(points)
    b = b_matrix(n)
    a = a_matrix(n, b, xpoints)
    print(a)
    print(b)
    spline_points = b_spline(a, xpoints)
    xspline = [x for x, _ in spline_points]
    yspline = [y for _, y in spline_points]
    plt.plot(xspline, yspline, '-')
    plt.plot(xpoints, ypoints, 'ro')
    # plt.show()
    n = len(xspline)
    for i in range(n-1):
        print((f(xspline[i]) - yspline[i])/2 * 100)
    # plt.plot(xpoints, ypoints, 'ro')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
