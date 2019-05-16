import math
import matplotlib.pyplot as plt
import numpy as np

N = 4


def generate_points():
    global N
    points = set()
    left_fun = [lambda x: -3 - x, lambda x: 3 + x]
    right_fun = [lambda x: 3 - x, lambda x: -3 + x]
    for i in np.arange(-3, 1, 1.5):
        for fun in left_fun:
            points.add((i, fun(i)))
    for i in np.arange(0, 4, 1.5):
        for fun in right_fun:
            points.add((i, fun(i)))
    N = len(points)
    return list(points)


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


def generate_matrix(polar_points):
    m = N // 2
    rows = []
    for i in range(len(polar_points)):
        row = [1, ]
        thetha = polar_points[i][0]
        for j in range(1, m + 1):
            if N % 2 != 0 or j != m:
                row.append(math.cos(j * thetha))
            row.append(math.sin(j * thetha))
        rows.append(row)
    print(f"Dim: n:{len(rows)} m:{len(rows[0])}")
    print(f"Rows: {rows}")
    matrix = np.matrix(rows)
    print(f"Matrix: {matrix}")
    return matrix


def fur_value(B, thetha):
    if N % 2 == 0:
        m = N // 2
    else:
        m = (N - 1) // 2
    b = B.tolist()
    res = 0
    b = [x[0] for x in b]
    res += b[0]
    i = 1
    for j in range(1, m):
        res += b[i] * math.cos(j * thetha)
        res += b[i+1] * math.sin(j * thetha)
        i += 2

    return res


def main():
    points = generate_points()
    polar_points = transform_polar(points)
    print(points)
    print("Polar: " + str(polar_points))
    plt.scatter([x[0] for x in points], [y[1] for y in points])
    plt.show()
    print(f"n: {len(points)} polr_n: {len(polar_points)}")
    A = generate_matrix(polar_points)
    rows = []
    for thetha, r in polar_points:
        rows.append([r, ])

    X = np.asmatrix(rows)
    B = np.linalg.solve(A, X)
    steps = 100
    fur_points = []
    for thetha in np.arange(0, 2 * math.pi, math.pi / steps):
        fur_points.append((thetha, fur_value(B, thetha)))
    plt.polar([x[0] for x in fur_points], [x[1] for x in fur_points])
    plt.polar([x[0] for x in polar_points], [x[1] for x in polar_points], 'ro')
    for x, y in polar_points:
        print(f"{fur_value(B, x)} == {y}")
    #plt.show()


if __name__ == "__main__":
    main()
