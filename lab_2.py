from numpy import poly1d, arange, polyval
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import math


def cheb(a, b, k):
    x = []
    for i in range(k+1):
        root = 0.5*(a+b) + 0.5*(b-a)*math.cos((2*i-1)/k*math.pi)
        x.append(root)
    return sorted(list(set(x)))


def lagrange(x, w):
    M = len(x)
    p = poly1d(0.0)
    for j in range(M):
        pt = poly1d(w[j])
        for k in range(M):
            if k == j:
                continue
            fac = x[j]-x[k]
            pt *= poly1d([1.0, -x[k]])/fac
        p += pt
    return p


def f(x):
    if 1 <= x <= 2:
        return 1/3 * x + 5/3
    elif 2 < x <= 4:
        return 4
    elif 4 < x <= 6:
        return -(1/2) * x + 9/2
    return 1

def main():
    w = [1, 2, 4, 4, 2, 1]
    x = [-2, 1, 2, 4, 5, 7]

    xchen = [x for x in arange(-2, 12, 1)]
    xvalcheb = cheb(0, 11, 50)
    print("Xcheb " + str(xvalcheb))
    print("Xchen " + str(xchen))
    yval = list(map(f, xchen))
    yvalcheb = list(map(f, xvalcheb))
    print("Yval " + str(yval))
    chebp = lagrange(xchen, yval)
    polcheb = lagrange(xvalcheb, yvalcheb)
    print(polcheb)
    print(chebp)
    xval = [x for x in arange(0, 10, 0.1)]
    #for x in xchen:
    #    print(chebp(x), f(x))
    #plt.plot(xcheb, yval, "go")
    #yvalcheb = [polcheb(x) for x in xval]
    yvalnorm = [chebp(x) for x in xval]
    print("Yvalche " + str(yvalcheb))
    plt.ylim(ymax=10)
    plt.plot(xval, yvalnorm)
    #plt.plot(xval, yvalcheb)
    #plt.plot(xchen, yval, "ro")
    plt.show()


if __name__ == "__main__":
    main()