from scipy.linalg import lu_factor, lu_solve
from numpy import random, matrix, matmul, linalg
import copy
import numpy as np


matirces = [np.array([[10, 2, 0, 1], [1, 8, 2,2], [3, -2, -14, 1], [0, -1, 4, 6]])]

def dd(X):
    D = np.diag(np.abs(X))
    S = np.sum(np.abs(X), axis=1) - D
    if np.all(D > S):
        return True
    else:
        return False


def gauss(A, b, x, n):
    L = np.tril(A)
    U = A - L
    for i in range(n):
        x = np.dot(np.linalg.inv(L), b - np.dot(U, x))
        print(str(i).zfill(3))
        print(x)
    return x


def main():
    n, m = random.randint(low=2, high=10), random.randint(low=2, high=10)
    A = random.rand(n, m)
    A = matirxes[0]
    n, m = 4, 4
    x, b = random.rand(m), random.rand(m)
    print(f"n:{n} m:{m}")
    if n == m and linalg.det(A) != 0:  #LU
        lu, piv = lu_factor(A)
        solve = lu_solve((lu, piv), b)
        print(f"LU method {solve}" )
    if n != m: #LS
        print("Not a square matrix. Using LS method")
        tmp = copy.deepcopy(A)
        tmp.transpose()
        tmp_A = matmul(A.transpose(), A)
        lu, piv = lu_factor(tmp_A)
        solve = lu_solve((lu, piv), b)
        print(f"LS method {solve}")
    elif linalg.eigvals > 0: #LL
        L = linalg.cholesky(A)
        print(f"Cholesky method {L}")
    elif dd(A): #G-S
        gauss(A, b, x, 20)


if __name__ == "__main__":
    main()