# TEST FDM
# FINITE DIFFERENCE METHOD
# --------------------------

# import packages/modules
import numpy as np
# from scipy.integrate import
import matplotlib.pyplot as plt

#! DEFINE PROBLEM
# ----------------

# d2y/dx2 - y = 0
# x = 0: dy/dx = 0
# x = 1: y = 1

#! MATRIX FUNCTION
# ----------------


def a1fun(i, j):
    """
        first node - bc1
    """

    if j == i-1:
        a = 0
    elif j == i:
        a = -alpha
    elif j == i+1:
        a = 2
    else:
        a = 0

    # res
    return a


def f1fun():
    """
        f matrix
    """
    a = 0
    return a


def aifun(i, j):
    """
        interior nodes
    """

    if j == i-1:
        a = 1
    elif j == i:
        a = -alpha
    elif j == i+1:
        a = 1
    else:
        a = 0

    # res
    return a


def fifun():
    """
        f matrix
    """
    f = 0
    return f


def a2fun(i, j):
    """
        last nodes before bc2
    """

    if j == i-1:
        a = 1
    elif j == i:
        a = -alpha
    elif j == i+1:
        a = 0
    else:
        a = 0

    # res
    return a


def f2fun():
    """
        f matrix
    """
    f = -1
    return f


#! SETTING
# --------

# domain size
L = 1
# dx size
dx = 0.1
# number of nodes
N = 1/dx + 1

#! BOUNDARY CONDITION
# --------------------

# bc1: y1?
y0 = 'y2'
# bc2
yN = 1
# unknowns
M = N-1
ym = np.zeros(M, 1)
# x values
xm = np.zeros(M, 1)

#! MATRIX
# ------

# coefficient matrix
A = np.zeros(M, M)
# constant matrix
F = np.zeros(M, 1)
# coefficient constants
alpha = 2+(dx ^ 2)

for i in range(M):
    # x values
    xm[i, 1] = i/10 - 1/10
    # bc1
    if i == 0:
        for j in range(M):
            A[i][j] = a1fun(i, j)

        F[i][1] = f1fun()

    # interior nodes
    elif i > 0 and i < M:
        for j in range(M):
            A[i][j] = aifun(i, j)

        F[i][1] = fifun()

    # node before bc2
    elif i == M:
        for j in range(M):
            A[i][j] = a2fun(i, j)

        F(i, 1) = f2fun()

#! SOLUTION
# ----------

# [A][y] = [F] -> [y] = [F]/[A]
# y = A\F

# plot(xm',y', '-*')
