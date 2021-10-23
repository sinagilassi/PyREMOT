
# import module/package
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
# internals
from solvers.solFiEl import FiElClass
from solvers.solCatParticle2 import FiElCatParticleClass


# number of elements
NuEl = 5

# init finite element class
FiElClassInit = FiElClass(NuEl)
# res
res0 = FiElClassInit.initFiEl()
# print("res0: ", res0)
# ->
NuEl = res0['NuEl']
NuToCoPo = res0['NuToCoPo']
hi = res0['hi']
li = res0['li']
xi = res0['xi']
Xc = res0['Xc']
N = res0['N']
Q = res0['Q']
A = res0['A']
B = res0['B']
odeNo = 1

# build matrix
FiElCatParticleClassInit = FiElCatParticleClass(
    NuEl, NuToCoPo, hi, Xc, N, Q, A, B, odeNo)


# NOTE
### system of solve nonlinear equations ###
# initial guess
x0 = 1*np.ones(NuToCoPo)

# define function


def funSet(x):
    '''
    build nonlinear matrix
    '''
    dxdtMat = np.zeros(NuToCoPo)
    # ! define linear/nonlinear term ***
    nlMat = np.zeros((NuToCoPo, 1))
    # reshape x
    x_Reshaped = np.reshape(x, (NuToCoPo, 1))
    # set
    ocSet = 0
    n = 0
    # loop
    for i in range(NuToCoPo):
        if i == ocSet:
            nlMat[i, 0] = 0
            ocSet = 3*(n+1)
            n = n + 1
        else:
            nlMat[i, 0] = -2*(x[i]**2)

    # BC1
    nlMat[0, 0] = -6*(x[0])

    # res
    res1 = FiElCatParticleClassInit.buildMatrix(x)
    # print("res1: ", res1)
    # ->
    Ri = res1['Ri']
    fi = res1['fi']

    # [R][Y]=[RY]
    RYMatrix = np.matmul(Ri, x_Reshaped)

    # res
    dxdtMat = RYMatrix + fi + nlMat

    # flatten
    dxdtMat_Flat = dxdtMat.flatten()

    return dxdtMat_Flat


# fsolve
yi = optimize.fsolve(funSet, x0, args=())
# plot
plt.plot(xi, yi, "-o")
plt.ylabel('Y')
plt.xlabel('Dimensionless Length')
plt.legend(loc='best')
plt.show()
