import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

# Mass Transfer in a Catalytic Particle [spherical shape]
# --------------------------------------
# pde (Ci,r) => (y,x)
# # [dy/dt] = [d^2y/dx^2] + [2/x][dy/dx] - [alpha][rhoc][r]
# BCs:
# x = 0: dy/dx = 0
# x = 1: [Di][dy/dx] = [beta][y - yb]
# rhoc: catalyst density [kg catalyst/m^3 particle]
# Di: diffusion coefficient [m^2/s]
# alpha: rp^2/Di
# beta: hi*rp/Di

# NOTE
# reaction term should be defined!

# Constants
# ----------
# diffusivity coefficient [m^2/s]
Di = 2.273e-06
# mass transfer coefficient [m/s]
h = 0.0273
# particle diameter [m]
Rp = 0.002
# alpha
alpha = (Rp**2)/Di
# beta
beta = Rp*h/Di
# bulk concentration [mol/m^3]
Cb = 1e-2
# initial concentration [mol/m^3]
Ci0 = 1e-3

# Approximate Solution
# ---------------------
# y = d1 + d2*x^2 + d3*x^4 + d4*x^6 + ... + d[N+1]*x^2N

# Define Collocation Points
# -------------------------
# x1,x2,x3,x4
# x1 = 0
# x1 = 0.28523
# x2 = 0.76505
# x3 = 1

# 6 points [spherical shape]
# x1 = 0
x1 = 0.215353
x2 = 0.420638
x3 = 0.606253
x4 = 0.763519
x5 = 0.885082
x6 = 0.965245
x7 = 1

# initial boundary condition
X0 = 0
# last boundary condition
Xn = 1

# collocation points
# 4 points
# Xc = np.array([x1, x2, x3])
# 6 points
Xc = np.array([x1, x2, x3, x4, x5, x6, x7])
# 5 points
# Xc = np.array([x1, x2, x3, x4, x5, x6])

# collocation + boundary condition points
N = np.size(Xc)
# collocation points number
Nc = N - 1

# Functions
# ---------


def fQ(j, Xc):
    # Q matrix
    return Xc**(2*j)


def fC(j, Xc):
    # C matrix
    if j == 0:
        return 0
    else:
        return (2*j)*(Xc**(2*j-1))


def fD(j, Xc):
    # D matrix
    if j == 0:
        return 0
    if j == 1:
        return 2
    else:
        return 2*j*(2*j-1)*(Xc**(2*j-2))

# R (only linear coefficient)
# first point is not BC1
# last point is BC2


def fR(i, j, Aij, Bij, N):
    # interior points [0,1,...,N-1]
    # BC2: N
    if i < N - 1:
        F = Bij + (2/Xc[i])*Aij
    # last node (BC2)
    else:
        if j == N - 1:
            F = Aij - beta
        else:
            F = Aij
    return F

# f - constant values matrix


def ff(i, A, B, N):
    if i < N-1:
        F = 0
    else:
        F = beta*Cb
    return F

# nonlinear function
# BC2: first boundary type


def fdydt(t, y, params):
    # parameters
    nMat = params['N']
    R = params['R']
    f = params['f']
    # matrix size
    n = nMat
    fxMat = np.zeros(n)

    for i in range(n):
        # reaction rate expression term (linear/nonlinear)
        # consumption/production
        reaTerm = -10*y[i]
        if i < n - 1:
            # for internal points
            Fsum = 0
            Fj = 0
            for j in range(n):
                if j == i:
                    Fsum = R[i][j]*y[j] + reaTerm
                    Fj = Fsum + Fj
                else:
                    Fsum = R[i][j]*y[j]
                    Fj = Fsum + Fj
            # for point[i]
            fxMat[i] = Fj + f[i]
        else:
            # for last point (BC2)
            Fsum = 0
            Fj = 0
            for j in range(n):
                if j < n - 1:
                    Fsum = R[i][j]*y[j]
                    Fj = Fsum + Fj
                else:
                    Fsum = (R[i][j] - beta)*y[j]
                    Fj = Fsum + Fj
            # for point[N+1]
            fxMat[i] = Fj + f[i]

        # reset
        Fj = 0
        Fsum = 0

    # return
    return fxMat

# Evaluate Solution at Collocation Points
# ----------------------------------------
# point x1
# y(1) = d1 + d2*x(1)^2 + d3*x(1)^4 + d4*x(1)^6
# point x2
# y(2) = d1 + d2*x(2)^2 + d3*x(2)^4 + d4*x(2)^6
# point x3
# y(1) = d1 + d2*x(3)^2 + d3*x(3)^4 + d4*x(3)^6


# define Q matrix
Q = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        Q[i][j] = fQ(j, Xc[i])
# print("Q Matrix: ", Q)

# y = Q*d
# d = y/Q = y*[Q inverse]

# Evaluate First Derivative at Collocation Points
# ------------------------------------------------
# point x1
# dy(1) = 0 + 2*d2*x1 + 4*d3*x1^3 + ...

# dy = [dy1 dy2 dy3 dy4];
# C0 = [
# 0 1 2*x1 3*x1^2;
# 0 1 2*x2 3*x2^2;
# 0 1 2*x3 3*x3^2;
# 0 1 2*x4 3*x4^2
# ]

# define C matrix
C = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        C[i][j] = fC(j, Xc[i])
# print("C Matrix: ", C)

# d = [d1 d2 d3 d4];
# y' = A*y
# Q inverse
invQ = np.linalg.inv(Q)
# A matrix
A = np.dot(C, invQ)
# print("A Matrix: ", A)

# Evaluate Second Derivative at Collocation Points
# ------------------------------------------------
# point x1
# ddy(1) = 0 + 2*d2 + 12*d3*x1^2 + ...

# ddy = [ddy1 ddy2 ddy3 ddy4];
# D0 = [
# 0 0 2 6*x1;
# 0 0 2 6*x2;
# 0 0 2 6*x3;
# 0 0 2 6*x4
# ]

# define D matrix
D = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        D[i][j] = fD(j, Xc[i])
# print("D Matrix: ", D)

# d = [d1 d2 d3 d4];
# y'' = B*y
# B matrix
B = np.dot(D, invQ)
# print("B Matrix: ", B)

# Residual Formulation
# ---------------------
# linear parts (RHS)
# non-linear parts (LHS) appers in non-linear function
# R1
# R1 = [A(1,1)-6 A(1,2) A(1,3) A(1,4)]
# R4
# R4 = [A(4,1) A(4,2) A(4,3) A(4,4)]
# R2/R3
# R2 = [1/6*B(2,1)-A(2,1) 1/6*B(2,2)-A(2,2) 1/6*B(2,3)-A(2,3) 1/6*B(2,4)-A(2,4)]
# R3 = [1/6*B(3,1)-A(3,1) 1/6*B(3,2)-A(3,2) 1/6*B(3,3)-A(3,3) 1/6*B(3,4)-A(3,4)]
# R = [R1; R2; R3; R4]

# define R matrix
R = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        R[i][j] = fR(i, j, A[i][j], B[i][j], N)

print("R Matrix: ", R)

# f matrix - constant values matrix
# define f matrix
f = np.zeros(N)
for i in range(N):
    f[i] = ff(i, A, B, N)

print("f Matrix: ", f)

# System of ODEs
# ----------------------------------------
# Xc0
Xc0 = np.copy(Xc)

# initial conditions (IC)
y0Set = Ci0*np.ones(N)
print("y0Set: ", y0Set)

# time points [s]
tFinal = 0.5
t = (0.0, tFinal)
ts = 5
tSpan = np.linspace(0, tFinal, ts)

# solve equation
# parameters
params = {
    "N": N,
    "R": R,
    "f": f
}
# solve ODE
sol = solve_ivp(fdydt, t, y0Set, method="LSODA", t_eval=tSpan, args=(params,))
# y [ode(i),t]
# solY = sol.y
# y [y(Xc(i)),Xc(i)]
solY = np.transpose(sol.y)

# NOTE
# find value for x = 0: dy/dx = 0
ds = []
solPoint = []
for i in range(ts):
    # yi loop
    yiLoop = solY[i, :]
    # d loop
    dLoop = np.dot(invQ, yiLoop)
    print("d Matrix: ", dLoop)
    ds.append(dLoop)

    # all points [0,1]
    # y[0]
    yi0Loop = dLoop[0]
    # yi
    yisLoop = [yi0Loop, *yiLoop]
    # xi
    xisLoop = [0, *Xc]
    # save loop
    solPoint.append({
        "xi": xisLoop,
        "yi": yisLoop
    })

# plot results
# for i in range(ts):
#     plt.plot(solPoint[i]['xi'], solPoint[i]['yi'], label="t=" + str(tSpan[i]))

#  linestyle='--', marker='o'


# def appFun(x, d):
#     # approximate function
#     nd = np.size(d)
#     fsum = 0
#     F = 0
#     for i in range(nd):
#         # F = d(1) + d(2)*x^2 + d(3)*x^4 + d(4)*x^6 + ...
#         fsum = d[i]*(x**(2*i))
#         F = F + fsum
#     # res
#     return F


# evaluate in orthogonal points
# evalPoint = []
# yVal = []
# # arbitrary points [0,1]
# # Xc0 = [0.1,0.4,0.64,0.85]
# Xc0 = np.linspace(0, 1, ts)
# print("Xc arbitrary points: ", Xc0)

# for k in range(ts):
#     # yVal set
#     yVal.append([])
#     for i in range(ts):
#         _loop = appFun(Xc0[i], ds[k])
#         yVal[k].append(_loop)
#     # save loop
#     evalPoint.append({
#         "xi": Xc0,
#         "yi": yVal[k]
#     })


# plot results
# for i in range(ts):
#     plt.plot(evalPoint[i]['xi'], evalPoint[i]['yi'], "o")

# plt.ylabel('Rate')
# plt.xlabel('Length')
# plt.legend(loc='best')
# plt.show()
