# ODE SOLVER
# -----------

# import package/module
import numpy as np
import matplotlib.pyplot as plt

# f in the IVP y’ = f(t,y), y(t0)=y0


def dFdtFun(t, z, params):
    a, b, c, d = params
    x, y = z
    return [a*x - b*x*y, -c*y + d*x*y]


def RK4(t0, tn, n, y0, f, params):
    """
    Runge-Kutta "Classic" Order 4 method
    """
    # params
    # params = (1.5, 1, 3, 1)
    # set
    h = abs(tn-t0)/n
    t = np.linspace(t0, tn, n+1)
    # size
    y0Shape = np.shape(y0)
    # y = np.zeros(n+1)
    # y matrix shape
    yMatShape = (y0Shape[0], n+1)
    y = np.zeros(yMatShape)
    y[:, 0] = y0
    for i in range(0, n):
        K1 = np.array(f(t[i], y[:, i], params))
        K2 = np.array(f(t[i]+h/2, y[:, i]+K1*h/2, params))
        K3 = np.array(f(t[i]+h/2, y[:, i]+K2*h/2, params))
        K4 = np.array(f(t[i]+h, y[:, i]+K3*h, params))
        _loopVar = h*(K1+2*K2+2*K3+K4)/6
        y[:, i+1] = y[:, i] + _loopVar
    return y


def AdBash3(t0, tn, n, y0, f, params):
    """
    Adams-Bashforth 3 Step Method
    """
    # params
    # params = (1.5, 1, 3, 1)
    #
    h = np.abs(tn-t0)/n
    t = np.linspace(t0, tn, n+1)
    # size
    y0Shape = np.shape(y0)
    # y = np.zeros(n+1)
    # y matrix shape
    yMatShape = (y0Shape[0], n+1)
    y = np.zeros(yMatShape)
    #
    y[:, 0:3] = RK4(t0, t0+2*h, 2, y0, f, params)
    K1 = np.array(f(t[1], y[:, 1], params))
    K2 = np.array(f(t[0], y[:, 0], params))
    for i in range(2, n):
        K3 = K2
        K2 = K1
        K1 = np.array(f(t[i], y[:, i], params))
        _loopVar = h*(23*K1-16*K2+5*K3)/12
        y[:, i+1] = y[:, i] + _loopVar
    return y


def PreCorr3(t0, tn, n, y0, f, params):
    """
    Adams-Bashforth 3/Moulton 4 Step Predictor/Corrector
    """
    h = np.abs(tn-t0)/n
    t = np.linspace(t0, tn, n+1)
    # y = np.zeros(n+1)
    y0Shape = np.shape(y0)
    # y = np.zeros(n+1)
    # y matrix shape
    yMatShape = (y0Shape[0], n+1)
    y = np.zeros(yMatShape)
    #
    # Calculate initial steps with Runge-Kutta 4
    # y[0:3] = RK4(t0,t0+2*h,2,y0)
    y[:, 0:3] = RK4(t0, t0+2*h, 2, y0, f, params)
    # K1 = f(t[1],y[1])
    # K2 = f(t[0],y[0])
    K1 = np.array(f(t[1], y[:, 1], params))
    K2 = np.array(f(t[0], y[:, 0], params))
    for i in range(2, n):
        K3 = K2
        K2 = K1
        # K1 = f(t[i],y[i])
        K1 = np.array(f(t[i], y[:, i], params))
        # Adams-Bashforth Predictor
        y[:, i+1] = y[:, i] + h*(23*K1-16*K2+5*K3)/12
        # K0 = f(t[i+1],y[i+1])
        K0 = np.array(f(t[i+1], y[:, i+1], params))
        # Adams-Moulton Corrector
        y[:, i+1] = y[:, i] + h*(9*K0+19*K1-5*K2+K3)/24
    return y


# run
# n = 300
# t0 = 0
# tn = 15
# y0 = [10, 5]
# t = np.linspace(t0, tn, n+1)
# paramsSet = (1.5, 1, 3, 1)
# funSet = dFdtFun

# sol = RK4(t0, tn, n, y0, paramsSet)
# sol = AdBash3(t0, tn, n, y0, funSet, paramsSet)
# sol = PreCorr3(t0, tn, n, y0, funSet, paramsSet)

# z = sol
# plt.plot(t, z[0, :], t, z[1, :])
# plt.xlabel('t')
# plt.legend(['x', 'y'], shadow=True)
# plt.title('Lotka-Volterra System')
# plt.show()
