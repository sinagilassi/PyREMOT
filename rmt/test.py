import numpy as np
from library.plot import plotClass as pltc
import matplotlib.pyplot as plt
from library.saveResult import saveResultClass as sRes

from docs.rmtThermo import *

a1 = np.array([1, 2, 3])
a2 = np.array([5, 2, 8])
b = np.array([3, 4, 5])
a = [a1, a2]
c0 = [10, 20, 30]
c = [[1, 2, 3], [5, 6, 9], [7, 2, 9]]
# pltc display
# pltc.plot2D(a, b)
# for i in range(len(c)):
#     plt.plot(b, c[i], label='line 1')

# line1 = plt.plot(b, a1, label='line 1')
# plt.legend()
# plt.show()

data = [
    {
        "x": b,
        "y": a1,
        "leg": "line 1"
    },
    {
        "x": b,
        "y": a2,
        "leg": "line 2"
    }
]

# pltc.plots2D(data, "time", "mole flowrate", "model cas 1")

# lineList = pltc.plots2DSetXYList(c0, c)
# print(f"lineList: {lineList}")

# lineData = pltc.plots2DSetDataList(lineList, ["lb1", "lb2", "lb3"])
# print(f"lineData: {lineData}")

# sRes.saveListToText(c0)
# sRes.saveListToCSV(c, ["name", "age", "year"])

# aStr = a1.join(",")
# print(aStr)


def f(*args):
    print(args)


# f("a", "b", "c")


# text
# a1 = calHeatCapacityAtConstantPressure(
#     ["H2", "CO2", "H2O", "CO", "CH3OH", "DME"], 300)
# print(a1)

# z1 = np.array([1, 2])
# z2 = np.array([-1, 2])
# # z3 = np.dot(z1, z2)
# z3 = calMixtureHeatCapacityAtConstantPressure(z1, z2)
# print(z3)

# reactions
R1 = "CO2 + 3H2 <=> CH3OH + H2O"
R2 = "CO + H2O <=> H2 + CO2"
R3 = "2CH3OH <=> DME + H2O"
# z1 = calStandardEnthalpyOfReaction(R3)
# print(z1)

# fun


def cpFun(T): return eval("T + 1")


def z1(T): return eval("T + 1")


print(z1(1))
