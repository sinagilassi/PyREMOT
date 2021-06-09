import numpy as np
from library.plot import plotClass as pltc
import matplotlib.pyplot as plt
from library.saveResult import saveResultClass as sRes

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

aStr = a1.join(",")
print(aStr)
