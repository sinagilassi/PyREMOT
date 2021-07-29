# heat of reaction
import json
import numpy as np
from library.plot import plotClass as pltc
import matplotlib.pyplot as plt
from library.saveResult import saveResultClass as sRes
from docs.rmtUtility import rmtUtilityClass as rmtUtil
from core.utilities import *
from docs.rmtThermo import *
# transport properties
from core.eqConstants import CONST_EQ_GAS_DIFFUSIVITY
from docs.gasTransPor import calGasDiffusivity
# component data
from data.componentData import componentDataStore

# component list
compList = ["H2", "CO2", "H2O", "CO", "CH3OH", "DME"]

# app data
appData = componentDataStore['payload']

# component data
compData = []
# component data index
compDataIndex = []

# init library
for i in compList:
    _loop1 = [
        j for j, item in enumerate(appData) if i in item.values()]
    compDataIndex.append(_loop1[0])

for i in compDataIndex:
    compData.append(appData[i])

# mole fraction
MoFri = [0.666466666666667,	0.266586666666667,	0.000100000000000000,
         0.0666466666666666	, 0.000100000000000000,	0.000100000000000000]

# temperature [K]
T = 523
# pressure [Pa]
P = 3500000

# molecular weight [g/mol]
MWi = rmtUtil.extractCompData(compData, "MW")
# critical temperature [K]
Tci = rmtUtil.extractCompData(compData, "Tc")
# critical pressure [Pa]
Pci = rmtUtil.extractCompData(compData, "Pc")

# prepare data
paramsData = {
    "MoFri": MoFri,
    "T": T,
    "P": P,
    "MWi": MWi,
    "CrTei": Tci,
    "CrPri": Pci
}

# diffusivity coefficient of components in the mixture
res = calGasDiffusivity(
    CONST_EQ_GAS_DIFFUSIVITY['Chapman-Enskog'], compList, paramsData)
# log
print("Dij: ", res)

# save modeling result
with open('test3.txt', 'a') as f:
    f.write(str(res))
