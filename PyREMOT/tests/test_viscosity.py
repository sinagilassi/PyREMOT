# heat of reaction
import numpy as np
from library.plot import plotClass as pltc
import matplotlib.pyplot as plt
from library.saveResult import saveResultClass as sRes
from docs.rmtUtility import rmtUtilityClass as rmtUtil
from core.utilities import *
from docs.rmtThermo import *
# transport properties
from core.eqConstants import CONST_EQ_GAS_DIFFUSIVITY
from docs.gasTransPor import calGasViscosity, calMixturePropertyM1
# component data
from data.componentData import componentDataStore, viscosityEqList

# component list
compList = ["H2", "CO2", "H2O", "CO", "CH3OH", "DME"]
# component number
compNo = len(compList)

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
MoFri = [0.50, 0.25, 0.0001, 0.25, 0.0001, 0.0001]

# temperature [K]
T = 350
# pressure [Pa]
P = 3500000

# molecular weight [g/mol]
MWi = rmtUtil.extractCompData(compData, "MW")
# critical temperature [K]
Tci = rmtUtil.extractCompData(compData, "Tc")
# critical pressure [Pa]
Pci = rmtUtil.extractCompData(compData, "Pc")

# viscosity of components in the mixture
Vii = calGasViscosity(compList, T)
# log
print("GaVii: ", Vii)
# viscosity mixture
ViiMix = calMixturePropertyM1(compNo, Vii, MoFri, MWi)
print("GaVii Mix: ", ViiMix)
