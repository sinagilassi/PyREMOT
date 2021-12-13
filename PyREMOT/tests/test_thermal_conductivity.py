# CALCULATE GAS THERMAL CONDUCTIVITY
# ------------------------------------

# import modules/packages
# externals
import numpy as np
import matplotlib.pyplot as plt
# internals
from PyREMOT.library.plot import plotClass as pltc
from PyREMOT.library.saveResult import saveResultClass as sRes
from PyREMOT.docs.rmtUtility import rmtUtilityClass as rmtUtil
from PyREMOT.core.utilities import *
from PyREMOT.docs.rmtThermo import *
from PyREMOT.core.eqConstants import CONST_EQ_GAS_DIFFUSIVITY
from PyREMOT.docs.gasTransPor import calGasThermalConductivity, calGasViscosity, calMixturePropertyM1
from PyREMOT.data.componentData import componentDataStore, viscosityEqList

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
MoFri = [0.4998, 0.2499, 0.0001, 0.2499, 0.0001, 0.0001]

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

# thermal conductivity of components in the mixture
TheConi = calGasThermalConductivity(compList, T)
# log
print("TheConi: ", TheConi)
# mixture
TheConMix = calMixturePropertyM1(compNo, TheConi, MoFri, MWi)
print("TheConMix: ", TheConMix)
