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
MoFri = [0.50,	0.25,	0.0001,
         0.25	, 0.0001,	0.0001]

# temperature [K]
T = 523
# pressure [Pa]
P = 3500000


# Cp mean list
CpMeanList = calMeanHeatCapacityAtConstantPressure(compList, T)
