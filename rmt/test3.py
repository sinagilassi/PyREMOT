# heat of reaction
import numpy as np
from library.plot import plotClass as pltc
import matplotlib.pyplot as plt
from library.saveResult import saveResultClass as sRes
from docs.rmtUtility import rmtUtilityClass as rmtUtil
from core.utilities import *
from docs.rmtThermo import *
# component data
from data.componentData import componentDataStore

# component list
compList = ["H2", "CO2", "H2O", "CO", "CH3OH", "DME"]

# app data
appData = componentDataStore['payload']

# component data
compData = []

# init library
for i in compList:
    _loop1 = [
        item for item in appData if i == item['symbol']]
    compData.append(_loop1[0])

# mole fraction
MoFri = [0.4998, 0.2499, 0.0001, 0.2499, 0.0001, 0.0001]

# temperature [K]
T = 520
# pressure [Pa]
P = 5e6

# molecular weight [g/mol]
MWi = rmtUtil.extractCompData(compData, "MW")
# critical temperature [K]
Tci = rmtUtil.extractCompData(compData, "Tc")
# critical pressure [Pa]
Pci = rmtUtil.extractCompData(compData, "Pc")

# prepare data
params = {
    "MoFri": MoFri,
    "T": T,
    "P": P,
    "MWi": MWi,
    "CrTei": Tci,
    "CrPri": Pci
}

print("params: ", params)
