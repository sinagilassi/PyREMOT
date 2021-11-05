# heat of reaction
import numpy as np
from library.plot import plotClass as pltc
import matplotlib.pyplot as plt
from library.saveResult import saveResultClass as sRes
from docs.rmtUtility import rmtUtilityClass as rmtUtil
from core.utilities import *
from docs.rmtThermo import *


# component list
comList = ["H2", "CO2", "H2O", "CO", "CH3OH", "DME"]

# mole fraction
MoFri = [0.4998, 0.2499, 0.0001, 0.2499, 0.0001, 0.0001]

# temperature [K]
T = 900

# reactions
R1 = "CO2 + 3H2 <=> CH3OH + H2O"
R2 = "CO + H2O <=> H2 + CO2"
R3 = "2CH3OH <=> DME + H2O"
# z1 = calStandardEnthalpyOfReaction(R3)
# print(z1)

reactionSet = {
    "R1": "CO2 + 3H2 <=> CH3OH + H2O",
    "R2": "CO + H2O <=> H2 + CO2",
    "R3": "2CH3OH <=> DME + H2O",
}

# reaction list
reactionList = rmtUtil.buildReactionList(reactionSet)
print(f"reactionList: {reactionList}")

# reaction list sorted
reactionListSorted = rmtUtil.buildReactionCoefficient(reactionSet)
print(f"reactionListSorted: {reactionListSorted}")

# reaction stoichiometric coefficient vector
reactionStochCoeff = rmtUtil.buildReactionCoeffVector(
    reactionListSorted)
print(f"reactionStochCoeff: {reactionStochCoeff}")

# standard heat of reaction at 25C [kJ/kmol]
StHeRe25 = np.array(
    list(map(calStandardEnthalpyOfReaction, reactionList)))
print(f"StHeRe25: {StHeRe25}")

# heat capacity at constant pressure of mixture Cp [kJ/kmol.K]
# Cp mean list
CpMeanList = calMeanHeatCapacityAtConstantPressure(comList, T)
print(f"Cp mean list: {CpMeanList}")

# Cp mixture
CpMeanMixture = calMixtureHeatCapacityAtConstantPressure(
    MoFri, CpMeanList)
print(f"Cp mean mixture: {CpMeanMixture}")

# enthalpy change from Tref to T [kJ/kmol]
# enthalpy change
EnChList = np.array(calEnthalpyChangeOfReaction(reactionListSorted, T))
print(f"EnChList: {EnChList}")

# heat of reaction at T [kJ/kmol]
HeReT = np.array(EnChList + StHeRe25)
print(f"HeReT: {HeReT}")

# overall heat of reaction
# OvHeReT = np.dot(Ri, HeReT)
# print(f"HeReT: {OvHeReT}")
