# model input

# import package/module
import numpy as np

# feed properties
# ----------------
# H2/COx ratio
H2COxRatio = 2.0
# CO2/CO ratio
CO2COxRatio = 0.8
# mole fraction
y0_H2O = 0.0001
y0_CH3OH = 0.0001
y0_DME = 0.0001
# total molar fraction
tmf0 = 1 - (y0_H2O + y0_CH3OH + y0_DME)
# COx
COx = tmf0/(H2COxRatio + 1)
# mole fraction
y0_H2 = H2COxRatio*COx
y0_CO2 = CO2COxRatio*COx
y0_CO = COx - y0_CO2
# total mole fraction
tmf = y0_H2 + y0_CO + y0_CO2 + y0_H2O + y0_CH3OH + y0_DME
# CO2/CO2+CO ratio
CO2CO2CORatio = y0_CO2/(y0_CO2+y0_CO)
# res
feedMoFri = np.array([y0_H2, y0_CO2, y0_H2O, y0_CO, y0_CH3OH, y0_DME])
