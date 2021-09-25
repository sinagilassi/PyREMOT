# model input

# import package/module
import numpy as np
import core.constants as CONST
from core.utilities import roundNum
from core.config import MOLE_FRACTION_ACCURACY, CONCENTRATION_ACCURACY


def setFeedMoleFraction(H2COxRatio, CO2COxRatio):
    """
        set inlet feed mole fraction
    """
    # feed properties
    # H2/COx ratio
    # H2COxRatio = 2.0
    # CO2/CO ratio
    # CO2COxRatio = 0.8
    # mole fraction
    y0_H2O = 0.00001
    y0_CH3OH = 0.00001
    y0_DME = 0.00001
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
    feedMoFri = np.array([y0_H2, y0_CO2, y0_H2O, y0_CO,
                         y0_CH3OH, y0_DME], dtype=np.float32)
    # res
    return feedMoFri


def calConcentration(MoFri, P, T, unit="kmol/m^3"):
    """
    calculate concentration [kmol/m^3] | [mol/m^3]
    args:
        MoFri: mole fraction
        P: pressure [Pa]
        T: temperature [K]
    output:
        Ci: component concentration [kmol/m^3] | [mol/m^3]
    """
    # component no
    componentNo = len(MoFri)

    # concentraion
    Ci = np.zeros(componentNo)

    for i in range(componentNo):
        CiLoop = (P/(CONST.R_CONST*T))*MoFri[i]/1000
        Ci[i] = CiLoop
    # unit checking
    if unit == 'mol/m^3':
        Ci = 1e3*Ci

    # accuracy set
    res = roundNum(Ci, CONCENTRATION_ACCURACY)
    # res
    return res


def calTotalConcentration(Ci):
    """
        calculate total concentration [kmol/m^3]
    """
    # res
    res = roundNum(np.sum(Ci), CONCENTRATION_ACCURACY)
    return res
