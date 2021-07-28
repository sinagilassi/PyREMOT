# TRANSPORT PROPERTIES OF GASES
# ------------------------------

# import packages/modules
from math import sqrt
import numpy as np
import re
# internals
from core.constants import Tref, R_CONST
from data.componentData import heatCapacityAtConstatPresureList, standardHeatOfFormationList
from core.utilities import roundNum
from core.eqConstants import CONST_EQ_GAS_DIFFUSIVITY
from rmtUtility import rmtUtilityClass as rmtUtil


def main():
    pass


def calGasDiffusivity(equation, compList, **params):
    """ 
    calculate gas diffusivity [m2/s]
    args:
        params: changes with equation
        eq1: Chapman-Enskog
    """
    # choose equation
    if equation == 1:
        return calGaDiEq1(compList, **params)
    else:
        return -1


def calGaDiEq1(compList, **params):
    """ 
    calculate based on Chapman-Enskog 
    args:
        params: 
            compList: component name list
            MoFri: mole fraction list
            T: temperature [K]
            P: pressure [Pa]
            MWi: molecular weight list [g/mol]
            CrTei: critical temperature [K]
            CrPri: critical pressure [bar]
    """
    # input
    MoFri = params['MoFri']
    T = params['T']
    P = params['P']
    MWi = params['MWi']
    CrTei = params['CrTei']
    CrPri = params['CrPri']
    # component no.
    compNo = len(compList)
    # e/K
    eK_Ratio = np.array([0.75*item for item in CrTei])
    # sigma - characteristic length of the intermolecular force law
    sigma = np.zeros(compNo)
    for i in range(compNo):
        _loop = (CrTei[i]/CrPri[i])**(1/3)
        sigma[i] = 2.44*_loop

    # e[i,j]
    eij = np.zeros((compNo, compNo))
    for i in range(compNo):
        for j in range(compNo):
            if i == j:
                eij[i][j] = 0
            else:
                eij[i][j] = sqrt(eK_Ratio[i]*eK_Ratio[j])

    # sigma[i,j]
    sigmaij = np.zeros((compNo, compNo))
    for i in range(compNo):
        for j in range(compNo):
            if i == j:
                sigmaij[i][j] = 0
            else:
                sigmaij[i][j] = 0.5*(sigma[i] + sigma[j])

    # omega[i,j]
    omegaij = np.zeros((compNo, compNo))
    for i in range(compNo):
        for j in range(compNo):
            if i == j:
                omegaij[i][j] = 0
            else:
                _Ts = T/eij[i][j]
                _omegaLoop = 44.54*(_Ts**-4.909) + 1.911*(_Ts**-1.575)
                omegaij[i][j] = _omegaLoop**0.10

    # diffusivity coefficient D[i,j]
    Dij = np.zeros((compNo, compNo))
    for i in range(compNo):
        for j in range(compNo):
            if i == j:
                Dij[i][j] = 0
            else:
                Dij[i][j] = (1e-4)*(0.0018583)*sqrt((T**3)*((1/MWi[i]) + (1/MWi[j]))) \
                    * (1/((P*9.86923e-6)*(sigmaij[i][j]**2)*omegaij[i][j]))

    # based on Blanc's law
    Dij_Cal = np.zeros((compNo, compNo))
    # diagonal matrix
    Dij_Transpose = np.transpose(Dij)
    Dij_New = Dij + Dij_Transpose

    for i in range(compNo):
        for j in range(compNo):
            if i == j:
                Dij_Cal[i][j] = 0
            else:
                Dij_Cal[i][j] = MoFri[j]/Dij_New[i][j]

    # mixture diffusivity coefficient D[i]
    Di = np.zeros(compNo)

    for k in range(compNo):
        Di[k] = np.sum(Dij_Cal[k, :])**(-1)

    # res
    return Di


if __name__ == "__main__":
    main()
