# TRANSPORT PROPERTIES OF GASES
# ------------------------------

# import packages/modules
from math import sqrt
import numpy as np
import re
# internals
from PyREMOT.docs.rmtUtility import rmtUtilityClass as rmtUtil
# core
from PyREMOT.core import Tref, R_CONST
from PyREMOT.core import roundNum
from PyREMOT.core import CONST_EQ_GAS_DIFFUSIVITY, CONST_EQ_GAS_VISCOSITY
# data
from PyREMOT.data import viscosityEqList
from PyREMOT.data import viscosityList
from PyREMOT.data.dataGasThermalConductivity import TherConductivityList
from PyREMOT.data.componentData import thermalConductivityEqList


def main():
    pass


# NOTE
### diffusivity ###

def calGasDiffusivity(equation, compList, params):
    """ 
    calculate gas diffusivity [m2/s]
    args:
        params: changes with equation
        eq1: Chapman-Enskog
    """
    # choose equation
    if equation == 1:
        return calGaDiEq1(compList, params)
    else:
        return -1


def calGaDiEq1(compList, params):
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
        for j in range(i, compNo):
            if i == j:
                eij[i][j] = 0
            else:
                eij[i][j] = sqrt(eK_Ratio[i]*eK_Ratio[j])

    # sigma[i,j]
    sigmaij = np.zeros((compNo, compNo))
    for i in range(compNo):
        for j in range(i, compNo):
            if i == j:
                sigmaij[i][j] = 0
            else:
                sigmaij[i][j] = 0.5*(sigma[i] + sigma[j])

    # omega[i,j]
    omegaij = np.zeros((compNo, compNo))
    for i in range(compNo):
        for j in range(i, compNo):
            if i == j:
                omegaij[i][j] = 0
            else:
                _Ts = T/eij[i][j]
                _omegaLoop = 44.54*(_Ts**-4.909) + 1.911*(_Ts**-1.575)
                omegaij[i][j] = _omegaLoop**0.10

    # diffusivity coefficient D[i,j]
    Dij = np.zeros((compNo, compNo))
    for i in range(compNo):
        for j in range(i, compNo):
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

# NOTE
### viscosity ###


def calGasVisEq1(params, T):
    """ 
    gas viscosity equation 1 - Pa.s
    args:
        params: 
            equation parameters list [A,B,C,D]
        T: temperature [K]
    """
    # try/except
    try:
        A = params[0]
        B = params[1]
        C = params[2]
        D = params[3]
        _res = A*1e-6*(T**B)/(1+C*(1/T)+D*(T**-2))
        return _res
    except Exception as e:
        raise


def calGasVisEq2(eqExpr, T):
    """ 
    gas viscosity equation - Pa.s
    args:
        eqExpr: equation expression
        T: temperature [K]
    """
    # try/except
    try:
        return eval(eqExpr)
    except Exception as e:
        raise


def calGasViscosity(comList, T):
    """
        cal: gas viscosity at low pressure 
        unit: [Pa.s]

        args:
            comList: component name list
            T: temperature [K]
    """
    # try/except
    try:
        # heat capacity
        _Vii = []

        # load data
        loadEqData = viscosityEqList
        loadData = viscosityList

        for i in comList:
            # get id
            eqIdData = [item['id']
                        for item in loadEqData if i == item['symbol']]
            # get eq parameters
            eqData = [{"eqParams": item['eqParams'], "eqExpr": item['eqExpr']}
                      for item in loadData if i == item['symbol']]
            # check
            _eqLen = len(eqIdData) + len(eqData)
            if _eqLen > 0:
                _eqIdSet = eqIdData[0]
                _eqData = eqData[0]
                if _eqIdSet == 1:
                    _eqParams = _eqData.get('eqParams')
                    _res = calGasVisEq1(_eqParams, T)
                    _Vii.append(_res)
                elif _eqIdSet == 2:
                    _eqExpr = _eqData.get('eqExpr')
                    # build fun
                    _res = calGasVisEq2(_eqExpr, T)
                    _Vii.append(_res)
                else:
                    print('viscosity data not found, update app database!')
                    raise
            else:
                print("component not found, update the app database!")
                raise

        # convert to numpy array
        Vii = np.array(_Vii)

        # res
        return Vii
    except Exception as e:
        print(e)


# NOTE
### mixture property ###

def calMixturePropertyM1(compNo, Xi, MoFri, MWi):
    '''
    calculate mixture property M1
        Method of Wilke
    args:
        compNo: component number
        Xi: property name []
        MoFri: mole fraction [-]
        MWi: molecular weight [g/mol]
    '''
    try:

        # wilke res
        wilkeCo = np.zeros((compNo, compNo))
        for i in range(compNo):
            for j in range(compNo):
                if i == j:
                    wilkeCo[i, j] = 1
                else:
                    if i < j:
                        # wilke coefficient mix
                        A = 1 + sqrt(Xi[i]/Xi[j])*((MWi[j]/MWi[i])**(1/4))
                        AA = A**2
                        B = 8*(1+(MWi[i]/MWi[j]))
                        BB = sqrt(B)
                        wilkeCo[i, j] = AA/BB
                    else:
                        C = (Xi[i]/Xi[j])*(MWi[j]/MWi[i]) * wilkeCo[j, i]
                        wilkeCo[i, j] = C
        # vars
        A = np.zeros(compNo)
        B = np.zeros((compNo, compNo))
        # mixture property
        mixProp = np.zeros(compNo)
        for i in range(compNo):
            A[i] = Xi[i]*MoFri[i]
            for j in range(compNo):
                B[i, j] = MoFri[j]*wilkeCo[i, j]
            # set
            mixProp[i] = A[i]/np.sum(B[i, :])

        mixPropVal = np.sum(mixProp)
        # res
        return mixPropVal
    except Exception as e:
        print(e)


# NOTE
### thermal conductivity ###

def calGasThermalConductivity(comList, T):
    """
        cal: gas thermal conductivity at low pressure 
        unit: [W/m.K]

        args:
            comList: component name list
            T: temperature [K]
    """
    # try/except
    try:
        # thermal conductivity list
        _ThCoi = []

        # load data
        loadEqData = thermalConductivityEqList
        loadData = TherConductivityList

        for i in comList:
            # get id
            eqIdData = [item['id']
                        for item in loadEqData if i == item['symbol']]
            # get eq parameters
            eqData = [{"eqParams": item['eqParams'], "eqExpr": item['eqExpr']}
                      for item in loadData if i == item['symbol']]
            # check
            _eqLen = len(eqIdData) + len(eqData)
            if _eqLen > 0:
                _eqIdSet = eqIdData[0]
                _eqData = eqData[0]
                if _eqIdSet == 1:
                    _eqParams = _eqData.get('eqParams')
                    _res = calGasTherCondEq1(_eqParams, T)
                    _ThCoi.append(_res)
                elif _eqIdSet == 2:
                    _eqExpr = _eqData.get('eqExpr')
                    # build fun
                    _res = calGasVisEq2(_eqExpr, T)
                    _ThCoi.append(_res)
                else:
                    print('viscosity data not found, update app database!')
                    raise
            else:
                print("component not found, update the app database!")
                raise

        # convert to numpy array
        ThCoi = np.array(_ThCoi)

        # res
        return ThCoi
    except Exception as e:
        print(e)


def calGasTherCondEq1(params, T):
    """ 
    gas thermal conductivity equation 1 - W/m.K
    args:
        params: 
            equation parameters list [C1, C2, C3, C4]
        T: temperature [K]
    """
    # try/except
    try:
        C1 = params[0]
        C2 = params[1]
        C3 = params[2]
        C4 = params[3]
        _var1 = C1*(T**C2)
        _var2 = 1 + (C3/T) + C4/(T**2)
        _res = _var1/_var2
        return _res
    except Exception as e:
        raise


def calGasTherCondEq2(eqExpr, T):
    pass


def calTest():
    return 1


if __name__ == "__main__":
    main()
