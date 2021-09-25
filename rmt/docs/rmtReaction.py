# REACTION RATE EXPRESSION
# -------------------------

# import packages/modules
import types
import math
from core import constants as CONST
import numpy as np


def reactionRateExe(loopVars, varDict, rateDict):
    """
    execute reaction rate expressions
    args:
        loopVars: main variables as: T, P, and MoFri, SpCoi (mol/m^3)
        varDict: defined variables by users
            _dict = {"key1": fun1, "key2": fun2, ...}
        rateDict: defined variables by users
            _dict = {"r1": fun1, "r2": fun2, ...}
    """
    # loop parameters
    T, P, MoFri, SpCoi = loopVars

    # loop dict
    loopDict = {
        "R_CONST": CONST.R_CONST,
        "T": T,
        "P": P,
        "MoFri": MoFri,
        "SpCoi": SpCoi
    }

    # create _dict
    _varDict = {}
    _varDict = {**loopDict, **varDict}

    # function calculate variables
    exeDict = {}

    for i in _varDict:
        # check function/scaler
        if isinstance(_varDict[i], types.FunctionType):
            _loop = _varDict[i](exeDict)
        else:
            _loop = _varDict[i]
        # add to the dict exec
        exeDict[i] = _loop

    # reaction rate list
    RiList = []

    for j in rateDict:
        _loop = rateDict[j](exeDict)
        RiList.append(_loop)

    # return
    return RiList


def componentFormationRate(compNo, comList, reactionStochCoeff, Ri):
    '''
    calculate component formation rate
        positive value for products
        negative value for reactants
    args: 
        compNo: number of components in reactions
            format: 
        comList: list of component name 
            format: 
                ["CH4","CO2",...]
        reactionStochCoeff: list of product and reactant stoch coefficients 
        Ri: formation rate [mol/m^3.s] | [kmol/m^3.s]
    output: 
        ri: component formation rate [mol/m^3.s], [kmol/m^3.s] (depend on Ri)
    '''
    # try/except
    try:
        # component formation rate [mol/m^3.s]
        # rf[mol/kgcat.s]*CaBeDe[kgcat/m^3]
        ri = np.zeros(compNo)
        for k in range(compNo):
            # reset
            _riLoop = 0
            for m in range(len(reactionStochCoeff)):
                for n in range(len(reactionStochCoeff[m])):
                    if comList[k] == reactionStochCoeff[m][n][0]:
                        _riLoop += reactionStochCoeff[m][n][1]*Ri[m]
                ri[k] = _riLoop

        # res
        return ri
    except Exception as e:
        raise
