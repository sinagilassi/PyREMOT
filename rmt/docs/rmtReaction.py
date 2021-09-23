# REACTION RATE EXPRESSION
# -------------------------

# import packages/modules
import types
import math
from core import constants as CONST


def reactionRateExe(loopVars, varDict, rateDict):
    """
    execute reaction rate expressions
    args:
        loopVars: main variables as: T, P, and MoFri
        varDict: defined variables by users
            _dict = {"key1": fun1, "key2": fun2, ...}
        rateDict: defined variables by users
            _dict = {"r1": fun1, "r2": fun2, ...}
    """
    # loop parameters
    T, P, MoFri = loopVars

    # loop dict
    loopDict = {
        "R_CONST": CONST.R_CONST,
        "T": T,
        "P": P,
        "MoFri": MoFri
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
