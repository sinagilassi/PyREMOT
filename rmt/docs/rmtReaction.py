# REACTION RATE EXPRESSION
# -------------------------

# import packages/modules
import math
from core import constants as CONST


def reactionRateExe(loopVars, params, varis, rates):
    """
    reaction rate expression
    """
    # loop parameters
    T, P, MoFri0, CaBeDe = loopVars

    # reaction rate
    rDict = {
        "PARAMS": params,
        "VARS": varis,
        "RATES": rates,
    }

    # function calculate variables
    varsDict = {}
    for i, j in rDict['VARS'].items():
        # exe var
        _loopValue = eval(j)
        # add to dic
        varsDict[i] = _loopValue
    # update value
    rDict.update({'VAL': varsDict})

    # variables
    # reactionVARS = rDict['VARS']
    # calculated variables
    # calculatedVARS = exeVars(T, P, MoFri)

    # # update rDict
    # rDict.update({"VARS": calculatedVARS})

    # calculate reaction rate expressions
    # reactionRates = rDict['RATES']
    # Ri value
    reactionRateValues = []
    # loop
    for x, y in rDict['RATES'].items():
        _loop = eval(y)
        # add
        reactionRateValues.append(_loop)

    # return
    return reactionRateValues
