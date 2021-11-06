# REACTOR MODELING TOOLS IN PYTHON
# ---------------------------------

# import packages/modules
import time
import timeit
from docs.rmtCore import rmtCoreClass
from data.componentData import componentSymbolList
from docs.rmtUtility import rmtUtilityClass as rmtUtil


def main():
    """
    Python Reactor Modeling Tools (PyREMOT) 
    """
    pass


def rmtExe(modelInput):
    """
    This script check model input, then starts computation
    """
    # try/exception
    try:
        # tic
        tic = timeit.timeit()

        # check input data
        # model type
        modelType = modelInput['model']
        # print(f"model type: {modelType}")

        # operating conditions
        pressure = modelInput['operating-conditions']['pressure']
        # print(f"pressure: {pressure}")
        temperature = modelInput['operating-conditions']['temperature']
        # print(f"temperature: {temperature}")

        # feed
        FeMoFri = modelInput['feed']['mole-fraction']
        # print(f"FeMoFri: {FeMoFri}")
        FeFlRa = modelInput['feed']['molar-flowrate']
        # print(f"FeFlRa: {FeFlRa}")
        MoFl = modelInput['feed']['molar-flux']
        # print(f"MoFl: {MoFl}")
        FeCom = modelInput['feed']['components']
        # print(f"FeCom: {FeCom}")

        # get all component
        compList = rmtUtil.buildComponentList(FeCom)

        # check component data availability
        for i in range(len(compList)):
            if compList[i] not in componentSymbolList:
                raise Exception("Component database is not up to date!")
        # checkCompData =

        # init class
        rmtCore = rmtCoreClass(modelType, modelInput)

        # init computation
        resModel = rmtCore.modExe()

        # tac
        tac = timeit.timeit()

        # computation time [s]
        comTime = (tac - tic)*1000

        # result
        res = {
            "resModel": resModel,
            "comTime": comTime
        }
        return res
    except Exception as e:
        print(e)
        raise


def rmtCom():
    '''
    display components available in the current version 
    '''
    try:
        # print(componentSymbolList)
        # name list
        compListName = ','
        return compListName.join(componentSymbolList)
    except Exception as e:
        raise


if __name__ == "__main__":
    main()
