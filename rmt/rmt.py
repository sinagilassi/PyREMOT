# REACTOR MODELING TOOLS IN PYTHON
# ---------------------------------

# import packages/modules
import timeit
import docs
from docs.rmtCore import rmtCoreClass


def main():
    """
        Reactor Modeling Tools in Python
    """
    pass


def rmtExe(modelInput):
    """
        This script check model input, then starts computation
    """

    # tic
    tic = timeit.timeit()

    # check input data
    # model type
    modelType = modelInput['model']
    print(f"model type: {modelType}")

    # operating conditions
    pressure = modelInput['operating-conditions']['pressure']
    print(f"pressure: {pressure}")
    temperature = modelInput['operating-conditions']['temperature']
    print(f"temperature: {temperature}")

    # feed
    FeMoFri = modelInput['feed']['mole-fraction']
    print(f"FeMoFri: {FeMoFri}")
    FeFlRa = modelInput['feed']['molar-flowrate']
    print(f"FeFlRa: {FeFlRa}")
    FeCom = modelInput['feed']['components']
    print(f"FeCom: {FeCom}")

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


if __name__ == "__main__":
    main()
