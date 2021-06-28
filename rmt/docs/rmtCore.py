# THE MAIN COMPUTATION SCRIPT
# ----------------------------

# import packages/modules
from pprint import pprint
from data.inputDataReactor import *
from core.setting import modelTypes, M1, M2
# from docs.pbReactor import runM1
from docs.cReactor import conventionalReactorClass as cRec
from docs.pbReactor import PackedBedReactorClass as pbRec
from data.componentData import componentDataStore
from .rmtUtility import rmtUtilityClass as rmtUtil
from .rmtThermo import calEnthalpyChangeOfReaction
from .rmtReaction import rmtReactionClass as rmtRec


class rmtCoreClass(pbRec, cRec):
    """
        script for different modeling modes
    """

    # ode time span
    t0 = 0
    tn = rea_L

    # init
    def __init__(self, modelMode, modelInput):
        self.modelMode = modelMode
        self.modelInput = modelInput

        # component list
        compList = self.modelInput['feed']['components']['shell']
        # reaction list
        reactionList = self.modelInput['reactions']
        # reaction rate list
        reactionRateList = self.modelInput['reaction-rates']

        # init database
        internalDataSet = self.initComponentData(compList)
        # print(internalDataSet)

        # reaction list sorted
        reactionListSortedSet = self.initReaction(reactionList)
        # print(reactionCoeffSet)

        # reaction rate expression
        reactionRateExpressionSet = self.initReactionRate(reactionRateList)
        # test fun
        # ans1 = reactionRateExpressionSet[0].reactionRateFunSet(T=1, P=2, y=3)
        # print("ans1: ", ans1)

        # pbRec
        pbRec.__init__(self, modelInput, internalDataSet,
                       reactionListSortedSet)

    def modExe(self):
        """
            select modeling script based on model type
        """
        # select model type
        modelMode = self.modelMode
        if modelMode == M1:
            return self.M1Init()
        elif modelMode == M2:
            return self.M2Init()

    def M1Init(self):
        """
            M1 model
            more info, check --help M1
        """
        # class init
        # modelInput = self.modelInput

        # start cal
        res = self.runM1()
        return res

    def M2Init(self):
        """
            M1 model
            more info, check --help M1
        """
        # class init
        # modelInput = self.modelInput

        # start cal
        res = self.runM2()
        return res

    def initComponentData(self, compList):
        """
            initialize component data as:
                heat capacity at constant pressure 
                heat of formation at 25C
        """
        # try/except
        try:
            # app data
            appData = componentDataStore['payload']

            # component data
            compData = []

            # init library
            for i in compList:
                _loop1 = [
                    item for item in appData if i == item['symbol']]
                compData.append(_loop1)

            # res
            return compData
        except Exception as e:
            raise

    def initReaction(self, reactionDict):
        """
            initialize reaction list to find stoichiometric coefficient
        """
        # try/except
        try:
            # reaction list sorted
            reactionListSorted = rmtUtil.buildReactionCoefficient(reactionDict)
            # print(reactionListSorted)
            # reaction stoichiometric coefficient vector
            reactionStochCoeff = rmtUtil.buildReactionCoeffVector(
                reactionListSorted)

            # reaction rate expression list

            # res
            return reactionListSorted
        except Exception as e:
            raise

    def initReactionRate(self, reactionRateDict):
        """
            initialize reaction rate expr list 
        """
        # try/except
        try:
            # reaction rate expr no
            reactionRateExprNo = list(reactionRateDict.keys())
            # reaction rate expr list
            reactionRateExprList = []

            # reaction rate expression list
            for i in reactionRateDict:
                _loop = rmtRec(reactionRateDict[i])
                # print("reaction rate expr: ",
                #       _loop.reactionRateFunSet(T=1, P=2, y=3))
                # add to list
                reactionRateExprList.append(_loop)

            # res
            return reactionRateExprList
        except Exception as e:
            raise
