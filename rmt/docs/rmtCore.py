# THE MAIN COMPUTATION SCRIPT
# ----------------------------

# import packages/modules
from data.inputDataReactor import *
from core.setting import M1, M2, M3, M4, M5, M6, M7, M8
# from docs.pbReactor import runM1
from docs.cReactor import conventionalReactorClass as cRec
from docs.pbReactor import PackedBedReactorClass as pbRec
from docs.batchReactor import batchReactorClass as bRec
from docs.pfReactor import PlugFlowReactorClass as pfRec
from docs.pbHeterReactor import PackedBedHeteroReactorClass as pbHeterRec
from data.componentData import componentDataStore
from .rmtUtility import rmtUtilityClass as rmtUtil
from .rmtReaction import rmtReactionClass as rmtRec


class rmtCoreClass():
    """
        script for different modeling modes
    """

    # ode time span
    t0 = 0
    tn = rea_L

    # global vars

    def __init__(self, modelMode, modelInput):
        self.modelMode = modelMode
        self.modelInput = modelInput

        # bRec.__init__(self, modelInput, internalDataSet,
        #               reactionListSortedSet)

    # property list
    # @property
    # def internalDataSet(self):
    #     return self._internalDataSet

    # @internalDataSet.setter
    # def internalDataSet(self, value):
    #     self._internalDataSet = value

    def gVarCal(self, compList, reactionList):
        """
        init global var
        """
        # init database
        internalDataRes = self.initComponentData(compList)

        # reaction list sorted
        initReactionRes = self.initReaction(reactionList)
        # reactionListSortedRes = initReactionRes['res1']
        # reactionStochCoeffListRes = initReactionRes['res2']
        # res
        return [internalDataRes, initReactionRes]

    def modExe(self):
        """
            select modeling script based on model type
        """
        # component list
        compList = self.modelInput['feed']['components']['shell']
        # reaction list
        reactionList = self.modelInput['reactions']
        # reaction rate list
        reactionRateList = self.modelInput['reaction-rates']

        # set data
        # init globals vars
        gVarRes = self.gVarCal(compList, reactionList)
        # set res
        # init database
        _internalDataSet = gVarRes[0]
        # print(internalDataSet)

        # reaction list sorted
        _reactionListSortedSet = gVarRes[1]['res1']
        _reactionStochCoeffListSet = gVarRes[1]['res2']
        # print(reactionCoeffSet)

        # select model type
        modelMode = self.modelMode
        if modelMode == M1:
            return self.M1Init(_internalDataSet, _reactionListSortedSet, _reactionStochCoeffListSet)
        elif modelMode == M2:
            return self.M2Init()
        elif modelMode == M3:
            return self.M3Init()
        elif modelMode == M4:
            return self.M4Init(_internalDataSet, _reactionListSortedSet, _reactionStochCoeffListSet)
        elif modelMode == M5:
            return self.M5Init(_internalDataSet, _reactionListSortedSet, _reactionStochCoeffListSet)
        elif modelMode == M6:
            return self.M6Init(_internalDataSet, _reactionListSortedSet, _reactionStochCoeffListSet)
        elif modelMode == M7:
            return self.M7Init(_internalDataSet, _reactionListSortedSet, _reactionStochCoeffListSet)
        elif modelMode == M8:
            return self.M8Init(_internalDataSet, _reactionListSortedSet, _reactionStochCoeffListSet)

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
            # component data index
            compDataIndex = []

            # init library
            for i in compList:
                _loop1 = [
                    j for j, item in enumerate(appData) if i in item.values()]
                compDataIndex.append(_loop1[0])

            for i in compDataIndex:
                compData.append(appData[i])

            # old version
            # init library
            # for i in compList:
            #     _loop1 = [
            #         item for item in appData if i == item['symbol']]
            #     compData.append(_loop1[0])

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
            # print("reactionListSorted: ", reactionListSorted)
            # reaction stoichiometric coefficient vector
            reactionStochCoeff = rmtUtil.buildReactionCoeffVector(
                reactionListSorted)
            # print("reactionStochCoeff: ", reactionStochCoeff)

            # res
            return {"res1": reactionListSorted, "res2": reactionStochCoeff}
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
                if i != "VAR":
                    _loop = rmtRec(reactionRateDict[i])
                    # print("reaction rate expr: ",
                    #       _loop.reactionRateFunSet(T=1, P=2, y=3))
                    # add to list
                else:
                    _loop = reactionRateDict[i]

                reactionRateExprList.append(_loop)

            # res
            return reactionRateExprList
        except Exception as e:
            raise

    def M1Init(self, internalData, reactionListSorted, reactionStochCoeffList):
        """
        M1 model: Packed-bed Plug-flow reactor
        """
        # init PBPR
        pbRecInit = pbRec(self.modelInput, internalData,
                          reactionListSorted, reactionStochCoeffList)
        # run algorithm
        res = pbRecInit.runM1()
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

    def M3Init(self):
        """
        M3 Model: Batch Reactor
        """

    def M4Init(self, internalData, reactionListSorted, reactionStochCoeffList):
        """
        M4 Model: Plug-flow Reactor
        """
        # init plug-flow reactor
        pfRecInit = pfRec(self.modelInput, internalData,
                          reactionListSorted, reactionStochCoeffList)
        # run algorithm
        res = pfRecInit.runM1()
        # result
        return res

    def M5Init(self, internalData, reactionListSorted, reactionStochCoeffList):
        """
        M1 model: Packed-bed Plug-flow reactor (heterogenous)
        """
        # init PBPR
        pbHeterRecInit = pbHeterRec(self.modelInput, internalData,
                                    reactionListSorted, reactionStochCoeffList)
        # run algorithm
        res = pbHeterRecInit.runM1()
        return res

    def M6Init(self, internalData, reactionListSorted, reactionStochCoeffList):
        """
        M6 model: dynamic Packed-bed Plug-flow reactor (homogenous)
        """
        # init reactor
        reInit = pbRec(self.modelInput, internalData,
                       reactionListSorted, reactionStochCoeffList)
        # run algorithm
        res = reInit.runM2()
        return res

    def M7Init(self, internalData, reactionListSorted, reactionStochCoeffList):
        """
        M6 model: steady-state Packed-bed Plug-flow reactor (homogenous)
        """
        # init reactor
        reInit = pbRec(self.modelInput, internalData,
                       reactionListSorted, reactionStochCoeffList)
        # run algorithm
        res = reInit.runM3()
        return res

    def M8Init(self, internalData, reactionListSorted, reactionStochCoeffList):
        """
        M6 model: steady-state Packed-bed Plug-flow reactor (homogenous)
        """
        # init reactor
        reInit = pbRec(self.modelInput, internalData,
                       reactionListSorted, reactionStochCoeffList)
        # run algorithm
        res = reInit.runM4()
        return res
