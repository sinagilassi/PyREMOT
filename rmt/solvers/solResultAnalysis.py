# RESULT ANALYSIS
# ----------------

# import packages/modules
import numpy as np
# internal
from docs.rmtUtility import rmtUtilityClass as rmtUtil
from library.plot import plotClass as pltc
from docs.modelSetting import MODEL_SETTING, PROCESS_SETTING
from core.utilities import roundNum, selectFromListByIndex, selectRandomForList


def setOptimizeRootMethod(y, params1, params2, param3=0):
    """
    set results of optimize.root function
    args:
        params1:
            compNo: component number
            noLayer: number of var layers
            varNoRows: 1
            varNoColumns: number of finite nodes in the z direction
        params2: 
            Cif: species concentration of feed gas
            Tf: feed temperature
    """
    # distribute y[i] value through the reactor length
    #  try/except
    try:
        compNo, noLayer, varNoRows, varNoColumns = params1
        Cif, Tf, processType = params2

        # concentration
        dataYs_Concentration_DiLeVa = y[:-
                                        1] if processType != PROCESS_SETTING['ISO-THER'] else y[:]
        # temperature
        dataYs_Temperature_DiLeVa = y[-1] if processType != PROCESS_SETTING['ISO-THER'] else np.repeat(
            param3, varNoColumns).reshape((varNoRows, varNoColumns))

        # convert to real value
        # concentration
        SpCo_mz_ReVa = np.zeros((compNo, varNoColumns))
        T_mz_ReVa = np.zeros((1, varNoColumns))
        # concentration
        for i in range(compNo):
            # dimensionless analysis: real value
            Cif_Set = Cif[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
                Cif)
            SpCo_mz_ReVa[i, :] = rmtUtil.calRealDiLessValue(
                dataYs_Concentration_DiLeVa[i, :], Cif_Set)
        # temperature
        T_mz_ReVa[0, :] = rmtUtil.calRealDiLessValue(
            dataYs_Temperature_DiLeVa[0, :], Tf, mode="TEMP")

        # result
        res = {
            "data1": SpCo_mz_ReVa,
            "data2": T_mz_ReVa
        }
        # return
        return res
    except Exception as e:
        raise


def sortResult2(y, params1, params2):
    """
    sort result of modeling of particle diffusion-reaction 
    args:
        params1:
            compNo: component number
            noLayer: number of var layers
            varNoRows: 1
            varNoColumns: number of finite nodes in the r direction
        params2: 
            Cif: species concentration of feed gas
            Tf: feed temperature
    """
    # distribute y[i] value through the reactor length
    #  try/except
    try:
        compNo, noLayer, varNoRows, varNoColumns = params1
        Cif, Tf, processType = params2

        # concentration
        dataYs_Concentration_DiLeVa = y[:-
                                        1] if processType != PROCESS_SETTING['ISO-THER'] else y[:]
        # temperature
        dataYs_Temperature_DiLeVa = y[-1, :].reshape((1, varNoColumns)) if processType != PROCESS_SETTING['ISO-THER'] else np.repeat(
            0, varNoColumns).reshape((1, varNoColumns))

        # convert to real value
        # concentration
        SpCo_mz_ReVa = np.zeros((compNo, varNoColumns))
        T_mz_ReVa = np.zeros((1, varNoColumns))
        # concentration
        for i in range(compNo):
            # dimensionless analysis: real value
            Cif_Set = Cif[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
                Cif)
            SpCo_mz_ReVa[i, :] = rmtUtil.calRealDiLessValue(
                dataYs_Concentration_DiLeVa[i, :], Cif_Set)
        # temperature
        T_mz_ReVa[0, :] = rmtUtil.calRealDiLessValue(
            dataYs_Temperature_DiLeVa[0, :], Tf, mode="TEMP")

        # result
        res = {
            "data1": SpCo_mz_ReVa,
            "data2": T_mz_ReVa
        }
        # return
        return res
    except Exception as e:
        raise

# NOTE


def sortedResult3(yC_DiLeVa, yT_DiLeVa, yCs_DiLeVa, yTs_DiLeVa, params1, params2):
    """
    sort result of heterogenous modeling
    args:
        params1:
            compNo: component number
            noLayer: number of var layers
            varNoRows: 1
            varNoColumns: number of finite nodes in the z direction
            rNo: number of finite nodes in the r directions
            zNo: the same as varNoColumns
        params2: 
            Cif: species concentration of feed gas
            Tf: feed temperature
    """
    # distribute y[i] value through the reactor length
    #  try/except
    try:
        compNo, noLayer, varNoRows, varNoColumns, rNo, zNo = params1
        Cif, Tf, processType = params2

        # convert to real value
        # gas phase concentration/temperature
        SpCo_mz_ReVa = np.zeros((compNo, varNoColumns))
        T_mz_ReVa = np.zeros((1, varNoColumns))
        # solid phase concentration/temperature
        # all species concentration in solid phase (catalyst)
        SpCosi_mzr_ReVa = np.zeros((compNo, rNo, zNo))
        Ts_mzr_ReVa = np.zeros((rNo, zNo))

        # NOTE
        ### gas phases ###
        # concentration
        for i in range(compNo):
            # dimensionless analysis: real value
            Cif_Set = Cif[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
                Cif)
            SpCo_mz_ReVa[i, :] = rmtUtil.calRealDiLessValue(
                yC_DiLeVa[i, :], Cif_Set)
        # temperature
        T_mz_ReVa[0, :] = rmtUtil.calRealDiLessValue(
            yT_DiLeVa[0, :], Tf, mode="TEMP")

        # NOTE
        ### solid phases ###
        # concentration
        for i in range(compNo):
            # dimensionless analysis: real value
            Cif_Set = Cif[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
                Cif)
            SpCosi_mzr_ReVa[i] = rmtUtil.calRealDiLessValue(
                yCs_DiLeVa[i, :], Cif_Set)
        # temperature
        Ts_mzr_ReVa[0, :] = rmtUtil.calRealDiLessValue(
            yTs_DiLeVa[0:, :], Tf, mode="TEMP")

        # result
        res = {
            "data1": SpCo_mz_ReVa,
            "data2": T_mz_ReVa,
            "data3": SpCosi_mzr_ReVa,
            "data4": Ts_mzr_ReVa
        }
        # return
        return res
    except Exception as e:
        raise

# NOTE
# homogenous modeling results


def sortResult4(y, params1, params2):
    """
    sort result of homogenous modeling 
    args:
        params1:
            compNo: component number
            noLayer: number of var layers
            varNoRows: 1
            varNoColumns: number of finite nodes in the z direction
        params2: 
            Cif: species concentration of feed gas
            Tf: feed temperature
    """
    # distribute y[i] value through the reactor length
    #  try/except
    try:
        compNo, varNoRows, varNoColumns = params1
        Cif, Tf, Pf, processType = params2

        # concentration
        dataYs_Concentration_DiLeVa = y[:-
                                        2] if processType != PROCESS_SETTING['ISO-THER'] else y[:-1]
        # pressure
        dataYs_Pressure_DiLeVa = y[-2].reshape(
            (1, varNoColumns)) if processType != PROCESS_SETTING['ISO-THER'] else y[-1].reshape((1, varNoColumns))
        # temperature
        dataYs_Temperature_DiLeVa = y[-1].reshape((1, varNoColumns)) if processType != PROCESS_SETTING['ISO-THER'] else np.repeat(
            0, varNoColumns).reshape((1, varNoColumns))

        # convert to real value
        # concentration
        SpCo_mz_ReVa = np.zeros((compNo, varNoColumns))
        T_mz_ReVa = np.zeros((1, varNoColumns))
        P_mz_ReVa = np.zeros((1, varNoColumns))
        # concentration
        for i in range(compNo):
            # dimensionless analysis: real value
            Cif_Set = Cif[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
                Cif)
            SpCo_mz_ReVa[i, :] = rmtUtil.calRealDiLessValue(
                dataYs_Concentration_DiLeVa[i, :], Cif_Set)

        # pressure
        P_mz_ReVa[0, :] = rmtUtil.calRealDiLessValue(
            dataYs_Pressure_DiLeVa[0, :], Pf)
        # temperature
        T_mz_ReVa[0, :] = rmtUtil.calRealDiLessValue(
            dataYs_Temperature_DiLeVa[0, :], Tf, mode="TEMP")

        # result
        res = {
            "data1": SpCo_mz_ReVa,
            "data2": P_mz_ReVa,
            "data3": T_mz_ReVa
        }
        # return
        return res
    except Exception as e:
        raise


def sortResult5(y, params1, params2):
    """
    sort result of homogenous modeling 
    args:
        params1:
            compNo: component number
            noLayer: number of var layers
            varNoRows: 1
            varNoColumns: number of finite nodes in the r direction
        params2: 
            Cif: species concentration of feed gas
            Tf: feed temperature
    """
    # distribute y[i] value through the reactor length
    #  try/except
    try:
        compNo, varNoRows, varNoColumns = params1
        Cif, Tf, processType = params2

        # concentration
        dataYs_Concentration_DiLeVa = y[:-
                                        1] if processType != PROCESS_SETTING['ISO-THER'] else y[:]
        # temperature
        dataYs_Temperature_DiLeVa = y[-1, :].reshape((1, varNoColumns)) if processType != PROCESS_SETTING['ISO-THER'] else np.repeat(
            0, varNoColumns).reshape((1, varNoColumns))

        # convert to real value
        # concentration
        SpCo_mz_ReVa = np.zeros((compNo, varNoColumns))
        T_mz_ReVa = np.zeros((1, varNoColumns))
        # concentration
        for i in range(compNo):
            # dimensionless analysis: real value
            Cif_Set = Cif[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
                Cif)
            SpCo_mz_ReVa[i, :] = rmtUtil.calRealDiLessValue(
                dataYs_Concentration_DiLeVa[i, :], Cif_Set)
        # temperature
        T_mz_ReVa[0, :] = rmtUtil.calRealDiLessValue(
            dataYs_Temperature_DiLeVa[0, :], Tf, mode="TEMP")

        # result
        res = {
            "data1": SpCo_mz_ReVa,
            "data2": T_mz_ReVa
        }
        # return
        return res
    except Exception as e:
        raise

# NOTE
# plot results


def plotResultsSteadyState(dataPack):
    '''
    plot results
    args:
        dataPack:
            modelId: model id,
            processType: process type,
            successStatus: ode success status,
            computation-time: elapsed time [s],
            dataShape: dataShape,
            labelList: labelList,
            indexList: indexList,
            dataTime: [],
            dataXs: dataXs,
            dataYCons1: dataYs_Concentration_DiLeVa,
            dataYCons2: dataYs_Concentration_ReVa,
            dataYTemp1: dataYs_Temperature_DiLeVa,
            dataYTemp2: dataYs_Temperature_ReVa,
            dataYs: dataYs_All
    '''
    # try/except
    try:
        # model info
        modelId = dataPack[0]['modelId']
        processType = dataPack[0]['processType']
        # calculation status
        successStatus = dataPack[0]['successStatus']
        # data
        dataXs = dataPack[0]['dataXs']
        dataYs_All = dataPack[0]['dataYs']
        labelList = dataPack[0]['labelList']
        indexList = dataPack[0]['indexList']
        elapsed = dataPack[0]['computation-time']
        # set
        plotTitle = f"Steady-State Modeling {modelId}, computation-time {elapsed}"
        xLabelSet = "Reactor Length (m)"
        yLabelSet = "Concentration (mol/m^3)"
        compNo = indexList[0]
        indexPressure = indexList[1]
        indexTemp = indexList[2]

        # check
        if successStatus is True:
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataXs, dataYs_All)
            # -> add label
            dataList = pltc.plots2DSetDataList(XYList, labelList)
            # datalists
            dataLists = [dataList[0:compNo], dataList[indexPressure], dataList[indexTemp]
                         ] if processType != PROCESS_SETTING['ISO-THER'] else [dataList[0:compNo], dataList[indexPressure]]
            # select datalist
            _dataListsSelected = selectFromListByIndex([], dataLists)
            # subplot result
            pltc.plots2DSub(_dataListsSelected, xLabelSet,
                            yLabelSet, plotTitle)
            pass
        else:
            dataPack = []
    except Exception as e:
        raise


def plotResultsDynamic(resPack, tNo):
    '''
    plot results
    args:
        resPack: 
            computation-time: elapsed time [s]
            dataPack:
                modelId: model id,
                processType: process type,
                successStatus: ode success status,
                dataShape: dataShape,
                labelList: labelList,
                indexList: indexList,
                dataTime: [],
                dataXs: dataXs,
                dataYCons1: dataYs_Concentration_DiLeVa,
                dataYCons2: dataYs_Concentration_ReVa,
                dataYTemp1: dataYs_Temperature_DiLeVa,
                dataYTemp2: dataYs_Temperature_ReVa,
                dataYs: dataYs_All

    '''
    # try/except
    try:
        # get
        elapsed = resPack['computation-time']
        dataPack = resPack['dataPack']

        # model info
        modelId = dataPack[0]['modelId']
        processType = dataPack[0]['processType']
        # calculation status
        successStatus = dataPack[0]['successStatus']
        # data
        dataXs = dataPack[0]['dataXs']
        dataYs_All = dataPack[0]['dataYs']
        labelList = dataPack[0]['labelList']
        indexList = dataPack[0]['indexList']

        # set
        plotTitle = f"Steady-State Modeling {modelId}, computation-time {elapsed}"
        xLabelSet = "Reactor Length (m)"
        yLabelSet = "Concentration (mol/m^3)"
        compNo = indexList[0]
        indexPressure = indexList[1]
        indexTemp = indexList[2]

        # random tNo
        tNoList = list(range(tNo))
        tNoRandomList = selectRandomForList(tNoList, 2)

        # REVIEW
        # display result at specific time
        for i in tNoRandomList:

            # calculation status
            successStatus = dataPack[i]['successStatus']

            # check
            if successStatus is True:
                # data
                dataXs = dataPack[i]['dataXs']
                dataYs_All = dataPack[i]['dataYs']
                labelList = dataPack[i]['labelList']
                indexList = dataPack[i]['indexList']

                # plot setting: build (x,y) series
                XYList = pltc.plots2DSetXYList(dataXs, dataYs_All)
                # -> add label
                dataList = pltc.plots2DSetDataList(XYList, labelList)
                # datalists
                dataLists = [dataList[0:compNo], dataList[indexTemp]
                             ] if processType != PROCESS_SETTING['ISO-THER'] else [dataList[0:compNo]]
                # select datalist
                _dataListsSelected = selectFromListByIndex([], dataLists)
                # subplot result
                pltc.plots2DSub(_dataListsSelected, xLabelSet,
                                yLabelSet, plotTitle)
            else:
                dataPack = []
    except Exception as e:
        raise
