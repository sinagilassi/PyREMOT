# RESULT ANALYSIS
# ----------------

# import packages/modules
import numpy as np
# internal
from docs.rmtUtility import rmtUtilityClass as rmtUtil
from docs.modelSetting import MODEL_SETTING, PROCESS_SETTING


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
