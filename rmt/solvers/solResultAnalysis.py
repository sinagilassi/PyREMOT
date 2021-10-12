# RESULT ANALYSIS
# ----------------

# import packages/modules
import numpy as np
# internal
from docs.rmtUtility import rmtUtilityClass as rmtUtil
from docs.modelSetting import MODEL_SETTING, PROCESS_SETTING


def setOptimizeRootMethod(y, params1, params2):
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
            0, varNoColumns).reshape((varNoRows, varNoColumns))

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
