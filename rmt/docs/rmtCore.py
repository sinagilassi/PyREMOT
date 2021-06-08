# THE MAIN COMPUTATION SCRIPT
# ----------------------------

# import packages/modules
from data.inputDataReactor import *
from core.setting import modelTypes, M1
from docs.pbReactor import runM1
from docs.cReactor import conventionalReactorClass


class rmtCoreClass:
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

    def modExe(self):
        """
            select modeling script based on model type
        """
        # select model type
        modelMode = self.modelMode
        if modelMode == M1:
            return self.M1Init()

    def M1Init(self):
        """
            M1 model
            more info, check --help M1
        """
        # class init
        modelInput = self.modelInput
        pbRec = runM1(modelInput)
        # start cal
        res = pbRec.runM1()
        return res
