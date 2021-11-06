# CHECK COMPONENT LIST
# ---------------------

# import packages/modules
import numpy as np
import math
import json
from data import *
from core import constants as CONST
from rmt import rmtExe, rmtCom
from core.utilities import roundNum
from docs.rmtUtility import rmtUtilityClass as rmtUtil

# NOTE
# display component list
res = rmtCom()
print(res)
