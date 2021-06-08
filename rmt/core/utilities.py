# UTILITY FUNCTIONS
# ------------------

# import packages/modules
import numpy as np
from .config import ROUND_FUN_ACCURACY

# round a number, set decimal digit


def roundNum(value, ACCURACY=ROUND_FUN_ACCURACY):
    return np.round(value, ACCURACY)

 # remove duplicates


def removeDuplicatesList(value):
    print(value)
    return list(dict.fromkeys(value))
