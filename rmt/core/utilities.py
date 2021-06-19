# UTILITY FUNCTIONS
# ------------------

# import packages/modules
import numpy as np
from .config import ROUND_FUN_ACCURACY


def roundNum(value, ACCURACY=ROUND_FUN_ACCURACY):
    """
        round a number, set decimal digit
        accuracy default: 3 digits
    """
    return np.round(value, ACCURACY)


def removeDuplicatesList(value):
    """
        remove duplicates
    """
    print(value)
    return list(dict.fromkeys(value))
