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


def selectFromListByIndex(indices, refList):
    '''
    extract selected element from a list
    args:
        indices: list of index
        refList: 1D array
    '''
    # select items
    selected_elements = [refList[index] for index in indices]
    return selected_elements
