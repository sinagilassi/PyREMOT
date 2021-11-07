# UTILITY FUNCTIONS
# ------------------

# import packages/modules
import numpy as np
from PyREMOT.core.config import ROUND_FUN_ACCURACY


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
    # length
    indexLength = len(indices)
    if indexLength != 0:
        # select items
        selected_elements = [refList[index] for index in indices]
    else:
        selected_elements = refList
    return selected_elements


def selectRandomList(myList, no, labelNameTime):
    """
    select sorted random index from array
    args:
        myList: 2D array 
        no: number of element to be selected
        labelNameTime: label name time
    """
    # array shape
    myListShape = np.shape(myList)

    # set elements
    myListIndex = [*range(myListShape[0])]
    myListIndex_2 = myListIndex[1:-1]
    myListIndex_3 = np.sort(np.random.choice(myListIndex_2, no, replace=False))

    # select from array
    # element
    myList_2 = []
    # label
    labelNameTime_2 = []
    for i in myListIndex_3:
        myList_2.append(myList[i])
        # label
        labelNameTime_2.append(labelNameTime[i])

    a = myList[0]
    b = myList_2
    c = myList[-1]
    # combine
    myList_3 = [a, *b, c]

    # set label
    a1 = labelNameTime[0]
    b1 = labelNameTime_2
    c1 = labelNameTime[-1]
    labelNameTime_3 = [a1, *b1, c1]

    # res
    res = {
        "data1": myList_3,
        "data2": labelNameTime_3
    }
    return res


def selectRandomForList(myList, no):
    """
    select sorted random index from array
    args:
        myList: 2D array 
        no: number of element to be selected (scaler)
    """
    # array shape
    myListShape = np.shape(myList)

    # set elements
    myListIndex = [*range(myListShape[0])]
    myListIndex_2 = myListIndex[1:-1]
    myListIndex_3 = np.sort(np.random.choice(myListIndex_2, no, replace=False))
    # set
    a = myList[0]
    b = myListIndex_3
    c = myList[-1]
    # combine
    myList_2 = [a, *b, c]
    # res
    return myList_2
