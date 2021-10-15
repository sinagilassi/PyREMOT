# SOLVER SETTING
# ---------------

# define solver setting
# number of finite difference points along the reactor length [zNo]
# number of orthogonal collocation points inside the catalyst particle [rNo]

# S1
# heterogenous model

# S2
# homogenous dynamic model for plug-flow reactors
# timesNo: in each loop (time)

# S3
# homogenous steady-state model for plug-flow reactors
# NOTE
# timeNo = zNo of S2 for comparison

# S4

DIFF_SETTING = {
    "BD": -1,
    "CD": 0,
    "FD": 1
}

solverSetting = {
    "S1": {
        "zNo": 20,
        "rNo": 5
    },
    "S2": {
        "tNo": 10,
        "zNo": 100,
        "rNo": 5,
        "timesNo": 5
    },
    "S3": {
        "timesNo": 25
    },
    "T1": {
        "zMesh": {
            "zNoNo": [20, 10],
            "DoLeSe": 30,
            "MeReDe": 2.5
        },
        "zNo": 30,
        "rNo": 0,
        "dFdz": DIFF_SETTING['BD'],
        "d2Fdz2": {
            "BC1": DIFF_SETTING['CD'],
            "BC2": DIFF_SETTING['CD'],
            "G": DIFF_SETTING['BD']
        },
        "dTdz": DIFF_SETTING['BD'],
        "d2Tdz2": {
            "BC1": DIFF_SETTING['CD'],
            "BC2": DIFF_SETTING['CD'],
            "G": DIFF_SETTING['BD']
        },
    },
}
