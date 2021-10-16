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

# T2
# backward diffrentate for dF/dz
# central diffrentate for d2F/dz2

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
        "rNo": 10,
        "timesNo": 5
    },
    "S3": {
        "timesNo": 25
    },
    "T1": {
        "zMesh": {
            "zNoNo": [100, 3],
            "DoLeSe": 95,
            "MeReDe": 1.3
        },
        "zNo": 30,
        "rNo": 0,
        "dFdz": DIFF_SETTING['BD'],
        "d2Fdz2": {
            "BC1": DIFF_SETTING['CD'],
            "BC2": DIFF_SETTING['CD'],
            "G": DIFF_SETTING['CD']
        },
        "dTdz": DIFF_SETTING['BD'],
        "d2Tdz2": {
            "BC1": DIFF_SETTING['CD'],
            "BC2": DIFF_SETTING['CD'],
            "G": DIFF_SETTING['CD']
        },
    },
}
