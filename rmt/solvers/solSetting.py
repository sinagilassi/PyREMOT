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


solverSetting = {
    "S1": {
        "zNo": 20,
        "rNo": 5
    },
    "S2": {
        "tNo": 50,
        "zNo": 4,
        "rNo": 4,
        "timesNo": 5
    },
    "S3": {
        "timesNo": 25
    }
}
