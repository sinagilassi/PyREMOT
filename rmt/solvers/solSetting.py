# SOLVER SETTING
# ---------------

# define solver setting
# number of finite difference points along the reactor length [zNo]
# number of orthogonal collocation points inside the catalyst particle [rNo]

# S1
# heterogenous model

# S2
# homogenous dynamic model for plug-flow reactors

# S3
# homogenous steady-state model for plug-flow reactors


solverSetting = {
    "S1": {
        "zNo": 20,
        "rNo": 5
    },
    "S2": {
        "tNo": 5,
        "zNo": 10,
        "timesNo": 10
    },
    "S3": {
        "zNo": 10,
        "timesNo": 10
    }
}
