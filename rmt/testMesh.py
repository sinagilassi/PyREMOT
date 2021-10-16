# import module//package
from solvers.solFiDi import FiDiMeshGenerator
from solvers.solSetting import solverSetting

# input
# domain length
DoLe = 1
# mesh setting
zMesh = solverSetting['T1']['zMesh']
# number of nodes
NoNo = zMesh['zNoNo']
# domain length section
DoLeSe = zMesh['DoLeSe']
# mesh refinement degree
MeReDe = zMesh['MeReDe']

# display
display = True

res = FiDiMeshGenerator(NoNo, DoLe, DoLeSe, MeReDe, display)
# print("res: ", res)
