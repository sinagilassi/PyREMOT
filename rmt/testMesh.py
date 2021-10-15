# import module//package
from solvers.solFiDi import FiDiMeshGenerator

# input
# number of nodes
NoNo = [20, 10]
# domain length
DoLe = 1
# domain length section
DoLeSe = 30
# mesh refinement degree
MeReDe = 2.5
# display
display = True

res = FiDiMeshGenerator(NoNo, DoLe, DoLeSe, MeReDe, display)
print("res: ", res)
