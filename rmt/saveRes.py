
import numpy as np
import json

a = [[1, 2, 3], [4, 5.235412, 6], [7, 8.4875125, 9], [10, 11, 0.000000002]]
b = np.array(a)
res = b.tolist()

# # save result
# with open('res.json', 'w') as f:
#     json.dump(res, f)

# # read result
# with open("res.json", "r") as f:
#     resGet = f.readlines()

# # print
# print("resGet: ", resGet)

# # convert to numpy
# resNum = np.array(resGet[0])
# print("resNum: ", resNum, " resNum Shape: ", resNum.shape)
# # print("resNum[0]: ", resNum[0])
# # print("resNum[1]: ", resNum[1, :])
# # print("resNum[2]: ", resNum[2, :])

# numpy save
np.savetxt('res2.txt', b, fmt='%.10e')

c = np.loadtxt('res2.txt', dtype=np.float64)
print("c: ", c, " c Shape: ", c.shape)
