import numpy as np

a = np.array([1, 2, 3])
b = np.zeros(3)
print(f"a: {a}")
print(f"b: {b}")

b = np.copy(a)
print(f"b: {b}")

b[0] = 10
print(f"a: {a}")
print(f"b: {b}")
