# CORE CLASS
# define a differential equation
# -------------------------------

# import packages/modules
import numpy as np


class solCoreClass:
    """
        define differential equation as:
            A[d2y/dx2] + B[dy/dx] + C[y] + D = 0

    """

    def __init__(self, A, B, C, D, dx=1.0E-5):
        """
            initialize class
            dx: space between two nodes, sometimes represented by dx
        """
        print("solCoreClass is init")
        self.A, self.B, self.C, self.D = A, B, C, D
        self.dx = float(dx)

    def __call__(self, x):
        """
            build approximate function based on finite difference
        """
        print("[solCoreClass] is called!")
        return 10 + x**2

    def fp(self, x):
        """
            build first derivative based on central difference method
            fp = df/dx = [f(x+h) - f(x-h)]/[2*dx]
        """
        dx = self.dx
        dfdx = (self(x+dx) - self(x-dx))/(2.0*dx)
        return dfdx

    def fpp(self, x):
        """
            build second derivative based on central difference method
            fp = df/dx = [f(x+h) - 2*self(x) + f(x-h)]/[dx^2]
        """
        dx = self.dx
        d2fdx2 = (self(x+dx) - 2.0*self(x) + self(x-dx))/(dx**2)
        return d2fdx2


class solTest(solCoreClass):
    def __init__(self, A, B, C, D, dx=1e-5):
        print("solTest is init")
        super().__init__(A, B, C, D, dx=dx)

    def __call__(self, x):
        print("[solTest] is called!")
        return x + 200


test = solTest(1, 2, 3, 4)
print(test)
x = 1
val = test(x)
print(val)
df = test.fp(x)
print(df)
#
print("-----------")
test2 = solCoreClass(1, 1, 1, 1)
print(test2)
x = 1
val2 = test2(x)
print(val2)
df2 = test2.fp(x)
print(df2)
