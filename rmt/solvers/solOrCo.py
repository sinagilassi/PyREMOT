# ORTHOGONAL COLLOCATION METHOD
# -------------------------------

# import package/module
import numpy as np


class OrCoClass:
    # class vars

    # constants
    # Approximate Solution
    # ---------------------
    # y = d1 + d2*x^2 + d3*x^4 + d4*x^6 + ... + d[N+1]*x^2N

    # Define Collocation Points
    # -------------------------
    # x1,x2,x3,x4
    # x1 = 0
    # x1 = 0.28523
    # x2 = 0.76505
    # x3 = 1

    # 6 points [spherical shape]
    # x1 = 0
    x1 = 0.215353
    x2 = 0.420638
    x3 = 0.606253
    x4 = 0.763519
    x5 = 0.885082
    x6 = 0.965245
    x7 = 1

    # initial boundary condition
    X0 = 0
    # last boundary condition
    Xn = 1

    # collocation points
    # Xc = np.array([x1, x2, x3])
    # 6 points
    Xc = np.array([x1, x2, x3, x4, x5, x6, x7])
    # 5 points
    # Xc = np.array([x1, x2, x3, x4, x5, x6])

    # collocation + boundary condition points
    # 4 points [symmetric 3 points]
    N = np.size(Xc)
    # collocation points number
    Nc = N - 1

    def __init__(self, odeNo):
        self.odeNo = odeNo

    @property
    def odeNo(self):
        return self._odeNo

    @odeNo.setter
    def odeNo(self, val):
        self._odeNo = val

    def fQ(self, j, Xc):
        '''
        Q matrix
        '''
        return Xc**(2*j)

    def fC(self, j, Xc):
        '''
        C matrix
        '''
        if j == 0:
            return 0
        else:
            return (2*j)*(Xc**(2*j-1))

    def fD(self, j, Xc):
        '''
        D matrix
        '''
        if j == 0:
            return 0
        if j == 1:
            return 2
        else:
            return 2*j*(2*j-1)*(Xc**(2*j-2))

    def buildMatrix(self):
        '''
        build Q,C,D matrix
        '''
        # try/except
        try:
            # number of OC points
            N = OrCoClass.N

            # residual matrix shape
            residualMatrixShape = (self.odeNo*N, self.odeNo*N)
            # rhs
            rhsMatrixShape = self.odeNo*N
            # fdydt
            fdydtShape = self.odeNo*N

            # Evaluate Solution at Collocation Points
            # ----------------------------------------
            # point x1
            # y(1) = d1 + d2*x(1)^2 + d3*x(1)^4 + d4*x(1)^6
            # point x2
            # y(2) = d1 + d2*x(2)^2 + d3*x(2)^4 + d4*x(2)^6
            # point x3
            # y(1) = d1 + d2*x(3)^2 + d3*x(3)^4 + d4*x(3)^6

            # define Q matrix
            Q = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    Q[i][j] = self.fQ(j, OrCoClass.Xc[i])

            # y = Q*d
            # d = y/Q = y*[Q inverse]

            # Evaluate First Derivative at Collocation Points
            # ------------------------------------------------
            # point x1
            # dy(1) = 0 + 2*d2*x1 + 4*d3*x1^3 + ...

            # dy = [dy1 dy2 dy3 dy4];
            # C0 = [
            # 0 1 2*x1 3*x1^2;
            # 0 1 2*x2 3*x2^2;
            # 0 1 2*x3 3*x3^2;
            # 0 1 2*x4 3*x4^2
            # ]

            # define C matrix
            C = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    C[i][j] = self.fC(j, OrCoClass.Xc[i])

            # d = [d1 d2 d3 d4];
            # y' = A*y
            # Q inverse
            invQ = np.linalg.inv(Q)
            # A matrix
            A = np.dot(C, invQ)

            # Evaluate Second Derivative at Collocation Points
            # ------------------------------------------------
            # point x1
            # ddy(1) = 0 + 2*d2 + 12*d3*x1^2 + ...

            # ddy = [ddy1 ddy2 ddy3 ddy4];
            # D0 = [
            # 0 0 2 6*x1;
            # 0 0 2 6*x2;
            # 0 0 2 6*x3;
            # 0 0 2 6*x4
            # ]

            # define D matrix
            D = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    D[i][j] = self.fD(j, OrCoClass.Xc[i])
            # print("D Matrix: ", D)

            # d = [d1 d2 d3 d4];
            # y'' = B*y
            # B matrix
            B = np.dot(D, invQ)

            # result
            res = {
                "Xc": OrCoClass.Xc,
                "Q": Q,
                "A": A,
                "B": B
            }

            return res

        except Exception as e:
            raise


# test
# myClass = OrCoClass(3)
# res = myClass.buildMatrix()
# print("res:")
