# FINITE ELEMENT METHOD
# ----------------------

# import package/module
import numpy as np


class FiElClass:
    '''
    orthogonal collocation on finite element (ocfe)
    '''
    # define mesh
    # ----------------
    # domain length [m]
    DoLe = 1
    # number of elements
    # NuEl = 5
    # number of collocation points per element
    NuCoPo = 4
    # number of interior collocation points per element
    NuInCoPo = NuCoPo - 2
    # number of total collocation points
    # NuToCoPo = NuEl*(NuCoPo-1)+1

    # define collocation points
    # -------------------------
    # x1,x2,x3,x4
    x1 = 0
    x2 = 0.21132
    x3 = 0.78868
    x4 = 1
    # u value
    u1 = x1
    u2 = x2
    u3 = x3
    u4 = x4

    # collocation points
    Xc = np.array([x1, x2, x3, x4])
    # 6 points
    # Xc = np.array([x1, x2, x3, x4, x5, x6, x7])
    # 5 points
    # Xc = np.array([x1, x2, x3, x4, x5, x6])

    # collocation + boundary condition points
    N = NuCoPo
    # collocation points number
    Nc = NuInCoPo
    # collocation points in elements
    Uc = np.array([u1, u2, u3, u4])

    def __init__(self, NuEl):
        self.NuEl = NuEl
        pass

    def NuToCoPoSet(self):
        '''
        number of total collocation points
        '''
        NuToCoPo = self.NuEl*(FiElClass.NuCoPo-1)+1
        return NuToCoPo

    def fQ(self, j, Xc):
        '''
        Q matrix
        '''
        if j == 0:
            return 1
        else:
            return Xc**(j)

    def fC(self, j, Xc):
        '''
        C matrix
        '''
        if j == 0:
            return 0
        elif j == 1:
            return 1
        else:
            return j*(Xc**(j-1))

    def fD(self, j, Xc):
        '''
        D matrix
        '''
        if j == 0:
            return 0
        elif j == 1:
            return 0
        elif j == 2:
            return 2
        else:
            return j*(j-1)*(Xc**(j-2))

    def hiSet(self):
        '''
        set h[i] length
        '''
        # h = 1/NuEl;
        hMatShape = (self.NuEl)
        h = np.zeros(hMatShape)
        for i in range(self.NuEl):
            h[i] = self.DoLe/self.NuEl

        # res
        return h

    def LiSet(self, hi):
        '''
        set L[i] length
        '''
        LiMat = np.zeros(self.NuEl+1)
        sum = 0
        for k in range(len(hi)):
            if k == 0:
                LiMat[k] = 0
            else:
                sum = hi[k] + sum
                LiMat[k] = sum

        # last length
        LiMat[-1] = np.sum(hi)
        # return
        return LiMat

    def xiSet(self, NuToCoPo, hi, li):
        '''
        set x[i] in the domain
        args:
            NuToCoPo: total number of collocation points
        '''
        xiMatShape = (NuToCoPo)
        xiMat = np.zeros(xiMatShape)
        # set index
        n = 0
        for k in range(self.NuEl):
            for j in range(self.NuCoPo-1):
                if j == 0:
                    xiMat[n] = li[k]
                else:
                    xiMat[n] = li[k] + hi[k]*self.Xc[j]
                # set
                n = n+1
        # last node
        xiMat[-1] = li[-1]
        # res
        return xiMat

    def initFiEl(self):
        '''
        init finite element method
        '''
        try:
            # number of OC points
            N = FiElClass.N

            # Evaluate Solution at Collocation Points
            # ----------------------------------------
            # define Q matrix
            Q = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    Q[i][j] = self.fQ(j, FiElClass.Xc[i])

            # y = Q*d
            # d = y/Q = y*[Q inverse]

            # Evaluate First Derivative at Collocation Points
            # ------------------------------------------------
            # define C matrix
            C = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    C[i][j] = self.fC(j, FiElClass.Xc[i])

            # d = [d1 d2 d3 d4];
            # y' = A*y
            # Q inverse
            invQ = np.linalg.inv(Q)
            # A matrix
            A = np.dot(C, invQ)

            # Evaluate Second Derivative at Collocation Points
            # ------------------------------------------------
            # define D matrix
            D = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    D[i][j] = self.fD(j, FiElClass.Xc[i])
            # print("D Matrix: ", D)

            # d = [d1 d2 d3 d4];
            # y'' = B*y
            # B matrix
            B = np.dot(D, invQ)

            # define h - mesh setting
            hi = self.hiSet()

            # li
            li = self.LiSet(hi)

            # total number of collocation points
            NuToCoPo = self.NuToCoPoSet()

            # xi length
            xi = self.xiSet(NuToCoPo, hi, li)

            # result
            res = {
                "NuEl": self.NuEl,
                "NuToCoPo": NuToCoPo,
                "hi": hi,
                "li": li,
                "xi": xi,
                "N": FiElClass.N,
                "Xc": FiElClass.Xc,
                "Q": Q,
                "A": A,
                "B": B
            }

            # return
            return res
        except Exception as e:
            raise
