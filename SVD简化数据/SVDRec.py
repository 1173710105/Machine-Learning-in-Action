from numpy import *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]


def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))


def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]


def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)


if __name__ == '__main__':
    Data = loadExData()
    U, Sigma, VT = linalg.svd(Data)
    print('U=', U)
    print('Sigma=', Sigma)
    print('VT=', VT)
    Sig2 = mat([[Sigma[0], 0], [0, Sigma[1]]])
    newData = U[:, :2]*Sig2*VT[:2, :]
    print('newData=',newData)

