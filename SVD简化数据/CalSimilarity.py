from numpy import *
from numpy import linalg as la


# 计算欧氏距离, 归化到0~1
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

# 计算皮尔逊相关系数, 归化到0~1
def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar=0)[0][1]

# 计算余弦相似度
def cosSim(inA,inB):
    num = float(inA.T*inB)  # 分子
    denom = la.norm(inA)*la.norm(inB)  # 分母, la.norm() 是计算向量的范数
    return 0.5+0.5*(num/denom)
