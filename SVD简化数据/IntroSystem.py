from numpy import *
from numpy import linalg as la

def loadExData():
    return[[4, 4, 0, 2, 2],
           [4, 0, 0, 3, 3],
           [4, 0, 0, 1, 1],
           [1, 1, 1, 2, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]

def loadExData2():
    return [[2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


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

# 计算相似度
def standEst(dataMat, user, simMeas, item):
    '''
    :param dataMat: 数据集合
    :param user: 用户
    :param simMeas: 相似度计算方法
    :param item: 该物品
    :return:
    '''
    n = shape(dataMat)[1]  # 获得数据集的列数, 既物品个数
    simTotal = 0.0  # 记录总的相似度
    ratSimTotal = 0.0  # 记录乘以用户评分权重的总的相似度
    for j in range(n):  # 遍历每个物品
        userRating = dataMat[user, j]  # 获取用户对该物品的评分
        if userRating < 1e-3:  # 若果评分为0, 跳过该物品, 检索到下一个物品
            continue
        #print('item=',j)
        #print('nonzero=',nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0])
        # 这样做避免了在用余弦相似度计算过程中不准确问题, 因为若果用户对其没有评分, 这说明这0为一个未知数, 不能用于计算相似度
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]  # 获取该列物品与item列物品中被同时被评分了的用户学列

        if len(overLap) == 0:  # 若果两个物品没有被评过分, 则相似度为0, 无效数据
            similarity = 0  # 初始化相似度
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])  # 计算两个物品的相似度
        #print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity  # 计算总的相似度
        ratSimTotal += similarity * userRating  # 计算乘以用户评分权重的总的相似度
    if simTotal == 0:  # 若果总相似度为0, 极端情况, 无效数据
        return 0  # 返回评分为0
    else:
        return ratSimTotal / simTotal  # 返回用户对该物品的评分

# 基于SVD评分估计
def svdEst(dataMat, user, simMeas, item):
    '''
    # 注: 对于稀疏矩阵, SVD效果很好, 经过SVD处理, 矩阵会变稠密
    :param dataMat: 数据集
    :param user: 用户
    :param item: 物品列号
    :param simMeas: 计算相似度的方法
    :return: 用户对该物品的预测评分
    '''

    n = shape(dataMat)[1]  # 获得数据集的列数, 既物品个数
    simTotal = 0.0  # 记录总的相似度
    ratSimTotal = 0.0  # 记录乘以用户评分权重的总的相似度
    U, Sigma, VT = la.svd(dataMat)  # 矩阵分解
    print(Sigma)
    index = calSigmaIndex(Sigma)  # 选取sigma能量值超过90%的前n个奇异值的下标
    #print(index)
    Sig4 = mat(eye(index) * Sigma[:index])  # 获取总能量超过90%的前4个奇异值
    xformedItems = dataMat.T * U[:, :index] * Sig4.I  # 构建新矩阵
    #print('xformedItems=',xformedItems)
    for j in range(n):  # 遍历每个物品
        userRating = dataMat[user, j]  # 获取用户评分
        if userRating < 1e-3 or j == item:
            continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)  # 计算相似度
        #print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity  # 计算总的相似度
        ratSimTotal += similarity * userRating  # 计算乘以用户评分权重的总的相似度
    if simTotal == 0:  # 若果总相似度为0, 极端情况, 无效数据
        return 0  # 返回评分为0
    else:
        return ratSimTotal / simTotal  # 返回用户对该物品的评分

# 选取sigma能量值超过90%的前n个奇异值, 返回n值
def calSigmaIndex(Sigma):
    Sig = Sigma**2  # 处理Sigma
    index = len(Sigma)  # 初始化下标
    totalEnergy = sum(Sig)  # 计算总能量
    for i in range(1,len(Sigma)):  # 遍历所有的奇异值
        childEnergy = sum(Sig[:i])  # 计算钱i个奇异值的总能量
        if childEnergy >= totalEnergy*0.9:  # 满足条件
            index = i  # 记录下标
            break
    return index

# 基于物品的推荐系统
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    '''
    :param dataMat: 数据集
    :param user: 用户
    :param N: 推荐个数
    :param simMeas: 计算相似度的
    :param estMethod: 计算方法, 默认是相似度计算
    :return: 预测评分最高的前N个物品, 以及其评分
    '''

    unratedItems = nonzero(dataMat[user, :].A == 0)[1]  # 找到用户没有评分的物品
    #print('unratedItems=', unratedItems)
    if len(unratedItems) == 0:  # 如果用户都已经评过分了, 退出推荐系统
        return 'you rated everything'
    itemScores = []  # 创建列表, 用于记录物品评分
    for item in unratedItems:  # 遍历每一个用户没有评分的物品
        estimatedScore = estMethod(dataMat, user, simMeas, item)  # 预测用户对该物品的评分
        itemScores.append((item, estimatedScore))  # 记录该物品与预测评分
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]  # 取前N个评分最高的物品推荐出去


if __name__ == '__main__':
    dataMat = loadExData2()
    user = 0
    result = recommend(mat(dataMat), user)
    print(result)
    result = recommend(mat(dataMat), user, estMethod=svdEst)
    print(result)
