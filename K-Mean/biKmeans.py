from numpy import *
import matplotlib.pyplot as plt

# 加载数据
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')  # 将数据拆开返回一个列表
        fltLine = list((map(float, curLine)))  # 将列表里面的数字转化为浮点类型
        dataMat.append(fltLine)  # 添加到列表之中
    return dataMat  # 返回一个列表

# 计算欧氏距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # 计算欧氏距离

# 随机构造仔质心
def randCent(dataSet, k):
    #print(k)
    n = shape(dataSet)[1]  # 获取样本点的的维数
    centroids = mat(zeros((k, n)))  # zeros[k,n]构造一个k个元素，每个元素有n个0的列表
                                    # 将列表变为k行, n列的矩阵
    for j in range(n):  # 随机生成一个最小向量与最大向量之间的一个向量
        minJ = min(dataSet[:, j])  # 获取所有样本点, 第J维度的最小值
        rangeJ = float(max(dataSet[:,j]) - minJ)  # 第J列的最大在减去第J列的最小值, 获取最大值与最小值之间的范围
        centroids[:, j] = minJ + rangeJ*random.rand(k, 1)  # 随机生成质心的第J维的取值
                                                           # random.rand(k, 1) 随机生成k个, 取值范围为0~1的随机数
    return centroids  # 返回随机矩阵


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]  # 获取样本点的个数
    clusterAssment = mat(zeros((m,2)))  # 创造一个m行, 维度为2的矩阵, 每个元素的全零变量
                                        # 第一列存簇索引值，第二列存当前点到簇质心的距离
    centroids = createCent(dataSet, k)  # 随机创建k个簇心
    clusterChanged = True  # 创建标注标量，用来达到条件就终止循环
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 遍历每个样本
            minDist = inf   # 初始样本到簇质心的距离  nif 表示正无穷；-nif表示负无穷
            minIndex = -1   # 初始化最小样本点的下标
            for j in range(k):  # 遍历每个簇质心
                distJI = distMeas(centroids[j,:], dataSet[i,:])  # 计算样本点到簇质心的距离
                if distJI < minDist:    # 寻找距离该样本点最近的簇质心
                    minDist = distJI
                    minIndex = j   # 将样本分配到距离最小的质心那簇
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True  # 如果样本分配结果发生变化，更改标志变量

            # 将结果存入clusterAssment
            clusterAssment[i, :] = minIndex, minDist**2  # 将该点距离其最近簇心的下标和距离记录到矩阵之中
        #print(centroids)
        for cent in range(k):  # 遍历每个簇质心
            # 找到每个簇质心对应的样本
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 获取所有距离最近的质心为当前质心的样本点
            centroids[cent,:] = mean(ptsInClust, axis=0)  # 计算这些样本的均值，作为该簇的新质心

    return centroids, clusterAssment  # 返回质心 以及 样本点与质心之间的质心


#二分法寻找最优聚类
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]  # 获取样本点的个数
    clusterAssment = mat(zeros((m,2)))  # 创造一个m行, 维度为2的矩阵, 每个元素的全零变量
                                        # 第一列存簇索引值，第二列存当前点到簇质心的距离

    centroid0 = mean(dataSet, axis=0).tolist()[0]  # 初始化质心, 将所有点归结为一个簇, 将所有的点对应维度的平均值作为该质心的对应维度的值
    centList = [centroid0]  # 创造一个质心列表
    for j in range(m):  # 遍历每个样本点, 计算每个样本点距离质心的距离, 既是误差值
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j, :])**2

    while len(centList) < k:  # 当质心小于指定个数时, 循环继续
        lowestSSE = inf  # 设置最小的SSE

        for i in range(len(centList)):  # 遍历当前的质心
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]  # 选取一该质心为当前最近质心的样本点
            # 使用kMean算法对当前数据集进行二分，产生两个质心和对应样本点距离两个质心的最短距离，注意是针对等于当前质心下标的样本点
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 计算划分的SSE，即误差和（一个质心变两个，对应着一个样本集变两个）
            sseSplit = sum(splitClustAss[:,1])  # 计算二划分之后的SSE
            # 计算非当前质心的其他质心的SSE
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            #print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            #计算划分后的最优SSE，即最有序的生成簇
            if (sseSplit + sseNotSplit) < lowestSSE:  # 若果当前划分簇SSE小于最小SSE, 既记当前划分簇为最小划分
                # 最优的需要拆分的质心：一个簇，变成两个
                bestCentToSplit = i  # 最优进行二划分的簇心
                # 最优的二分质心点的值
                bestNewCents = centroidMat  # 对最优秀质点进行二划分之后的质心
                # 最优的距离上面两个质心的样本的距离
                bestClustAss = splitClustAss.copy()
                #最优的SSE的值
                lowestSSE = sseSplit + sseNotSplit

        # 更新样本点第一维度, 既记录当前样本点距离最近质心的维度, 更新为最新划分出来的质心的下标
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0],0] = len(centList)
        # 暂时做一个标记, 用与更新数据集合
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0],0] = bestCentToSplit
        #print('the bestCentToSplit is: ',bestCentToSplit)
        #print('the len of bestClustAss is: ', len(bestClustAss))

        # 更新质心列表, 将被划分的质点替代为两个最新划分出来的质点
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]  # 替换被划分的质点
        # 再新增一个质心
        centList.append(bestNewCents[1, :].tolist()[0])

        #print("划分前的最优距离:\n", shape(clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :]))
        #print('划分后的最优距离:\n',shape(bestClustAss))

        # 更新原来到最优质心的那些样本点对应的值，改成现在新的质心和对应距离
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],:] = bestClustAss

    return mat(centList), clusterAssment

def plot(dataSet,centValue, clusterList):

    x1=dataSet[:, 0]
    x2=dataSet[:, 1]
    fig=plt.figure('k均值算法')

    ax=fig.add_subplot(111)
    ax.scatter(list(x1),list(x2),s=15,c='red',marker='s')
    ax.scatter(list(centValue[:,0]), list(centValue[:,1]), s=15, c='green', marker='x')

    for i in range(cenValue.shape[0]):
        a = cenValue[i, 0]
        b = cenValue[i, 1]
        r = sqrt(max(clusterList[nonzero(clusterList[:, 0].A == i)[0], 1])[0,0])
        theta = arange(0, 2 * pi, 0.01)
        x = a + r * cos(theta)
        y = b + r * sin(theta)
        ax.plot(x, y)
        ax.axis('equal')
    '''
    for i in len(cenValue):
        r = 2.0
        a, b = list(cenValue[i, :])
        theta = arange(0, 2 * pi, 0.01)
        x = a + r * cos(theta)
        y = b + r * sin(theta)
        ax.plot(x, y)
        ax.axis('equal')
    '''
    plt.show()

if __name__ == '__main__':
    dataSet = loadDataSet('testSet.txt')
    datMat3 = mat(dataSet)
    print(datMat3)
    cenValue, clusterList = biKmeans(datMat3, 4)
    plot(datMat3, cenValue, clusterList)

