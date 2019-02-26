from numpy import *
import os
import operator

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 获取数据集样本点的个数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 创建一个矩阵行数为dataSetSize, 列数为1, 每个元素都为inX
    # 计算欧氏距离
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    # 排序
    sortedDistIndicies = distances.argsort()
    classCount ={}
    for i in range(k):  # 获取与传入数据最接近的前k个样本
        voteIlabel = labels[sortedDistIndicies[i]]  # 获取前k个样本中某一个的标签
        classCount[voteIlabel] = classCount.get(voteIlabel ,0) + 1  # 出现次数加一
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  #　从大到小排序
    return sortedClassCount[0][0]  # 获取前k个样本之中, 出现次数最多的标签

# 见图片转化为向量
def img2vector(filename):
    returnVect = zeros((1 ,1024))  # 初始化一个1*1024的矩阵
    # 将一个32*32的图片转化为向量
    # print(filename)
    '''
    fr = open(filename,'r',encoding='utf-8')
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    '''
    with open(filename) as file_object:
        for i in range(32):
            lineStr = file_object.readline()
            for j in range(32):
                returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []  # 记录所有实例的标签
    trainingFileList = os.listdir('digits/trainingDigits')  # 加载数据, 获取该目录下所有文件,　既是获取训练样本
    m = len(trainingFileList)  # 获取有多少个实例
    trainingMat = zeros((m, 1024))  # 创建一个m行, 1024列的矩阵
    for i in range(m):  # 遍历所有文件, 将图片转化为向量
        fileNameStr = trainingFileList[i]  # 获取文件名字
        fileStr = fileNameStr.split('.')[0]  # 分割文件名
        classNumStr = int(fileStr.split('_')[0])  # 获取该图片的标签
        hwLabels.append(classNumStr)  # 添加标签
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)  # 获取该图片的向量

    testFileList = os.listdir('digits/testDigits')  # 获取测试样本
    errorCount = 0.0  # 记录错误分类次数
    mTest = len(testFileList)  # 获取测试集长度
    for i in range(mTest):  # 遍历所有文件, 将图片转化为向量
        fileNameStr = testFileList[i]  # 获取文件名字
        fileStr = fileNameStr.split('.')[0]  # 分割文件名
        classNumStr = int(fileStr.split('_')[0])  # 获取该图片的标签
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)  # 将图片转化为向量
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)  # 分类获取分类器返回的标签
        #print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):  # 如果分类错误
            errorCount += 1.0  # 错误次数加一
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount /float(mTest)))
    print('the total right rate is: %f' % (1 - (errorCount /float(mTest)) ))


if __name__=='__main__':
    handwritingClassTest()
