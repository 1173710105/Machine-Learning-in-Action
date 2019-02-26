from numpy import *

def loadDataSet():
    # 创建特征集
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 创建标签
    return postingList, classVec

# 创建词列表
def createVocabList(dataSet):
    vocabSet = set([])  # 创建数据集合
    for document in dataSet:  # 遍历数据集
        vocabSet = vocabSet | set(document)  # 创建两个集合的并集
    return list(vocabSet)   # 一个列表

# 创建词向量 one—hot法
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)  # 创建一个长度为词列表长度, 所有所有元素都为0的列表(向量)
    for word in inputSet:  # 遍历输入句子的每一个单词
        if word in vocabList:  # 如果该词在词表之中
            returnVec[vocabList.index(word)] = 1  # 将对应位置的特征值设置为1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec  # 返回词向量

# 计算p(wi|c1) p(wi|c0) p(c1) p(c0)
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)  # 获取训练样本的个数
    numWords = len(trainMatrix[0])  # 获取每个样本向量的维度
    pAbusive = sum(trainCategory)/float(numTrainDocs)  # 计算标签为1的样本的先验概率
    # 这样初始化的作用为避免出现p(wi|ci)为0的情况
    p0Num = ones(numWords)  # 创建一个所有元素都为1, 维度为numWords的列表
    p1Num = ones(numWords)  # 创建一个所有元素都为1, 维度为numWords的列表
    p0Denom = 2.0  # 初始化分母
    p1Denom = 2.0  # 初始化分母
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 如果该样本的标签为1, 既是侮辱性文本
            p1Num += trainMatrix[i]  # 两个向量相加
            p1Denom += sum(trainMatrix[i])  # 计算每个向量各个维度的和
        else:  # 如果该样本的标签为0, 既是非侮辱性文本
            p0Num += trainMatrix[i]  # 两个向量相加
            print(p0Num)
            p0Denom += sum(trainMatrix[i])  # 计算每个向量各个维度的和
    p1Vect = log(p1Num/p1Denom)  # 计算p(wi|c1)  取对数是防止下溢出
    p0Vect = log(p0Num/p0Denom)  # 计算p(wi|c0)
    return p0Vect, p1Vect, pAbusive

# 拓扑贝叶斯分类器
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''

    :param vec2Classify: 要分类的向量
    :param p0Vec: p(w|c0)
    :param p1Vec: p(w|c1)
    :param pClass1: p(c1)
    :return:
    '''
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # 对数相加,既是相乘
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
# 测试
def testingNB():
    listOPosts,listClasses = loadDataSet()  # 获取数据
    myVocabList = createVocabList(listOPosts)  # 获取词列表
    trainMat=[]  # 初始化训练样本矩阵
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  # 获取每一个样本对应的词向量
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))


if __name__=='__main__':
    testingNB()