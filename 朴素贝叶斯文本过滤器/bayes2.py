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

# 创建词向量 one—hot法 词集模型
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
            p0Denom += sum(trainMatrix[i])  # 计算每个向量各个维度的和
    p1Vect = log(p1Num/p1Denom)  # 计算p(wi|c1)  取对数是防止下溢出
    p0Vect = log(p0Num/p0Denom)  # 计算p(wi|c0)
    return p0Vect, p1Vect, pAbusive

# 朴素贝叶斯分类器
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
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))  # 计算p(wi|c1) p(wi|c0) p(c1)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    #print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    #print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

# 创建词向量 one—hot法 词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1  # 每个词可以出现多次
    return returnVec

# 切分文本
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 获取所有单词, 并且去除空格和标点符号, 将所有单词变为小写


def spamTest():
    docList = []  # 存储所有邮件的文本信息
    classList = []  # 储存邮件的类别
    fullText = []  # 储存所有邮件单词
    for i in range(1, 26):
        # 获取垃圾邮件的信息
        wordList = textParse(open('email/spam/%d.txt' % i).read())  #
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # 对应垃圾邮件的标签
        # 获取非垃圾邮件信息
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)  #
        fullText.extend(wordList)
        classList.append(0)  # 对应非垃圾邮件的标签

    vocabList = createVocabList(docList)  # 创建词列表
    trainingSet = list(range(50))  # 获取1~50的列表
    testSet = []  # 获取训练数据集

    # 随机获取10个作为测试数据集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))  # 随机获取一个下标
        testSet.append(trainingSet[randIndex])  # 增加该实例到测试集之中
        del (trainingSet[randIndex])  # 从训练集之中删除该实例

    trainMat = []  # 训练词向量矩阵
    trainClasses = []  # 记录标签
    for docIndex in trainingSet:  # 遍历每一个训练数据集, 获取每一个样本
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))  # 将样本转化为词向量, 并且记录
        trainClasses.append(classList[docIndex])  # 记录该样本的类别

    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))  # 计算p(wi|c1) p(wi|c0) p(c1)

    errorCount = 0
    for docIndex in testSet:  # 遍历每一个测试数据集合
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])  # 获取该训练数据集对应的词向量
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  # 预测结果与原本结果比较
            errorCount += 1  # 预测错误, 错误次数加一
            #print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount) / len(testSet))  # 统计错误率
    # return vocabList,fullText
    return float(errorCount) / len(testSet)


if __name__ == '__main__':
    errorList = []
    for i in range(1,10):
        errorList.append(spamTest())
    print('the total accurate rating is :%s'%(1 - float(sum(errorList))/len(errorList)))
    print('the total error rating is :%s'%(float(sum(errorList))/len(errorList)))