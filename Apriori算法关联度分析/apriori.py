from numpy import *

# 创建数据集合
def loadDataSet():
    # 列表每一个元素(子列表)代表一条交易记录, 子列表里面的每一个元素代表该次交易购买的商品
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

# 创建商品集合列表
def createC1(dataSet):
    C1 = []  # 初始化列表, 用于存储商品信息
    for transaction in dataSet:
        for item in transaction:
            if [item] not in C1:  # 如果商品信息没有出现过, 则添加到列表之中
                C1.append([item])
    return list(map(frozenset, C1))

# 计算项集的支持度, 获取满足最小支持度的项集
def scanD(D, Ck, minSupport):
    ssCnt = {}  # 创建空字典, 记录商品类型以及出现次数
    for tid in D:  # 遍历数据集, 即遍历数据集的交易记录
        for can in Ck:  # 遍历商品集合列表
            if can.issubset(tid):  # 若果该商品在交易记录之中, 出现次数加一
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))  # 获取一用有多少次交易记录
    retList = []  # 储存满足最小支持度的商品
    supportData = {}  # 记录所有商品以及商品集合的支持度信息
    for key in ssCnt:  # 遍历每一件商品
        support = ssCnt[key] / numItems  # 获取该商品的支持度
        if support >= minSupport:  # 满足最小支持度
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

# 合并项集
def aprioriGen(Lk, k):
    retList = []  # 储存满足最小支持度的商品 或者是 商品项集
    lenLk = len(Lk)  # 获取数据集合长度
    for i in range(lenLk):  # 遍历数据集
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]  # 获取前k-2个数据
            L2 = list(Lk[j])[:k - 2]  # 获取前k-2个数据
            L1.sort()
            L2.sort()
            if L1 == L2:  # 年如果前k-2个相同
                retList.append(Lk[i] | Lk[j])  # 合并
    return retList  # 返回合并之后的项集

# 完整的apriori算法
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)  # 获取商品集合列表
    # D = map(set, dataSet)
    L1, supportData = scanD(dataSet, C1, minSupport)  # 计算项集的支持度, 获取满足最小支持度的项集L1
    L = [L1]  # L 为存储满足最小支持度的不同个数的商品项集

    # {0,1} {0,2} {1,2} 三者两两合并结果都一样,正常合成需要三次, k-2可减少合并次数,只需合拼一次
    k = 2  # 初始化k
    while (len(L[k - 2]) > 0):  # 如果列表地k-2个元素(值列表大于0),这说明存在合并的前项,循环继续
        Ck = aprioriGen(L[k - 2], k)  # 获取合并之后的项集
        Lk, supK = scanD(dataSet, Ck, minSupport)  # 获取在该项集情况下, 获取满足最小支持度的项集Lk
        supportData.update(supK)  # 更新商品 以及 商品集 支持度信息
        L.append(Lk)  # 将足最小支持度的项集增加到L中
        k += 1  # 增加k数值, 计算更大的商品集
    return L, supportData

# 生成关联规则
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []  # 记录规则列表
    for i in range(1, len(L)):  # 遍历所有的满足最小支持度的项集数大于等于2的商品项集
        for freqSet in L[i]:  # 遍历项集数为i的所有子商品项集
            #print('freqSet=%s'%freqSet)
            H1 = [frozenset([item]) for item in freqSet]  # 获取该子商品项集里面的所有商品项集
            #print('H1=%s'%H1)
            if i > 1:  # 若果该商品集元素的里面不只两项
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:  # 若果商品集元素的里面只有两项,
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

# 计算可信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []  # 用于存储后件的商品项集
    #print('H=%s'%H)
    for conseq in H:  # 遍历候选规则集合
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # 计算信用度
        if conf >= minConf:  # 若果信用度满足最小信用度要求
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))  # bigRuleList 增加这一个规则
            prunedH.append(conseq)  # 增加这一商品集合
            #print('conseq=%s'%conseq)
            #print('prunedH=%s'%prunedH)
    return prunedH

# 生成候选规则集合
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])  # 获得该商品项集里面每个元素所包含的商品
    #print('m=%s'%m)
    if len(freqSet) > (m + 1):  # 若果满足该条件, 则说明该商品集合可以构建规则
        Hmp1 = aprioriGen(H, m + 1)  # 合并项集, 生成候选规则集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)  # 获取后件的商品项集
        if (len(Hmp1) > 1):  # 若后件商品个数大于1, 说明仍有可能存在构建规则的可能
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

if __name__ == '__main__':
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    retList, supportData = scanD(dataSet, C1, 0.5)
    #print(retList)
    #print(supportData)

    L, supportData = apriori(dataSet)
    #print(L)
    #print(supportData)
    rules = generateRules(L, supportData, 0.5)
    print(rules)

