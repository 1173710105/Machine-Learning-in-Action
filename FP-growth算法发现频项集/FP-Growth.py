
class treeNode(object):
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue  # 节点名字
        self.count = numOccur  # 节点出现次数
        self.nodeLink = None  # 储存相似关系的指针
        self.parent = parentNode  # 储存父节点
        self.children = {}  # 储存子树

    #
    def inc(self, numOccur):
        self.count += numOccur

    # 按层展示树
    def disp(self, ind=1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)

# 构建FP树
def createTree(dataSet, minSup=1):
    headerTable = {}  # 头指针表, 记录样本点之中元素项, 以及其出现次数
    for trans in dataSet:  # 遍历数据集所有的样本点
        for item in trans:  # 遍历样本点里面的每一元素
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]  # 该元素出现次数加
            #print('dataSet[trans]=%s'%dataSet[trans],1)
    for k in list(headerTable.keys()):  # 遍历所有元素
        if headerTable[k] < minSup:  # 若果该元素不满足最小支持度
            #del (KeyList[k])
            del (headerTable[k])  # 删除该元素

    freqItemSet = set(headerTable.keys())  # 获取频项元素集合
    # print('freqItemSet: ',freqItemSet)
    if len(freqItemSet) == 0:  # 若果频项个数等于0, 退出
        return None, None
    for k in headerTable:  # 遍历所有的元素
        #print('headerTable[k]=%s' % headerTable[k])
        #print(k)
        headerTable[k] = [headerTable[k], None]  # 重新构造头表指针, 键存储该元素的, 值存储该元素的出现次数与存储关系指针(放在一个列表里面)
        #print('headerTable[k]=%s'%headerTable[k])
    # print('headerTable: ',headerTable)
    retTree = treeNode('Null Set', 1, None)  # 初始化树
    for tranSet, count in dataSet.items():  # 遍历树的所有键值对, 键是事务, 值是该事物出现次数
        localD = {}  # 记录元素以及其出现次数
        for item in tranSet:  # 遍历该事务的的每一个元素
            if item in freqItemSet:  # 判断该原始是否是频项元素
                localD[item] = headerTable[item][0]  # 记录该元素以及出现次数
        if len(localD) > 0:  # 若果该事务包含频项元素, 进行以下操作
            #print(localD.items())
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]  # 按元素出现次数从大到小的循序对元素进行排序
            #print('orderedItems=%s'%orderedItems)
            #print('orderedItems[0]=%s'%orderedItems[0])
            updateTree(orderedItems, retTree, headerTable, count)  # populate tree with ordered freq itemset
    return retTree, headerTable  # return tree and header table


def updateTree(items, inTree, headerTable, count):
    '''
    :param items: 该事务的子元素
    :param inTree: parent树
    :param headerTable:  头指针表
    :param count: 元素出现次数
    :return: nan
    '''
    #print('items[0]=%s'%items[0])
    if items[0] in inTree.children:  # 若果该元素在子树之中
        inTree.children[items[0]].inc(count)  # 子树节点标记的次数增加count
    else:  # 否则以该元素为子节点, 增加一颗新子树
        inTree.children[items[0]] = treeNode(items[0], count, inTree)  # items[0]表示子节点, count表示子节点的出现次数, inTree表示父节点
        #print('headerTable[items[0]][1]=',headerTable[items[0]][1])
        if headerTable[items[0]][1] == None:  # 若果该节点的关系指针为为空,增加关系指针
            headerTable[items[0]][1] = inTree.children[items[0]]  # 关系指针值指向新生成的子树的元素节点
        else:  # 否则更新头指针表
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:  # 若果该事务还包含其他频项元素, 重复上述过程
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


# 更新头指针表
def updateHeader(nodeToTest, targetNode):
    '''
    :param nodeToTest: 元素对应的关系指针
    :param targetNode: 该元素要指向的元素节点
    :return:
    '''
    while nodeToTest.nodeLink != None:  # 寻找最后的元素, 标志是其指针值为空
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode  # 增加指针到最后元素

# 往上回溯树, 从叶节点到根节点
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:  # 若果父节点不为空
        prefixPath.append(leafNode.name)  # 前缀路径增加该节点
        ascendTree(leafNode.parent, prefixPath)  # 递归往上搜寻

# 找前缀路径，获取条件模式基础
def findPrefixPath(basePat, treeNode):
    '''
    :param basePat:
    :param treeNode:
    :return:
    '''
    condPats = {}  # 创建 条件模式基字典
    while treeNode != None:  # 若果树不为空
        prefixPath = []  # 初始化 前缀路径
        ascendTree(treeNode, prefixPath)  # 往上搜索树, 找到前缀路径
        if len(prefixPath) > 1:  # 若果前缀路径长度大于1
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    #print()
    return condPats  # 返回条件模式基字典

# 递归查找频繁项集
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    #print(headerTable.items())
    #print(headerTable.values())
    #print('headerTable=%s'%headerTable)
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: str(p[1]))]  # 按元素出现次数从小到大的循序对元素进行排序, 返回一个列表
    #print('bigL=%s'%bigL)
    for basePat in bigL:  # 遍历所有的元素, 从叶节点开始
        newFreqSet = preFix.copy()  # 最新路径
        newFreqSet.add(basePat)  # 当前叶节点
        freqItemList.append(newFreqSet)  # 增加最新路径
        #print('basePat=%s'%basePat)
        #print('headerTable[basePat][1]=%s'%headerTable[basePat][1])
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])  # 获取前缀路径, 获取条件模式基础
        #print('condPattBases=%s'%condPattBases)
        myCondTree, myHead = createTree(condPattBases, minSup)  # 利用条件模式基构建条件FP树
        #print('myCondTree=%s'%myCondTree)
        #print('myHead=%s'%myHead)
        # print 'head from conditional tree: ', myHead
        if myHead != None:  # 若果树不为空, 继续挖掘条件FP树
            # print 'conditional tree for: ',newFreqSet
            # myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)  # 递归调用


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

# 处理数据, 用于实现上述从列表到字典的类型转换过程
def createInitSet(dataSet):
    retDict = {}  # 构建字典, 键存储事物, 值储存1, 表示一个事物
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

if __name__ == '__main__':
    '''
    simpData = loadSimpDat()
    initSet = createInitSet(simpData)
    #print(initSet)
    myFPTree, myHeaderTab = createTree(initSet, 3)
    myFPTree.disp()
    freqItems = []
    mineTree(myFPTree, myHeaderTab, 3, set([]), freqItems)
    print(freqItems)
    '''

    # 将数据集导入
    parsedDat = [line.split() for line in open('kosarak.dat').readlines()]
    print(parsedDat)
    # 对初始集合格式化
    initSet = createInitSet(parsedDat)
    # 构建FP树,并从中寻找那些至少被10万人浏览过的新闻报道
    myFPtree, myHeaderTab = createTree(initSet, 50000)
    #myFPtree.disp()
    # 需要创建一个空列表来保存这些频繁项集
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, 10000, set([]), myFreqList)
    # 发现新闻报道或者新闻报道集合被最低阀值以上的人浏览过
    print(len(myFreqList))
    for value in myFreqList:
        print(value)
