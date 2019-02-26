from numpy import *

# 注意os也有open函数, 不要from os import *
returnVect = zeros((1, 1024))  # 初始化一个1*1024的矩阵
# 将一个32*32的图片转化为向量

with open('digits/trainingDigits/0_0.txt') as file_object:
    for i in range(32):
        lineStr = file_object.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])

print(returnVect)