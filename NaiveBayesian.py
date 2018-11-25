# Author:Stark-zheng
import numpy as np

'''贝叶斯分类器(两类数据)：输入为文档词与标签，每个文档一行，词之间用空格分割，最后为文档标签'''
class NBmodel(object):
    def __init__(self, wordsList):
        self.labels = []    # 标签列表，用来存放样本标签
        for i in wordsList:     # 将类别用不同数字代替，可根据样本类别修改
            label = i.pop()
            if label == '合法':
                self.labels.append(1)
            else:
                self.labels.append(0)
        self.wordsList = wordsList  # 词列表
        self.docLen = len(wordsList)    # 样本数量
        self.vocabSet = set([])     # 不重复词集合
        for i in wordsList:
            self.vocabSet = self.vocabSet | set(i)
        self.vocabList = list(self.vocabSet)      # 不重复词列表
        self.wordVec = self.word2Vec(wordsList)     # 将词列表转化为词向量
        self.pA = self.labels.count(1)/float(self.docLen)   # 类别1的先验概率
        self.p0vec = np.zeros(len(self.vocabList))          # 类别1各属性的概率向量
        self.p1vec = np.zeros(len(self.vocabList))          # 类别2各属性的概率向量
        self.train()                                        # 训练模型

    # 训练模型
    def train(self):
        p1Num = np.array([1 for i in self.vocabList])
        p0Num = np.array([1 for i in self.vocabList])
        p1Sum = 2.0
        p0Sum = 2.0
        for i in range(self.docLen):
            if self.labels[i] == 1:
                p1Num += self.wordVec[i]
                p1Sum += sum(self.wordVec[i])
            else:
                p0Num += self.wordVec[i]
                p0Sum += sum(self.wordVec[i])
        self.p0vec = np.log(p0Num / p0Sum)
        self.p1vec = np.log(p1Num / p1Sum)

    # 测试数据，输入为测试样本词与标签，打印精度
    def testModel(self, wordsList):
        labels = []
        correct = 0
        for i in wordsList:
            label = i.pop()
            if label == '合法':
                labels.append(1)
            else:
                labels.append(0)
        vocavList = self.word2Vec(wordsList)
        for i in range(len(vocavList)):
            if self.test(vocavList[i]) == labels[i]:
                correct += 1
        print('Test accuracy:', float(correct / len(labels)))

    # 测试样本，输入为一个样本词与标签，返回类别（1 or 0）
    def test(self, docVec):
        p1 = sum(docVec * self.p1vec) + np.log(self.pA)
        p0 = sum(docVec * self.p0vec) + np.log(1 - self.pA)
        if p1 > p0:
            return 1
        else:
            return 0

    # 将词列表转化为词向量
    def word2Vec(self, wordsList):
        words2vec = []
        for words in wordsList:
            vec = [0 for i in self.vocabList]
            for word in words:
                if word in self.vocabList:
                    vec[self.vocabList.index(word)] += 1
            words2vec.append(vec)
        return words2vec


if __name__ == '__main__':
    file = open('C:\\Users\mechrev\Desktop\新建文件夹\\2000--违法状态.txt', encoding='utf-8')
    file = file.readlines()
    feature = []
    for line in file[:]:  # 一行行读数据文件
        line = line.strip()
        tempVec = line.split(' ')
        feature.append(tempVec)  # 将特征放入特征列表
    model = NBmodel(feature[:int(len(feature)*0.8)])
    model.testModel(feature[int(len(feature)*0.8):])

