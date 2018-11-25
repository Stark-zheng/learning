# Author：Stark-zheng
import numpy as np
import random
import matplotlib.pyplot as plt

'''
    输入为文档对象dpre，由文档列表，总词的序号-词字典（nwdoc）与词-序号字典（wndoc），总次数（wordsNum）组成
    其中文档为文档对象，由文档词，文档标签，文档长度等组成
    k，为输入主题个数
    times，为迭代次数
'''
class LdaModel(object):
    def __init__(self, dpre, k, times):
        self.dpre = dpre        # 数据对象
        self.times = times      # 迭代次数
        self.K = k                  # 主题数
        self.docs = dpre.docs   # 文档对象
        self.wordsNum = dpre.wordsNum    # 总词数
        self.docNum = len(self.docs)     # 总文档数
        self.beta = 0.01            # 超参数α β
        self.alpha = 1.0/k
        # 各词对应的主题
        self.Z = np.array(
            [[0 for y in range(len(self.docs[x].words))] for x in range(self.docNum)])
        # wt，词在各主题上的分布
        # wtsum 各主题中词的数目
        # dt,各文档中各主题词的数目
        # dtsum,各文档中词的数目
        self.p = np.zeros(self.K)
        self.wt = np.zeros((self.wordsNum, self.K), dtype="int")
        self.wtsum = np.zeros(self.K, dtype='int')
        self.dt = np.zeros((self.docNum, self.K), dtype="int")
        self.dtsum = np.zeros(self.docNum, dtype='int')
        # 随机给词分配主题
        for i in range(len(self.docs)):
            self.dtsum[i] = len(self.docs[i].words)
            self.dtsum[i] = self.docs[i].len
            words = self.docs[i].words
            for j in range(len(words)):
                topic = random.randint(0, self.K - 1)
                self.Z[i][j] = topic
                self.wt[self.docs[i].words[j]][topic] += 1
                self.wtsum[topic] += 1
                self.dt[i][topic] += 1
        self.theta = np.array([[0.0 for y in range(self.K)] for x in range(self.docNum)])
        self.phi = np.array([[0.0 for y in range(self.wordsNum)] for x in range(self.K)])
        self.simiArr = np.array([[0.0 for i in range(self.K)] for x in range(self.K)])
    def trainModel(self):
        for x in range(self.times):
            for i in range(self.docNum):
                for j in range(len(self.docs[i].words)):
                    topic = self.sampling(i, j)
                    self.Z[i][j] = topic
        self._phi()
        self._theta()

    # Gibbs采样
    def sampling(self, i, j):
        topic = self.Z[i][j]
        word = self.docs[i].words[j]
        # print(word)
        # zz
        self.wt[word][topic] -= 1
        self.dt[i][topic] -= 1
        self.wtsum[topic] -= 1
        self.dtsum[i] -= 1
        # print(len(self.wt[word]))
        # print(len(self.wtsum))
        # print(len(self.dt[i]))
        # print(len(self.dtsum))
        self.p = (self.wt[word] + self.beta) / (self.wtsum + self.beta * self.wordsNum) * \
                 (self.dt[i] + self.alpha) / (self.dtsum[i] + self.alpha * self.K)
        # print(self.dtsum[i] + self.alpha * self.K)
        for k in range(1, self.K):
            self.p[k] += self.p[k - 1]
        n = random.uniform(0, self.p[self.K - 1])
        # print(self.wt[10:20])
        # print(n>1)
        # print(self.p[self.K - 1])
        for topic in range(self.K):
            if self.p[topic] > n:
                break
        # print(topic)
        # zz
        self.wt[word][topic] += 1
        self.dt[i][topic] += 1
        self.wtsum[topic] += 1
        self.dtsum[i] += 1
        return topic

    # 主题相似度计算，输入参数way为计算方式，为k时使用KL散度，为j时使用JS散度，为c时使用夹角余弦
    def tocSimi(self, way='k'):
        for i in range(self.K):
            for j in range(self.K):
                if way == 'k' or way == 'K':
                    self.simiArr[i][j] = self.KLdiv(self.theta[i], self.theta[j])
                if way == 'j' or way == 'J':
                    self.simiArr[i][j] = self.JSdiv(self.theta[i], self.theta[j])
                if way == 'c' or way == 'C':
                    self.simiArr[i][j] = self.cosSimi(self.theta[i], self.theta[j])
        for i in self.simiArr:
            print(i)

    # 计算向量p，q的KL散度
    def KLdiv(self, p, q):
        m = []
        n = []
        for i in range(len(p)):             # 去除q中为0的值，防止除数为0
            # if p[i] != 0 or q[i] != 0:  # 除数为0
            if q[i] != 0:
                m.append(p[i])
                n.append(q[i])
        p = m
        q = n
        # p, q = zip(*filter(lambda i, j: i != 0 or j != 0, zip(p, q)))
        a = 0.0
        for i, j in zip(p, q):
            a += (i * np.log(i/float(j)))
        return a

    # 计算向量p，q的JS散度
    def JSdiv(self, p, q):
        n = []
        for i in range(len(p)):
            n.append(0.5 * (p[i] + q[i]))
        p = p + np.spacing(1)       # np.spacing(1) 最小非负数，使其变为浮点数
        q = q + np.spacing(1)
        n = n + np.spacing(1)
        return 0.5 * self.KLdiv(p, n) + 0.5 * self.KLdiv(q, n)

    # 计算向量p，q的夹角余弦值
    def cosSimi(self, p, q):
        m = 0.0
        a = 0.0
        b = 0.0
        for i in range(len(p)):
            m += p[i] * q[i]
            a += p[i] ** 2
            b += q[i] ** 2
        # cs = sum(m)/((sum(p)  ** 0.5) * (sum(q) ** 0.5))
        if a == 0 or b == 0:
            return None
        else:
            return m / ((a * b) ** 0.5)

    # 计算theta值
    def _theta(self):
        for i in range(self.docNum):
            self.theta[i] = (self.dt[i] + self.alpha) / (self.dtsum[i] + self.K * self.alpha)

    # 计算phi值
    def _phi(self):
        for i in range(self.K):
            # self.wt.T是wt的转置，转置后为‘tw’形式，即主题里各词的分布
            self.phi[i] = (self.wt.T[i] + self.beta) / (self.wtsum[i] + self.wordsNum * self.beta)

    # 输出主题中的词
    # num为输出主题词的个数，display为显示方式（默认显示词+概率）1为只显示词
    def topicView(self, num=7, display=0):
        for i in self.phi:
            temp = []
            i = list(i)
            for j in range(num):
                if display == 0:
                    temp.append((self.dpre.nwdoc[i.index(max(i))], max(i)))
                else:
                    temp.append(self.dpre.nwdoc[i.index(max(i))])
                i[i.index(max(i))] = 0
            print(temp)

    # 输出文档中占比前num各主题，
    # display为输出方式（0为仅显示主题，1为显示主题并显示概率）
    # visualization=0为输出打印，1为可视化显示
    def docView(self, num=3, display=0, visualization=0):
        if visualization == 1:
            num_list = list(range(self.K))
            for i in self.theta:
                plt.bar(num_list, i)
                plt.show()
        else:
            for i in self.theta:
                temp = []
                i = list(i)
                for j in range(num):
                    if display == 0:
                        temp.append((i.index(max(i)), max(i)))
                    else:
                        temp.append(i.index(max(i)))
                    i[i.index(max(i))] = 0
                print(temp)
