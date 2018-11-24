from keras.utils import np_utils
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, \
    BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
import keras.backend as K
import numpy


K.set_image_data_format("channels_last")
K.set_learning_phase(1)

# 卷积残差块——convolutional_block
def Net_model(input_shape=(64, 64, 3), f=3, filters=(6, 12, 6), stage=1, block='a', s=2, classes=3):
    """
    param :
    X -- 输入的张量，维度为（m, n_H_prev, n_W_prev, n_C_prev）
    f -- 整数，指定主路径的中间 CONV 窗口的形状（过滤器大小，ResNet中f=3）
    filters -- python整数列表，定义主路径的CONV层中过滤器的数目
    stage -- 整数，用于命名层，取决于它们在网络中的位置
    block --字符串/字符，用于命名层，取决于它们在网络中的位置
    s -- 整数，指定使用的步幅
    return:
    X -- 卷积残差块的输出，维度为：(n_H, n_W, n_C)
    """

    X_input = Input(input_shape)
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # 定义基本的名字
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # 过滤器
    (F1, F2, F3) = filters

    # 保存输入值,后面将需要添加回主路径
    X_shortcut = X

    # 主路径第一部分
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding="valid",
               name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)

    # 主路径第二部分
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
               name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)

    # 主路径第三部分
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    # shortcut路径
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding="valid",
                        name=conv_name_base + "1", kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + "1")(X_shortcut)

    # 主路径最后部分,为主路径添加shortcut并通过relu激活
    X = layers.add([X, X_shortcut])
    X = Activation("relu")(X)

    # 最后阶段
    # 平均池化
    X = AveragePooling2D(pool_size=(2, 2))(X)

    # 输出层
    X = Flatten()(X)  # 展平
    X = Dense(classes, activation="softmax", name="fc" + str(3), kernel_initializer=glorot_uniform(seed=0))(X)

    # 创建模型
    model = Model(inputs=X_input, outputs=X, name="ResNet50")
    return model



class FeatureSelection(object):
    def __init__(self):
        self.tempSet1 = set([])  # 空集合   集合里存放的是不重复元素
        self.countDic1 = {}  # 空字典
        self.dictionary = {}  # 单词到ID号的字典
        self.dictionary1 = {}  # ID号到单词的字典
        self.vocabList = []
        self.vocabLong = 0

    def train(self, features):  # features是一个列表，里面的每一个元素是一个列表，是每一篇文本的所有单词（包括重复）
        self.getTF(features)        # 计算TF
        self.createDec()            # 建立词字典
        TFIDF = self.getTFIDF(features)     # 计算TFIDF
        return TFIDF

    # 计算TFIDF
    def getTFIDF(self, features):
        IDF = self.getIDF(features)
        TFIDF = np.zeros((len(features), img_rows * img_cols), dtype="float")
        for i in range(0, len(features)):
            for j in range(0, self.vocabLong):
                if str(self.dictionary1[j]) + '|Y=' + str(i) not in self.countDic1:
                    TFIDF[i][j] = 0
                else:
                    TFIDF[i][j] = float(
                        (self.countDic1[
                             str(self.dictionary1[j]) + '|Y=' + str(i)] / len(features[i])) * IDF[j])  # 未归一化
        TFIDF = self.normalize(TFIDF, features)
        return TFIDF

    # TFIDF归一化
    def normalize(self, TFIDF, features):
        for i in range(0, len(features)):
            norm = 0.0
            for j in range(0, self.vocabLong):
                norm += TFIDF[i][j] ** 2
            if norm != 0:
                norm = np.sqrt(norm)
            for j in range(0, self.vocabLong):
                TFIDF[i][j] = TFIDF[i][j] / norm  # 归一化
            # print('norm=' + str(norm))
        return TFIDF

    # 计算IDF
    def getIDF(self, features):
        IDF = np.zeros(self.vocabLong, dtype="int")
        DF = self.getDF(features)
        for w in self.tempSet1:
            # print(self.dictionary[w])
            IDF[self.dictionary[w]] = np.log(float((len(features) / (DF[self.dictionary[w]] + 1))))
        return IDF

    # 计算DF
    def getDF(self, features):
        DF = np.zeros(self.vocabLong, dtype="int")
        for w in self.tempSet1:  # w是一个单词
            for i in range(0, len(features)):
                tempstr = str(w) + '|Y=' + str(i)
                if tempstr in self.countDic1:
                    DF[self.dictionary[w]] += 1  # 某个单词共在多少篇文本中出现过
        return DF

    # 建立词字典
    def createDec(self):
        self.dictionary = dict(zip(self.vocabList, range(self.vocabLong)))
        self.dictionary1 = dict(zip(range(self.vocabLong), self.vocabList))

    # 计算TF
    def getTF(self, features):
        for i in range(0, len(features)):  # 有几个文本循环几次
            for j in range(0, len(features[i])):  # 第i个文本里有几个单词就循环几次
                self.tempSet1.add(str(features[i][j]))  # 不重复元素
                tempstr = str(features[i][j]) + '|Y=' + str(i)
                # print(tempstr)
                if tempstr in self.countDic1:
                    self.countDic1[tempstr] += 1  # 第i 篇文本中各个单词出现的次数 即TF
                else:
                    self.countDic1[tempstr] = 1
        self.vocabList = list(self.tempSet1)  # 将不重复单词转换为列表
        self.vocabLong = len(self.vocabList)
        print('Different words：' + str(self.vocabLong))


def load_data(a, labels):
    labelList = list(set(labels))
    label = []
    for i in labels:
        label.append(labelList.index(i))
    label = np.array(label)
    label = label.astype(np.int)

    train_data = a[:int(len(a)*0.8)]  # 取整行的数据
    train_label = label[:int(len(a)*0.8)]

    test_data = a[int(len(a)*0.8):]
    test_label = label[int(len(a)*0.8):]
    # test_data = a
    # test_label = label
    rval = [(train_data, train_label), (test_data, test_label)]

    return rval


img_rows, img_cols = 250, 90


df = pd.read_excel(r'C:\Users\111\Desktop\作业\2017年山东监测数据.xlsx')
labellist = list(df['媒体形式'])
if __name__ == '__main__':
    # the data, shuffled and split between train and test sets
    file = open("C:\\Users\\111\Desktop\作业\\3w数据集2.txt", 'r', encoding='utf-8')
    file = file.readlines()
    # 数据集特征集

    FS = FeatureSelection()

    # for i in range(20):
    feature = []
    labels = []
    for line in file[:]:  # 一行行读数据文件
        line = line.strip()
        tempVec = line.split(' ')
        # 将读进来的每一行按空格分开  一行代表一个文本
        label = tempVec.pop().strip()  # 将特征与标签分开
        labels.append(label)  # 将标签放入标签列表
        feature.append(tempVec)  # 将特征放入特征列表
        # features是一个列表，它的每一个元素都是一篇文本的所有单词集合 tempVec是一个列表

    print(feature[1])
    print(labels[1])

    nb_classes = len(set(labels))
    print('classes=' + str(nb_classes))
    a = FS.train(feature)
    # FS.train(feature)

    (X_train, y_train), (X_test, y_test) = load_data(a, labels)  # X=data,Y=label
    # print(X_train.shape[0])

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    Y_train = np_utils.to_categorical(y_train, nb_classes)  # Y_train是One-hot编码 320行 40列
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    # 运行构建的模型图
    model = Net_model(input_shape=(250, 90, 1), classes=3)

    # 编译模型来配置学习过程
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # 训练模型
    model.fit(X_train, Y_train, epochs=5, verbose=1, batch_size=40)

    # 测试集性能测试
    preds = model.evaluate(X_test, Y_test)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy =" + str(preds[1]))
