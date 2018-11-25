# Author:Stark-zheng
from collections import OrderedDict
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers import Conv2D
import random
np.random.seed(1337)  # for reproducibility

# nb_classes = 17  # 共17种文本
nb_epoch = 40  # 迭代的轮数
batch_size = 40  # 分批处理数据

# input image dimensions
img_rows, img_cols = 250, 90  # 数据尺寸   250   90
# number of convolutional filters to use
nb_filters1, nb_filters2 = 6, 12
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

def Net_model(lr=0.005, decay=0.005, momentum=0.9):
    model = Sequential()
    model.add(Conv2D(nb_filters1, (nb_conv, nb_conv), strides=(1, 1), padding='valid', input_shape=(img_rows, img_cols, 1)))
    print(model.output_shape)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Convolution2D(nb_filters2, (nb_conv, nb_conv)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1000))  # Full connection
    #  tanh    relu
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # sgd = SGD(lr=lr)
    # model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adadelta')

    return model


class FeatureSelection(object):
    def __init__(self):
        self.tempSet1 = set([])  # 空集合   集合里存放的是不重复元素
        self.countDic1 = OrderedDict()  # 空字典
        self.dictionary = OrderedDict()  # 单词到ID号的字典
        self.dictionary1 = OrderedDict()  # ID号到单词的字典
        self.vocabList = []
        self.vocabLong = 0

    def process(self, features):  # features是一个列表，里面的每一个元素是一个列表，是每一篇文本的所有单词（包括重复）
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
        print(self.vocabLong)


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
    rval = [(train_data, train_label), (test_data, test_label)]

    return rval


def train_model(model, X_train, Y_train):
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1)
    model.save_weights('model_weights_pro_other_2.h5', overwrite=True)
    return model
# batch_size=40  epochs=40


def test_model(model, X, Y):
    model.load_weights('model_weights_pro_other_2.h5')
    score = model.evaluate(X, Y, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return score

if __name__ == '__main__':
    # the data, shuffled and split between train and test sets
    file = open("C:\\Users\\111\Desktop\作业\\3w数据集2.txt", 'r', encoding='utf-8')
    file = file.readlines()
    # random.shuffle(file)
    print(len(file))
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

    nb_classes = len(set(labels))
    print('class=' + str(nb_classes))
    FS = FeatureSelection()
    a = FS.process(feature)
    # FS.train(feature)

    (X_train, y_train), (X_test, y_test) = load_data(a, labels)  # X=data,Y=label
    # print(X_train.shape[0])

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)  # Y_train是One-hot编码 320行 40列
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Net_model()
    train_model(model, X_train, Y_train)          # 如果准确率达到了一个令人满意的程度，可以将这一句注释掉，直接加载权重表，预测即可
    print("Y_test.shape", Y_test.shape)
    print(X_test[0].shape)
    print('真实值', Y_test[1])
    score = test_model(model, X_test, Y_test)

    model.load_weights('model_weights_pro_other_2.h5')
    classes = model.predict(X_test, verbose=0)
    print("output shape:", model.output_shape, model.output)
    # print('预测值', classes[0], Y_test[0])
    # print('预测值', classes[21], Y_test[21])

