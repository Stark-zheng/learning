import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy
import pandas as pd
from PIL import Image
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers import Conv2D
numpy.random.seed(1337)

nb_classes = 7  # 共7个种类
nb_epoch = 10  #迭代的轮数
batch_size = 40

#  图像的尺寸
img_rows, img_cols = 300, 250
# 使用卷积滤波器的数量
nb_filters1, nb_filters2 = 20, 40
# 最大池的池面积大小
nb_pool = 3
# 卷积核大小
nb_conv = 3


def Net_model(lr=0.01, decay=1e-6, momentum=0.9):
    model = Sequential()
    model.add(Conv2D(nb_filters1, (nb_conv, nb_conv), strides=(1, 1),
                     padding='valid', input_shape=(img_rows, img_cols, 4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters2, (nb_conv, nb_conv)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Flatten())
    model.add(Dense(1000))  # Full connection

    model.add(Activation('relu'))
    model.add(Dense(nb_classes))  # 7类
    model.add(Activation('softmax'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

    return model


def make_data(path):
    a = {'neutral': 0, 'angry': 1, 'disgusted': 2, 'fearful': 3,
         'happy': 4, 'sad': 5, 'surprised': 6}

    imageList = os.listdir(path)
    count = len(imageList)
    labels = []
    for item in imageList:
        i = item.split('-')[0]
        labels.append(a[i])
    TrainList = []
    labels = numpy.array(labels).T
    for i in range(count):
        string = str(imageList[i])
        List = mpimg.imread(path + '\\' + string)
        TrainList.append(List)
    faces = numpy.empty((count, 300000))

    for i in range(len(TrainList)):
        a = numpy.asarray(TrainList[i], dtype='float64') / 256
        faces[i] = a.reshape(-1)

    rval = (faces, labels)

    return rval


def train_model(model, X_train, Y_train):
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1)
    model.save_weights('model_weights_pro_other.h5', overwrite=True)
    return model


def test_model(model, X, Y):
    model.load_weights('model_weights_pro_other.h5')
    score = model.evaluate(X, Y, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return score


if __name__ == '__main__':
    trainPath = r'E:\train'
    testPath = r'E:\test'
    (X_train, y_train) = make_data(trainPath)
    (X_test, y_test) = make_data(testPath)
    print('train:',X_train.shape[0])
    print('test:',X_test.shape[0])

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 4)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 4)

    Y_train = np_utils.to_categorical(y_train, nb_classes)  # Y_train是One-hot编码 320行 40列
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Net_model()
    train_model(model, X_train, Y_train)
    score = test_model(model, X_test, Y_test)

    model.load_weights('model_weights_pro_other.h5')
    classes = model.predict(X_test, verbose=0)
    print("output shape:", model.output_shape, model.output)
    a = {'neutral': 0, 'angry': 1, 'disgusted': 2, 'fearful': 3,
         'happy': 4, 'sad': 5, 'surprised': 6}
    key = list(a.keys())
    print(key[list(Y_test[20]).index(1)])
    # print('预测值', Y_test[20])
    # print('预测值', Y_test[50])
