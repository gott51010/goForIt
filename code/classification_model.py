# 分类模型
import tensorflow as tf
import numpy as np
import os
import sys
import time
import sklearn
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras

# 关于贝叶斯模型的一点入门知识  朴素贝叶斯分类器 Naive Bayes classifier 经常用于机器学习
# 参考url https://zhuanlan.zhihu.com/p/34277040
# https://baijiahao.baidu.com/s?id=1596461654133534773&wfr=spider&for=pc
# 贝叶斯统计学(ベイズ統計学)
# 比如你出门上班前想知道今天上班途中挤地铁时你身边站的是妹子还是汉子(贝叶斯统计学里叫主观概率)
# 以概率来说 因为男女性别的概率是50% 所以可能有一些人可能会认为就是50%
# 但从生活经验上来说是不对的 更科学一点的概率会再追加参数
# 比如你所在地区的男女分布并不是50%也许是49:51 ?  会和你坐同一条线路挤地铁上班的社畜人群中 男女比例也许是45:55? 甚至还要考虑到你出门的时间 妹子的出现概率
# 然而这一样来需要的数据就太大了
# 所以当我们选用贝叶斯统计时 一开始就只需要假设就是50%然后去上班就好(贝叶斯统计学里叫事前概率)
# 假设第一天遇见的是萌妹子 ok此时的概率就可以变更为100%了
# 然后之后连续9天 每天都是汉子 所以这个数据每天都被修正 第二天就是50% 第三天33.333% 以此类推...
# 到了第十天就会成10% 看起来是个比较科学的数值(贝叶斯统计学里叫事后概率)
# 当得到新情报后把其加入以前的样本再重新推算概率 这就是贝叶斯定理活跃于机器学习领域的特点
#

# 临时对应cpu报错问题
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 把helloTensorflow里的东西抄过来接着用
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


def show_single_image(img_arr):
    plt.imshow(img_arr, cmap="binary")
    plt.show()


# show_single_image(x_train[0])


def show_imgs(n_rows, n_cols, x_data, y_data, class_names):
    assert len(x_data) == len(y_data)
    assert n_rows * n_cols < len(x_data)
    plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index+1)
            plt.imshow(x_data[index], cmap="binary", interpolation='nearest')
            plt.axis('off')
            plt.title(class_names[y_data[index]])
    plt.show()


class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

show_imgs(3, 5, x_train, y_train, class_names)


# 干货
# 从官方文档上抄这个方法 Sequential() https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Sequential
#


model = tf.keras.models.Sequential()
# 这里写成input_shape[28, 28]的话会报错 具体参照文档
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation="softmax"))

# relu: y = max(0, x)
# softmax: 将向量变成概率分布

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
