import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
import time
import sklearn
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
# 上面引入部分不报错说明相关的库导入都成功
# 如果有缺少的库 最万无一失的导入方式是点左上角的 File ->settings ->project -> project interpreter
# 弹出界面的右边有个anaconda的图标是 use conda package manager 点后会刷新
# 同在这个界面点右上方的 + 号 在新弹出的窗口中搜索并安装需要的包,如有需要也可以指定较低版本的安装

# 查看python版本
print(sys.version_info)
# 此刻我的运行结果仅供参考 sys.version_info(major=3, minor=6, micro=10, releaselevel='final', serial=0)


for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)
# 查看各种库的版本信息 下方为我的运行结果 仅供参考 win10 + python3.6环境
# matplotlib 3.1.2
# numpy 1.17.4
# pandas 0.25.3
# sklearn 0.22.1
# tensorflow 2.0.0
# tensorflow_core.keras 2.2.4-tf


# 测试运行
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# 运行结果
# (5000, 28, 28) (5000,)
# (55000, 28, 28) (55000,)
# (10000, 28, 28) (10000,)
